import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import label
from pathlib import Path
from scipy.spatial import cKDTree

from tools import input_reading
from tools import outputs
from tools import raster_tools

def bathtub_module(twl_dic, coastlines_file_name, crs, input_type,mdt_filename=None,idw_files=False):
    """
    Main module to compute coastal flooding using a bathtub approach.

    This function loads the DEM raster and coastline data, reprojects them
    to a common CRS, generates a coastal buffer, and rasterizes it. Depending
    on the input type, it may interpolate Total Water Level (TWL) values using
    IDW before computing flooding.

    Parameters
    ----------
    twl_dic : dict
        Dictionary containing Total Water Level (TWL) data. It may include
        direct values per scenario or references to point data for interpolation.
    coastlines_file_name : str
        Name of the coastline shapefile (or GeoJSON).
    crs : rasterio.crs.CRS or str
        Target Coordinate Reference System for all spatial data.
    input_type : str
        Type of TWL input. If 'multi value', IDW interpolation is applied.
    mdt_filename : str, optional
        Specific DEM filename to process. If None, it searches for .tif files in 'exp_input_data'.
    idw_files : bool, optional
        If True, the interpolated IDW rasters will be saved to disk (default is False).

    Returns
    -------
    None
        Outputs flooding rasters to disk.
    """
    # Read DEM raster file path
    if mdt_filename is None:
        dem_file_path = input_reading.reading_folder_files('exp_input_data', '.tif')
        dem_file_path=next(iter(dem_file_path.values()))
    else:
        dem_file_path = Path.cwd() / 'data'/ mdt_filename

    # Build the coastline file path and load it as GeoDataFrame
    coastlines_file_path = Path.cwd() / 'data'/ coastlines_file_name
    coastlines_gdf = gpd.read_file(coastlines_file_path)

    # Reproject coastline to target CRS
    coastlines_gdf = coastlines_gdf.to_crs(crs)

    # Reproject DEM raster to target CRS
    dem_raster, meta = raster_tools.reproject_raster_crs(
        dem_file_path, crs
    )

    # Create a coastal buffer (25 meters inland)
    coast_buffer = coastlines_gdf.buffer(25)

    # Rasterize the buffered coastline (1 = coastal zone, 0 = elsewhere)
    coast_raster = rasterize(
        [(geom, 1) for geom in coast_buffer.geometry],
        out_shape=dem_raster.shape,
        transform=meta['transform'],
        fill=0
    )

    # If TWL input is spatial (points), interpolate using IDW
    if input_type == 'multi value':
        twl_dic = idw_submodule(twl_dic, idw_files, meta, mdt_filename)

    # Run flooding computation
    flooding_submodule(twl_dic, dem_raster, coast_raster, meta,mdt_filename)

def flooding_submodule(twl_dic, dem_raster, coast_raster, meta, mdt_filename):
    """
    Compute flooding extent and depth based on TWL and DEM.

    This function applies a bathtub flooding approach by comparing DEM elevation
    with Total Water Level (TWL), identifying connected flooded regions linked
    to the coastline, and computing water depth.

    Parameters
    ----------
    twl_dic : dict
        Dictionary where keys are scenario names and values are TWL values
        (scalar or raster).
    dem_raster : numpy.ndarray
        Digital Elevation Model as a 2D array.
    coast_raster : numpy.ndarray
        Binary raster indicating coastal buffer areas (1 = coast, 0 = non-coast).
    meta : dict
        Raster metadata.
    mdt_filename : str
        Filename of the MDT used to be included in the output filename.

    Returns
    -------
    None
        Writes flooding depth rasters to disk.
    """
    # Update metadata for output rasters
    meta.update(dtype="float32", nodata=np.nan)

    for scen, twl in twl_dic.items():
        # Create mask where DEM elevation is below TWL
        flood_mask = dem_raster < twl

        # Label connected components in the flood mask
        components, num = label(flood_mask)

        # Identify components that intersect the coastal buffer
        coastal_connected_components = np.unique(components[coast_raster == 1])

        # Keep only flood regions connected to the coast
        connected_flood = np.isin(components, coastal_connected_components)

        # Compute flood depth (TWL - DEM) only for connected regions
        flooded_pixels_depth = np.where(connected_flood, twl - dem_raster, np.nan)

        # Remove negative or zero depths
        flooded_pixels_depth = np.where(flooded_pixels_depth > 0, flooded_pixels_depth, np.nan)

        # Save output raster
        (Path.cwd() / 'outputs' / 'flood_maps').mkdir(parents=True, exist_ok=True)
        if mdt_filename is None:
            outputs.tif_output(f"flooding_{scen}", flooded_pixels_depth, meta)
            print(scen)
        else:
            name = mdt_filename.split("MDT_")[1].split(".tif")[0]
            outputs.tif_output(f"flood_maps/flooding_{name}_{scen}", flooded_pixels_depth, meta)

            print(name+'_'+scen)
            
def idw_submodule(twl_dic,idw_files,meta,mdt_file_name,power=2, k=12, chunk_size=100_000):
    """
    Interpolate Total Water Level (TWL) values over a raster grid using IDW.

    This function performs Inverse Distance Weighting (IDW) interpolation from
    point-based TWL data to a raster grid defined by metadata.

    Parameters
    ----------
    twl_dic : dict
        Dictionary containing:
        - 'file_name': name of the shapefile with TWL points
        - 'scens_names': list of scenario names (fields in the shapefile)
    idw_files : bool
        If True, saves the interpolated IDW rasters to disk.
    meta : dict
        Raster metadata containing transform, width, height, and CRS.
    mdt_file_name : str
        Name of the MDT file to use for naming the output IDW files.
    power : int, optional
        Power parameter for IDW weighting (default is 2).
    k : int, optional
        Number of nearest neighbors to use (default is 12).
    chunk_size : int, optional
        Number of grid points processed per iteration (default is 100 thousand).

    Returns
    -------
    new_twl_dic : dict
        Dictionary with interpolated TWL rasters (2D arrays) per scenario.
    """
    width = meta['width']
    height = meta['height']
    transform = meta['transform']

    # Create grid coordinates (pixel centers in spatial reference)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)  # type: ignore
    xs = np.array(xs)
    ys = np.array(ys)

    # Flatten the grid into a list of (x, y) points
    grid_points = np.column_stack((xs.ravel(), ys.ravel()))

    # Load TWL point data
    twl_points_file_path = Path.cwd() / f"outputs/{twl_dic['file_name']}"
    twl_gdf = gpd.read_file(twl_points_file_path)
    
    k = min(k, len(twl_gdf))

    # Reproject points to match raster CRS
    twl_gdf = twl_gdf.to_crs(meta['crs'])

    # Extract point coordinates
    points = np.array([(geom.x, geom.y) for geom in twl_gdf.geometry])

    # Build a KDtree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Initialize output dictionary with empty arrays
    new_twl_dic = {
        scen: np.empty(grid_points.shape[0], dtype=np.float32)
        for scen in twl_dic['scens_names']
    }

    # Process grid points in chunks to reduce memory usage
    for i in range(0, len(grid_points), chunk_size):
        chunk = grid_points[i:i + chunk_size]

        # Find k nearest neighbors and distances
        dist, idx = tree.query(chunk, k=k)

        # Avoid division by zero for coincident points
        dist[dist == 0] = 1e-10

        # Compute IDW weights
        weights = (1 / dist ** power).astype(np.float32)
        if weights.ndim == 1:
            weights = weights[:, None]
        if idx.ndim == 1:
            idx = idx[:, None]
        sum_weights = np.sum(weights, axis=1).astype(np.float32)

        for scen in twl_dic['scens_names']:
            # Extract scenario values from GeoDataFrame
            values = twl_gdf[scen].values.astype(np.float32)

            # Compute weighted average for the chunk
            chunk_results = np.sum(weights * values[idx], axis=1) / sum_weights

            # Store results in the output array
            new_twl_dic[scen][i:i + chunk_size] = chunk_results

    # Reshape flat arrays back to 2D grids and save rasters
    for scen, grid in new_twl_dic.items():
        grid_2d = grid.reshape((height, width))
        new_twl_dic[scen] = grid_2d
        
        if idw_files:
            (Path.cwd() / 'outputs' / 'idw').mkdir(parents=True, exist_ok=True)
            # Save interpolated raster
            if mdt_file_name is None:
                outputs.tif_output(f"idw/{scen}_idw", grid_2d, meta)
            else:
                name = mdt_file_name.split("MDT")[1].split(".tif")[0]
                outputs.tif_output(f"idw/{name}_{scen}_idw", grid_2d, meta)

    return new_twl_dic
