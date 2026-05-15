import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def reproject_raster_crs(input_path, crs):
    """
    Reproject a raster to a new Coordinate Reference System (CRS).

    This function reads a raster file, calculates the appropriate transformation
    for a target CRS, and reprojects the raster data accordingly. The output
    raster data is returned as a NumPy array, with nodata values converted to NaN,
    along with updated metadata.

    Parameters
    ----------
    input_path : str
        Path to the input raster file.
    crs : rasterio.crs.CRS or str
        Target Coordinate Reference System to reproject the raster into.

    Returns
    -------
    data : numpy.ndarray
        Reprojected raster data as a 2D array, with nodata values replaced by NaN.
    metadata : dict
        Updated raster metadata including CRS, transform, width, and height.

    Notes
    -----
    - Uses nearest neighbor resampling.
    - Assumes a single-band raster.
    - The output array shape is determined by the target CRS transformation.
    """
    with rasterio.open(input_path) as src:
        # Compute the new transform, width, and height for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds
        )

        # Create an empty destination array to store reprojected data
        data = np.empty((height, width), dtype=src.dtypes[0])

        # Perform the reprojection from source to the destination array
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest
        )

        # Replace nodata values with NaN for easier analysis
        data = np.where(data == src.nodata, np.nan, data)

        # Copy and update metadata to reflect the new CRS and geometry
        metadata = src.meta.copy()
        metadata.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

    return data, metadata