import rasterstats as rsts
import geopandas as gpd

from src.tools import input_reading
from src.tools import list_dics_functions as ldfun

def shape_shape_zonal_stats (
        geo_data_file_path,
        data_file,
        polygon_id_field,
        input_field,
        zonal_stats_value
):
    """
        Compute zonal statistics between two polygon shapefiles.

        For each polygon in the spatial distribution layer, the function
        intersects it with polygons from a data layer and computes
        a zonal statistic (mean, max, or sum) based on the specified field.

        PARAMETERS
        ----------
        geo_data_file_path : str or Path
            Path to the shapefile defining spatial zones (aggregation polygons).

        data_file : str or Path
            Path to the shapefile containing polygons with attribute data.

        polygon_id_field : str
            Field name in the spatial distribution layer used as unique ID
            for each zone.

        input_field : str
            Field name in the data layer from which values are extracted.

        zonal_stats_value : str
            Type of zonal statistic to compute:
            - 'mean' → area-weighted mean
            - 'max' → maximum value
            - 'sum' → sum of intersecting values

        RETURNS
        -------
        dict
            Dictionary mapping each zone ID to the computed zonal statistic.
        """
    # --------------------------------------------------------------
    # 1. Read input shapefiles
    # --------------------------------------------------------------
    spacial_distrib_gdf=gpd.read_file(geo_data_file_path)
    data_gdf=gpd.read_file(data_file)

    zonal_stats={}
    # --------------------------------------------------------------
    # 2. Loop through each spatial aggregation zone
    # --------------------------------------------------------------
    for index, spatial_zone in spacial_distrib_gdf.iterrows():
        intersections = []
        # ----------------------------------------------------------
        # 3. Compare with every polygon in data layer
        # ----------------------------------------------------------
        for index2, polygon in data_gdf.iterrows():
            # --------------------------------------------------
            # MEAN (area-weighted)
            # --------------------------------------------------
            if zonal_stats_value == 'mean':
                if spatial_zone['geometry'].intersects(polygon['geometry']):
                    intersections.append([spatial_zone['geometry'].intersection(polygon['geometry']).area,
                                          polygon[input_field]])
                if sum(x for x, y in intersections) == 0:
                    zonal_stats[spatial_zone[polygon_id_field]] = 0
                else:
                    zonal_stats[spatial_zone[polygon_id_field]] = (
                            sum(x * y for x, y in intersections) / sum(x for x, y in intersections))
            # --------------------------------------------------
            # MAX
            # --------------------------------------------------
            if zonal_stats_value == 'max':
                if spatial_zone['geometry'].intersects(polygon['geometry']):
                    intersections.append(polygon[input_field])
                if sum(x for x in intersections) == 0:
                    zonal_stats[spatial_zone[polygon_id_field]] = 0
                else:
                    zonal_stats[spatial_zone[polygon_id_field]] = max(
                        intersections)
            # --------------------------------------------------
            # SUM
            # --------------------------------------------------
            if zonal_stats_value == 'sum':
                if spatial_zone['geometry'].intersects(polygon['geometry']):
                    intersections.append(polygon[input_field])
                zonal_stats[spatial_zone[polygon_id_field]]=sum(x for x in intersections)
    return zonal_stats


def shape_raster_zonal_stats(
        spatial_units_data,
        values_data,
        polygon_id_field,
        zonal_stats_method,
        zonal_stats_value
):
    """
        Compute zonal statistics of a raster over polygon spatial units
        and return the results as a dictionary indexed by polygon ID.

        PARAMETERS
        ----------
        spatial_units_data : str or Path
            Path to the polygon shapefile defining the spatial units.

        values_data : str or Path
            Path to the raster containing the values to summarize.

        polygon_id_field : str
            Name of the polygon attribute used as a unique identifier.

        zonal_stats_method : str
            Method controlling raster–polygon interaction:
                - 'centers' → pixel counted if its center is inside polygon
                - 'all touched' → any pixel touched by polygon is counted

        zonal_stats_value : str
            Statistic to compute from raster values inside each polygon
            (e.g., 'mean', 'max').

        RETURNS
        -------
        dict
            Dictionary indexed by polygon ID containing the computed
            zonal statistic for each spatial unit.
        """
    # ------------------------------------------------------------------
    # 1. Compute zonal statistics depending on selected method
    # ------------------------------------------------------------------
    gdf = gpd.read_file(str(spatial_units_data))
    zonal_stats=None
    if zonal_stats_method=='centers':
        zonal_stats = rsts.zonal_stats(
            gdf,
            str(values_data),
            stats=[zonal_stats_value])
    elif zonal_stats_method=='all touched':
        zonal_stats = rsts.zonal_stats(
            str(spatial_units_data),
            str(values_data),
            all_touched=True,
            stats=[zonal_stats_value])
    # ------------------------------------------------------------------
    # 2. Clean and round results
    #    - Replace None values with 0
    #    - Round statistic to 3 decimal places
    # ------------------------------------------------------------------
    for item in zonal_stats:
        if item[zonal_stats_value] is None:
            item[zonal_stats_value] = 0
        item[zonal_stats_value] = round(item[zonal_stats_value], 3)

    # ------------------------------------------------------------------
    # 3. Convert polygon shapefile IDs into dictionary structure
    # ------------------------------------------------------------------
    zonal_stats_dic,_=input_reading.shp_to_dic(spatial_units_data,[polygon_id_field])

    # ------------------------------------------------------------------
    # 4. Merge zonal statistics list into dictionary of polygons
    # ------------------------------------------------------------------
    ldfun.add_listofdics_to_dic(zonal_stats_dic, zonal_stats)

    return zonal_stats_dic