from pathlib import Path
import geopandas as gpd
import csv
import rasterio as ras

from src.tools import dictionaries as dics
from src.tools import input_reading
from src.level_3_risk_analysis import level_3_analysis


def shapefile_output(
        filename,
        dic,
        crs,
        partial_agg_flag: bool | str
):
    """
        Create and export a shapefile from a dictionary of dictionaries
        containing impact assessment results.

        The function:
        1. Converts the input dictionary-of-dictionaries into a dictionary-of-lists
           suitable for GeoPandas.
        2. Builds a GeoDataFrame.
        3. Orders columns consistently (including hazard-related fields).
        4. Assigns a CRS.
        5. Saves the result as a shapefile.

        6. Optionally performs an external spatial aggregation by intersecting
       the output geometries with a zoning shapefile and assigning a
       Section identificator to each element.
       (It is done here because it's the only place in the code where there is a
       gdf with geometries.)

        PARAMETERS
        ----------
        filename : str
            Name of the output file (without extension).

        dic : dict of dict
            Dictionary where:
            - Keys = element IDs
            - Values = sub-dictionaries containing attributes and geometry

        crs : str or CRS
            Coordinate reference system to assign to the GeoDataFrame.

        partial_agg_flag : str
        If equal to 'external', performs spatial join with an external
        zoning shapefile and assigns section identifiers.

        RETURNS
        -------
        None
            The function writes the shapefile to disk.
        """
    # --------------------------------------------------------------
    # 1. Define output path
    # --------------------------------------------------------------
    namefile=filename+'.geojson'
    path = Path.cwd() / 'outputs/shps' / namefile

    path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 2. Convert dictionary-of-dictionaries → dictionary-of-lists
    #    (required structure for GeoDataFrame creation)
    # --------------------------------------------------------------
    columns_dic = {key: [] for key in dic[list(dic.keys())[0]].keys()}
    columns_dic[dics.keysdic['Elements ID']]=[]
    for key,row in dic.items():
        columns_dic[dics.keysdic['Elements ID']].append(key)
        for column, value in row.items():
            columns_dic[column].append(value)

    # --------------------------------------------------------------
    # 3. Create GeoDataFrame
    # --------------------------------------------------------------
    gdf = gpd.GeoDataFrame(columns_dic, geometry='geometry')

    # --------------------------------------------------------------
    # 4. Define column ordering
    # --------------------------------------------------------------
    # Base output fields
    keys_list=[dics.keysdic['Elements ID'], dics.keysdic['Type of system'],
               dics.keysdic['Exposed value'], dics.keysdic['Impact scenario'],
               dics.keysdic['Damage function'], dics.keysdic['Impact damage'],'geometry']
    # Identify hazard-related fields dynamically
    haz_keys = [k for k in next(iter(dic.values())).keys() if k not in keys_list]
    # Insert hazard keys before damage function column
    keys_list[4:4] = haz_keys
    # Reorder GeoDataFrame columns
    gdf = gdf[keys_list]

    # --------------------------------------------------------------
    # 5. Assign CRS
    # --------------------------------------------------------------
    gdf=gdf.set_crs(crs)

    # --------------------------------------------------------------
    # 5. Assign CRS
    # --------------------------------------------------------------
    gdf.to_file(path)

    # --------------------------------------------------------------
    # 7. Optional external spatial aggregation
    # --------------------------------------------------------------
    if partial_agg_flag=='external':
        # --- 1. Cargar las zonas desde el shapefile ---
        partial_agg_map_path = input_reading.reading_folder_files('spatial_distribution_input', '.shp')
        key = list(partial_agg_map_path.keys())[0]
        partial_agg_map_path = partial_agg_map_path[key]
        zones = gpd.read_file(partial_agg_map_path)

        # Asegúrate de que el CRS de las zonas sea compatible con tus polígonos
        # (por ejemplo, EPSG:4326 o el que uses)
        zones = zones.to_crs(crs)

        # --- 4. Hacer el cruce espacial (spatial join) ---
        joined = gpd.sjoin(gdf, zones, how="left", predicate="within")

        # --- 5. Añadir el nombre de zona a cada subdiccionario ---
        column=dics.keysdic["Section identificator"]
        for key, row in joined.iterrows():
            dic[row.get(dics.keysdic['Elements ID'])][column] = row.get(column)

def csv_output(filename, fields, new_field_names, dic):
    namefile = filename + '.csv'
    path = Path.cwd() / 'outputs/csvs' / namefile
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode='w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=new_field_names,
            delimiter=';'
        )
        writer.writeheader()

        for key, sub_dict in dic.items():
            row = {new_field_names[0]: key}
            for i in range(1, len(new_field_names)):
                if (
                    new_field_names[i] in dics.keysoutputdic.values()
                    or new_field_names[i] in dics.keysoutputdic['Exposed value'].values()
                    or new_field_names[i] in dics.keysoutputdic['Impact damage'].values()
                ):
                    mapped_key = dics.keysdic.get(fields[i])
                    if mapped_key is None or mapped_key not in sub_dict:
                        continue
                    row[new_field_names[i]] = sub_dict[mapped_key]
                else:
                    if fields[i] not in sub_dict:
                        continue
                    row[new_field_names[i]] = sub_dict[fields[i]]
            writer.writerow(row)

def tif_output(filename, results,kwargs):
    """
       Export a NumPy array as a GeoTIFF raster.

       Parameters:
       - filename: name of the output file (without extension).
       - results: NumPy array containing raster values.
       - kwargs: metadata dictionary (CRS, transform, width, height, etc.)
                 used to define the output raster.
       """
    # Build output file path
    namefile = filename + '.tif'
    path = Path.cwd()/ 'outputs' / namefile
    # Write raster using metadata
    with ras.open(path, 'w', **kwargs) as output:
        output.write(results, 1)

def listofdics_to_csv(listofdics,fields,new_field_names,path):
    """
        Export a list of dictionaries to a CSV file.

        Each dictionary in the list represents one row.
        Field names are mapped using the internal keys dictionary.

        Parameters:
        - listofdics: list of dictionaries containing results.
        - fields: list of logical field names used internally.
        - new_field_names: list of column headers for CSV output.
        - path: full path to the output CSV file.
        """

    with open(path, mode='w', newline='') as file:
        # Initialize CSV writer with semicolon delimiter
        writer = csv.DictWriter(
            file,
            fieldnames=new_field_names,
            delimiter=';'
        )
        # Write header row
        writer.writeheader()
        # Write each dictionary as a row
        for dic in listofdics:
            row = {}
            # Map internal field names to output column names
            for i in range(len(new_field_names)):
                row[new_field_names[i]] = dic[
                    dics.keysdic[fields[i]]
                ]
            writer.writerow(row)

def summary_output(summarydic):
    """
        Export overall summary results to a CSV file.

        The function:
        1. Defines the output path for the summary CSV.
        2. Specifies the human-readable output fields.
        3. Maps those fields to their internal dictionary keys.
        4. Writes the results to CSV using `listofdics_to_csv`.

        PARAMETERS
        ----------
        summarydic : list of dict
            List containing aggregated summary results.
            Each entry is a dictionary with summary metrics.

        RETURNS
        -------
        None
            Writes 'results_summary.csv' to the results/csvs folder.
        """
    # --------------------------------------------------------------
    # 1. Define output path
    # --------------------------------------------------------------
    path = Path.cwd()/ 'outputs/csvs/results_summary.csv'
    # --------------------------------------------------------------
    # 2. Define output fields (human-readable names)
    # --------------------------------------------------------------
    fields = ['Exposed system', 'Type of system', 'Exposed value summary', 'Impact scenario',
              'Impact damage summary']
    # --------------------------------------------------------------
    # 3. Map output field names to internal dictionary keys
    # --------------------------------------------------------------
    new_field_names=level_3_analysis.output_fields_keys(fields,summarydic)
    # --------------------------------------------------------------
    # 4. Export dictionary-of-dictionaries to CSV
    # --------------------------------------------------------------
    listofdics_to_csv(summarydic,fields,new_field_names,path)

def partial_agg_output(partialaggdic):
    """
        Export partial aggregation results (e.g., by section/zone)
        to a CSV file.

        The function:
        1. Defines the output path for partial aggregation results.
        2. Specifies the relevant output fields including Section identificator.
        3. Maps human-readable fields to internal keys.
        4. Writes the results to CSV.

        PARAMETERS
        ----------
        partialaggdic : list of dict
            List containing partially aggregated results,
            typically grouped by Section identificator.

        RETURNS
        -------
        None
            Writes 'partial_agg_result.csv' to the results/csvs folder.
        """
    # --------------------------------------------------------------
    # 1. Define output path
    # --------------------------------------------------------------
    path = Path.cwd().parent.parent.parent / 'results/csvs/partial_agg_result.csv'
    # --------------------------------------------------------------
    # 2. Define output fields (includes Section identificator)
    # --------------------------------------------------------------
    fields = ['Exposed system', 'Type of system', 'Section identificator', 'Exposed value summary',
              'Impact scenario', 'Impact damage summary']
    # --------------------------------------------------------------
    # 3. Map output field names to internal dictionary keys
    # --------------------------------------------------------------
    new_field_names = level_3_analysis.output_fields_keys(fields,partialaggdic)
    # --------------------------------------------------------------
    # 4. Export dictionary-of-dictionaries to CSV
    # --------------------------------------------------------------
    listofdics_to_csv(partialaggdic, fields, new_field_names, path)

def simple_csv_output(filename,geo_data_polygon_id_field,dic):
    #Export the dictionary with the results of the risk analysis to a .csv file. Each entry is in a single row,
    #they keys are in the first, and the keys of the sub dictionaries are the headers.

    namefile = filename + '.csv'
    path = Path.cwd().parent.parent.parent / 'results/csvs' / namefile
    headers = list(next(iter(dic.values())).keys())
    headers.remove('geometry')
    headers = [geo_data_polygon_id_field] + headers

    with open(path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
        # Write the header row (field output headers)
        writer.writeheader()
        # Write each row (each inner dictionary as a row)
        for key, sub_dict in dic.items():
            row = {geo_data_polygon_id_field: key, **sub_dict}
            row = {k: row.get(k, "") for k in headers}
            writer.writerow(row)

def simple_shapefile_output(filename,geo_data_polygon_id_field,dic,crs):
    #Create a new shapefile adding the impact scenario, impact value, damage fraction and consequences value

    namefile=filename+'.shp'
    path = Path.cwd().parent.parent.parent / 'results/shps' / namefile

    #Transform dic of dics into dic of lists. Each list contains the values of a given key from all the subdics
    columns_dic = {key: [] for key in dic[list(dic.keys())[0]].keys()}
    columns_dic[geo_data_polygon_id_field]=[]
    for key,row in dic.items():
        columns_dic[geo_data_polygon_id_field].append(key)
        for column, value in row.items():
            columns_dic[column].append(value)

    #Create a geodataframe from the dic of lists
    gdf = gpd.GeoDataFrame(columns_dic, geometry='geometry')

    #Define the crs for the file
    gdf=gdf.set_crs(crs)

    #Save the gdf a shapefile
    gdf.to_file(path)

