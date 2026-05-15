import geopandas as gpd
from pathlib import Path
import csv
import fiona
import copy

def reading_external_files(file_path, folder,geo_data_polygon_id_field,geo_dic=None,):
    """
    Read external data files (CSV or Shapefile) and convert them to internal dictionary format.

    This function constructs the absolute path to a file within the 'inputs' directory,
    determines its type by extension, and processes it into a dictionary indexed by
    element IDs.

    Parameters
    ----------
    file_path : str
        The name or relative path of the file to read.
    folder : str
        The subfolder name within the 'inputs' directory where the file is located.
    geo_dic : dict, optional
        A geographic dictionary template required for CSV processing (default is None).
    :param geo_data_polygon_id_field: The field name in the geospatial file is used to identify
        individual polygons for external input additions.
    :type geo_data_polygon_id_field: str or None

    Returns
    -------
    dict
        A dictionary containing the processed indicator data, indexed by element IDs.

    Raises
    ------
    ValueError
        If the file extension is not supported (only .csv and .shp are allowed).
    """
    # Extract extension and name from the filename
    ext = '.' + file_path.split('.')[-1]

    # Construct the absolute path to the input file relative to the project structure

    # Project structure: root / inputs / {folder} / {file_path}
    file_path_abs = Path.cwd().parent.parent.parent / 'inputs' / folder / file_path

    # Read external data and convert it to the internal dictionary format
    if ext == '.csv':
        # Convert CSV to dictionary using the provided geographic template
        indic_indiv_dic = csv_to_dic(file_path_abs, geo_dic)
    elif ext == '.shp':
        # For shapefiles, use Elements ID as index and convert to dictionary
        # The dictionary structure is {ID: {attribute: value, ...}}
        indic_indiv_dic = gpd.read_file(file_path_abs).set_index(geo_data_polygon_id_field).T.to_dict('dict')
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return indic_indiv_dic
def reading_folder_files(folder_name,extensions):
    """
        Search recursively inside a predefined inputs folder and return
        all files matching the given extensions.

        PARAMETERS
        ----------
        folder_name : str
            Name of the subfolder inside the global 'inputs' directory.

        extensions : tuple or str
            File extensions to filter (e.g., '.shp' or ('.shp', '.tif')).

        RETURNS
        -------
        dict
            Dictionary where:
            - key   : filename without extension
            - value : absolute Path to the file
        """
    # --------------------------------------------------------------
    # 1. Build the absolute path to the target folder
    #    Assumes project structure:
    #    project_root/
    #        inputs/
    #            <folder_name>/
    # --------------------------------------------------------------
    folder_path=Path.cwd()/folder_name
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder_path}")
    # --------------------------------------------------------------
    # 2. Recursively search for files matching the desired extensions
    # --------------------------------------------------------------
    files = [file for file in folder_path.rglob('*') if file.is_file() and file.name.endswith(extensions)]
    # --------------------------------------------------------------
    # 3. Create a dictionary:
    #    key → filename without extension
    #    value → absolute resolved path
    # --------------------------------------------------------------
    files_dic = {file.stem: file.resolve() for file in files}

    return files_dic

def reading_files(folder,extensions):
    """
    Reads files from a specified folder with specified extensions and extracts metadata.

    This function retrieves all files in the given folder that match the provided list
    of extensions. It returns a dictionary containing metadata for each file. For
    shapefiles (`.shp`), additional metadata such as the coordinate reference system
    (CRS) is included.

    :param folder: The folder path from which files will be retrieved.
    :type folder: str
    :param extensions: A list or tuple of file extensions to filter files in the folder.
    :type extensions: list[str] | tuple[str, ...]
    :return: A dictionary of filenames as keys and associated metadata as values. Metadata
        includes the file path, file extension, and optionally the CRS (for `.shp` files).
    :rtype: dict
    """

    # --------------------------------------------------------------
    # 1. Retrieve files in the folder with the desired extensions
    # --------------------------------------------------------------
    files_dic = reading_folder_files(folder, extensions)
    # --------------------------------------------------------------
    # 2. Build an extended dictionary with metadata
    # --------------------------------------------------------------
    extended_dic={}
    for name, path in files_dic.items():
        # Case A: Shapefile → include CRS information
        if path.suffix=='.shp' or path.suffix=='.geojson':
            with fiona.open(path) as src:
                extended_dic[name]={'path':path, 'crs':src.crs, 'extension':path.suffix}
        # Case B: Other raster/vector types → only store basic info
        else:
            extended_dic[name] = {'path': path, 'extension': path.suffix}
    return extended_dic

def reading_shp_to_dic(folder, keys):
    """
    Reads shapefile data from a specified folder into a dictionary, using the provided keys for
    data extraction. The first shapefile found in the folder is processed.

    :param folder: The path to the directory containing the shapefiles
        from which data will be read.
    :type folder: str
    :param keys: The list of field names to extract from the shapefile. The `geometry`
        field will automatically be included.
    :type keys: list[str]
    :return: A tuple containing the dictionary representation of the shapefile data
        and the file path of the processed shapefile.
    :rtype: tuple[dict, str]
    """
    # Retrieve all shapefiles in the specified folder
    files_dic = reading_folder_files(folder, '.shp')

    # Select the first shapefile found
    key = list(files_dic.keys())[0]
    file = files_dic[key]

    # Ensure geometry is included in the extracted fields
    keys.append('geometry')

    # Convert shapefile to dictionary using selected keys
    return shp_to_dic(file, keys), file

def shp_to_dic(file,keys):
    """
    Converts a specified set of columns from a shapefile into a nested dictionary,
    where the first key from the provided column names is used as the index for
    unique identification. The operation also retains the Coordinate Reference
    System (CRS) from the shapefile for spatial applications.

    :param file: The path to the shapefile to be read. Must be supported by GeoPandas.
    :type file: str
    :param keys: A list of column names to extract from the shapefile. The first key
        in the list is used as the unique identifier for the dictionary.
    :type keys: list[str]
    :return: A tuple containing:
        - A nested dictionary representation of the specified columns from the
          shapefile.
        - The Coordinate Reference System (CRS) of the input shapefile.
    :rtype: tuple[dict, any]
    """
    # ------------------------------------------------------------------
    # 1. Read shapefile into a GeoDataFrame
    # ------------------------------------------------------------------
    geodataframe = gpd.read_file(file)
    # Store Coordinate Reference System for later spatial outputs
    crs=geodataframe.crs
    # ------------------------------------------------------------------
    # 2. Convert selected columns into dictionary format
    #    - Select only requested fields
    #    - Set the first key as index (unique element ID)
    #    - Transpose and convert to nested dictionary
    # ------------------------------------------------------------------
    dic = geodataframe[keys].set_index(keys[0]).T.to_dict('dict')
    return dic,crs

def csv_to_dic(file,geo_dic):
    """
    Convert a CSV file into a dictionary, combining existing data in the provided dictionary
    with newly processed data from the CSV file. The resulting dictionary will have keys
    taken from the first column of the CSV and values composed of the remaining columns.

    :param file: The path to the CSV file to be processed.
    :type file: str
    :param geo_dic: An existing dictionary to be updated with data from the CSV file.
        The keys from the CSV file will be added as new keys in this dictionary, and
        their corresponding values will be dictionaries derived from the remaining columns.
    :type geo_dic: dict
    :return: A dictionary updated with the combined data from the existing dictionary
        and the information extracted from the CSV file.
    :rtype: dict
    """

    def convert_value(val):
        """Attempt to convert a string to int or float."""
        if val is None or val == "":
            return None
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return value  # Keep as string if conversion fails
    main_dic=copy.deepcopy(geo_dic)
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            key = row[reader.fieldnames[0]]

            value = {
                k: convert_value(v)
                for k, v in row.items()
                if k != reader.fieldnames[0]
            }

            main_dic[key] = value

    return main_dic

