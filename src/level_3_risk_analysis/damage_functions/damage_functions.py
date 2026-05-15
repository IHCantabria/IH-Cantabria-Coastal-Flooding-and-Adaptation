import numpy as np
from pathlib import Path
import geopandas as gpd
from rasterio.features import geometry_mask
import json
from scipy.interpolate import LinearNDInterpolator, interp1d

from src.tools import dictionaries as dics


class FunctionLibrary:
    """
        Load, index, and provide callable damage functions defined in a JSON file.

        The class:
        1. Reads a JSON dictionary containing function definitions.
        2. Builds a fast lookup index by function name.
        3. Lazily constructs callable interpolation functions on demand.
        4. Caches created callables to avoid recomputation.

        METHODS
        -------
        get(name)
           Return a callable damage function associated with the given name.
        """
    def __init__(self):
        # --------------------------------------------------------------
        # 1. Load JSON file containing damage function definitions
        #    Path is resolved relative to the current working directory.
        # --------------------------------------------------------------
        json_path=Path.cwd()/'src//level_3_risk_analysis//damage_functions//damage_functions_dictionary.json'
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # --------------------------------------------------------------
        # 2. Create index: function name → function metadata block
        #    This enables fast O(1) lookup when requesting functions.
        # --------------------------------------------------------------
        self.index = {f["name"]: f for f in self.data["functions"]}
        # --------------------------------------------------------------
        # 3. Cache for already-built callable functions
        #    Avoids rebuilding interpolators multiple times.
        # --------------------------------------------------------------
        self.cache = {}

    def get(self, name):
        """
            Retrieve a callable damage function by name.

            PARAMETERS
            ----------
            name : str
                Name of the function defined in the JSON dictionary.

            RETURNS
            -------
            callable
                Function that maps hazard inputs → damage fraction.
            """
        # --------------------------------------------------------------
        # 1. Return cached function if already created
        # --------------------------------------------------------------
        if name in self.cache:
            return self.cache[name]

        # --------------------------------------------------------------
        # 2. Retrieve function metadata from index
        # --------------------------------------------------------------
        fdata = self.index[name]
        tipo = fdata["type"]

        # --------------------------------------------------------------
        # 3. Build callable depending on the function type
        #    Currently supports multidimensional interpolation.
        # --------------------------------------------------------------
        f=None
        if tipo == "interpolation":
            # Damage values associated with interpolation points
            vals = np.array(fdata["values"])
            # Interpolation method specified in the function metadata
            kind = fdata["interpolation_type"]
            # CASE 1: One-dimensional interpolation
            if len(fdata["variables"]) == 1:
                # Create 1D interpolator
                # bounds_error=False → allows evaluation outside data range
                # fill_value="extrapolate" → extrapolates beyond known points
                f = interp1d(
                    np.array(fdata[fdata["variables"][0]]),
                    vals,
                    kind=kind,
                    bounds_error=False,
                    fill_value="extrapolate"
                )
            # CASE 2: Multi-dimensional interpolation
            else:
                # Build interpolation points matrix from variable arrays
                pts = np.column_stack([fdata[var] for var in fdata["variables"]])
                # Currently, only linear ND interpolation is supported
                if kind == "linear":
                    # Create linear N-D interpolator
                    f = LinearNDInterpolator(pts, vals)
        # --------------------------------------------------------------
        # 4. Store in cache and return
        # --------------------------------------------------------------
        self.cache[name] = f
        return f

def apply_damage_fun_shp(system_dic):
    """
        Compute the damage fraction for each exposed element in a vector system
        by applying the corresponding damage function to the hazard impact values.

        The computed damage fraction is added in-place to each element's dictionary.

        PARAMETERS
        ----------
        system_dic : dict
            Dictionary of exposed elements where:
            - key → element ID
            - value → dictionary containing:
                • hazard impact values
                • exposed value
                • damage function name
                • geometry and metadata

        RETURNS
        -------
        None
            The function modifies `system_dic` in place by adding the
            'Damage fraction' field to each element.
        """
    # ------------------------------------------------------------------
    # 1. Identify which keys correspond to hazard values
    #    (i.e., exclude metadata and non-hazard attributes)
    # ------------------------------------------------------------------
    indiv_element_dic = system_dic[list(system_dic.keys())[0]]
    haz_keys = [
        k for k in indiv_element_dic.keys()
        if k not in
           [
               dics.keysdic['Damage function'],
               dics.keysdic['Exposed value'],
               dics.keysdic['Type of system'],
               dics.keysdic['Impact scenario'],
               dics.keysdic['Area'],
               'geometry'
           ]
    ]

    # Load library containing all available damage functions
    lib = FunctionLibrary()

    # ------------------------------------------------------------------
    # 2. Apply the corresponding damage function to each element
    # ------------------------------------------------------------------
    for indiv_element_dic in system_dic.values():

        # Retrieve damage function name assigned to the element
        f_name=indiv_element_dic[dics.keysdic['Damage function']]

        application = lib.index[f_name]['application']


        # Collect hazard input values in the expected order
        input_values=[indiv_element_dic[key] for key in haz_keys]

        # Get the callable damage function from the library
        f=lib.get(f_name)

        if application == 'Relative':
            indiv_element_dic[dics.keysdic['Damage fraction']] = round(float(f(*input_values)), 3)
            indiv_element_dic[dics.keysdic['Impact damage']] = (
                    indiv_element_dic[dics.keysdic['Damage fraction']]*indiv_element_dic[dics.keysdic['Exposed value']])

        elif application=='Absolute':

            indiv_element_dic[dics.keysdic['Relative damage']] = round(float(f(*input_values)), 3)

            indiv_element_dic[dics.keysdic['Impact damage']] = (
                    indiv_element_dic[dics.keysdic['Relative damage']] * indiv_element_dic[
                dics.keysdic['Area']])

def apply_dam_fun_raster(raster_scen_data_list,combined_mask,dm_fun):
    """
        Apply a single damage function to hazard raster data.

        The function:
        1. Retrieves the selected damage function from the FunctionLibrary.
        2. Applies it pixel-wise to the hazard raster(s).
        3. Computes results only where all hazard inputs are valid
           (according to combined_mask).

        Parameters:
        - raster_scen_data_list: list of NumPy arrays representing hazard rasters.
        - combined_mask: boolean array indicating valid pixels across hazards.
        - dm_fun: name of the damage function to retrieve from the library.

        Returns:
        - NumPy array containing damage fractions (float32),
          with NaN where data are invalid.
        """

    # Initialize the function library and retrieve the selected damage function
    lib = FunctionLibrary()
    f = lib.get(dm_fun)

    # Create the output array filled with NaN
    # shape matching the first hazard raster
    raster_scen_data = np.full_like(
        raster_scen_data_list[0],
        np.nan,
        dtype=np.float32
    )

    # Apply damage function only on valid pixels (combined_mask == True)
    # The function may accept multiple hazard inputs
    raster_scen_data[combined_mask] = f(
        *[r[combined_mask] for r in raster_scen_data_list]
    )

    return raster_scen_data

def apply_dam_fun_file(raster_scen_data_list,combined_mask,dm_fun_file,kwargs):
    """
        Apply spatially distributed damage functions defined in a shapefile.

        The shapefile must contain:
        - Polygon geometries defining spatial zones.
        - A column 'DAM_FUN' specifying the damage function name
          for each polygon.

        The function:
        1. Reads the damage function shapefile.
        2. Creates a spatial mask for each polygon.
        3. Retrieves the corresponding damage function.
        4. Applies it only within the polygon and valid hazard pixels.

        Parameters:
        - raster_scen_data_list: list of NumPy arrays representing hazard rasters.
        - combined_mask: boolean array indicating valid pixels across hazards.
        - dm_fun_file: name of the shapefile (without extension).
        - kwargs: metadata dictionary containing at least the raster transform.

        Returns:
        - NumPy array containing spatially variable damage fractions (float32),
          with NaN where data are invalid.
        """
    # Initialize the function library
    lib = FunctionLibrary()

    # Build path to the damage function shapefile
    dm_fun_file_path=Path.cwd().parent.parent/'inputs'/'dam_fun_files'/dm_fun_file+'.shp'
    # Read shapefile containing spatially distributed damage functions
    gdf = gpd.read_file(dm_fun_file_path)

    # Create the output array filled with NaN
    # shape matching the first hazard raster
    raster_scen_data = np.full_like(
        raster_scen_data_list[0],
        np.nan,
        dtype=np.float32)

    # Iterate through each polygon feature in the shapefile
    for index, row in gdf.iterrows():
        geom = row.geometry

        # Create the mask for current geometry
        # True inside polygon, False outside
        mask = geometry_mask(
            [geom],
            transform=kwargs['transform'],
            invert=True,
            out_shape=raster_scen_data.shape
        )

        # Combine the geometry mask with hazard validity mask
        mask = combined_mask & mask
        # Retrieve damage function associated with this polygon
        dm_fun_name = row['DAM_FUN']
        f = lib.get(dm_fun_name)
        # Apply damage function only within polygon and valid hazard pixels
        raster_scen_data[mask] = f(
            *[r[mask] for r in raster_scen_data_list]
        )

    return raster_scen_data