from src.tools import input_reading
from src.tools import dictionaries as dics
from src.tools import outputs

from src.level_3_risk_analysis.shape_exp import shape_exp

state_counter=0

def main(hazard_input_dic: dict, params_dic: dict, exp_sys_folder, scen_raster_dic: dict = None) -> None:
    """
    Processes hazard input data, exposed systems, and calculates risk analysis based on the provided
    parameters. The function integrates hazard data, vector and raster-based exposed systems, and 
    computes results for defined scenarios, horizons, and return periods.

    Data is categorized and processed with specific methods for both types of exposed systems 
    (vector and raster), performing zonal statistical analysis for vector data, and raster-to-raster 
    analysis for raster data. Results are ultimately outputted as summary statistics and, if enabled, 
    partial aggregation results.

    :param hazard_input_dic: (dict) Dictionary containing hazard input data. Each key represents a hazard
        type (e.g., 'Flooding'), and its value is a dictionary with:
        - 'folder' (str): Name of the subfolder within 'haz_input_data/' where files are located.
        - 'extension' (str): File extension ('.tif' for raster, '.shp' for vector).
    :param params_dic: (dict) Dictionary containing global execution parameters:
        - 'scenarios' (list): List of climate/risk scenarios (e.g., ['RCP45']). Use [] if filenames do not depend on a scenario.
        - 'horizons' (list): Time horizons (e.g., ['2030']). Use [] if filenames do not depend on a horizon.
        - 'return periods' (list): List of return periods (e.g., ['100']). Use [] if filenames do not depend on a return period.
        - 'partial agg' (bool): Whether to generate results by territorial units (True) or only global (False).
        - 'zonal stats method' (str): For vector systems; 'centers' or 'all touched'.
        - 'zonal stats value' (str): For vector systems; statistic to compute ('mean' or 'max').
    :param scen_raster_dic: (dict, optional) Dictionary with metadata for raster exposure systems.
        Keys are filenames (without .tif), and values are dictionaries with:
        - 'Type of system' (str): Category of the system (e.g., 'POP', 'AGR').
        - 'Damage function' (str): Name of the damage function to apply. Use 'file' to apply spatially 
          distributed functions defined in an external shapefile.
        - 'Damage function file' (str, optional): Name of the shapefile (without extension) in 
          'inputs/dam_fun_files/' if 'Damage function' is set to 'file'.
    :return: None
    """
    # Global counter used to track progress of processed scenarios
    global state_counter

    # ------------------------------------------------------------------
    # 1. Load hazard files for each hazard indicator with additional metadata: crs, extension...
    # ------------------------------------------------------------------
    for indicator_indiv_dic in hazard_input_dic.values():
        indicator_indiv_dic['files'] =input_reading.reading_files('outputs/'+indicator_indiv_dic['folder'],indicator_indiv_dic['extension'])
    '''for k, v in hazard_input_dic.items():
        print(k, v['files'].keys())'''
    # Rearrange hazard dictionary according to scenarios, horizons, and return rates
    hazard_input_dic=rearranging_dics(hazard_input_dic,params_dic['scenarios'],params_dic['horizons'],params_dic['return periods'],params_dic['percentiles'])
    '''for scen_hor_rp, scen_hor_rp_dic in hazard_input_dic.items():
        for haz, haz_dic in scen_hor_rp_dic.items():
            print(scen_hor_rp, '->', haz_dic['path'])'''

    # ------------------------------------------------------------------
    # 2. Load exposed systems (vector .shp and raster .tif)
    # ------------------------------------------------------------------
    expsystdic=input_reading.reading_files('data/'+exp_sys_folder, ('.shp','.tif','.geojson'))

    # Containers for outputs
    summarydic = [] # Global summary results
    partialaggdic = [] # Partial aggregation results (optional)

    # ------------------------------------------------------------------
    # 3. Main processing loop over exposed systems
    # ------------------------------------------------------------------
    for syst, syst_dic in expsystdic.items():
        # --------------------------------------------------------------
        # CASE A: Vector exposed system (.shp)
        # --------------------------------------------------------------
        if syst_dic['extension'] == '.shp' or syst_dic['extension'] == '.geojson':

            for scen_hor_rp,scen_hor_rp_dic in hazard_input_dic.items():
                # Perform risk analysis using zonal statistics
                scensum,scen_partial_agg_dic=shape_exp.shape_exp(
                    syst,
                    scen_hor_rp,
                    syst_dic,
                    scen_hor_rp_dic,
                    params_dic['partial agg'],
                    params_dic['zonal stats method'],
                    params_dic['zonal stats value'])

                # Store results
                summarydic.append(scensum)
                if params_dic['partial agg']:
                    partialaggdic.append(scen_partial_agg_dic)

                print(scen_hor_rp)
                state_counter += 1
        # --------------------------------------------------------------
        # Print processed system name
        print(syst)

    # Export the summary dictionary and the aggregated partial dictionary (if needed) to a .csv file.
    outputs.summary_output(summarydic)
    if params_dic['partial agg']:
        outputs.partial_agg_output(partialaggdic)
def output_fields_keys(fields,dic):
    """
        Map output field names to internal keys used in a dictionary of system elements.

        For most fields, the mapping is direct using `dics.keysoutputdic`.
        For 'Exposed value' and 'Impact damage', the mapping depends on the
        system type of the first element in the dictionary.

        PARAMETERS
        ----------
        fields : list of str
            List of human-readable output field names.

        dic : dict of dict or list of dict
            Dict of system elements. Each element is a sub-dictionary
            containing 'Type of system' and other attributes.

        RETURNS
        -------
        list
            List of internal keys corresponding to each output field.
        """
    fieldkeys = []
    # Identify system type from the first element (assumes uniform type)
    for field in fields:
        if field == 'Exposed value' or field == 'Impact damage':
            system_type = dic[list(dic.keys())[0]][dics.keysdic['Type of system']]
            # Map field based on system type
            fieldkeys.append(dics.keysoutputdic[field][system_type])
        else:
            # Direct mapping for other fields
            fieldkeys.append(dics.keysoutputdic[field])
    return fieldkeys

def parse_file_params(file_name):
    """Extrae (rp, scenario, percentile, horizon) por posición desde el final."""
    tokens = file_name.split('_')
    # Formato: ..._{RP}_{SCENARIO}_{PERCENTILE}_{HORIZON}
    # Los últimos 4 tokens son: horizon, percentile, scenario, rp (en orden inverso)
    if len(tokens) >= 6:  # flooding + Martil + RP + SCEN + PCT + HOR
        return {
            'rp':         tokens[-4],
            'scenario':   tokens[-3],
            'percentile': tokens[-2],
            'horizon':    tokens[-1]
        }
    return None


def rearranging_dics(hazard_input_dic, scenarios, horizons, return_rates, percentiles):

    scen_hor_ret_dic = {}

    for scen in (scenarios or ['']):
        for hor in (horizons or ['']):
            for ret in (return_rates or ['']):
                for pct in (percentiles or ['']):

                    key = '_'.join(filter(None, [scen, hor, ret, pct]))
                    scen_hor_ret_dic[key] = {}

                    for haz, haz_dic in hazard_input_dic.items():
                        for file_name, file_dic in haz_dic['files'].items():
                            p = parse_file_params(file_name)
                            if p is None:
                                continue
                            if (p['scenario']   == scen and
                                p['horizon']     == hor  and
                                p['rp']          == ret  and
                                p['percentile']  == pct):
                                scen_hor_ret_dic[key][haz] = file_dic

    return scen_hor_ret_dic