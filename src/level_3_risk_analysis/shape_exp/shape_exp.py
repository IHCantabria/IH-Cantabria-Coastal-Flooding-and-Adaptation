from src.tools import input_reading
from src.tools import list_dics_functions as ldfun
from src.level_3_risk_analysis.damage_functions import damage_functions as dmfun
from src.tools import outputs
from src.tools import dictionaries as dics
from src.level_3_risk_analysis import level_3_analysis
from src.tools import compute_zonal_stats

def shape_exp(
        syst,
        scen_hor_rp,
        expsystdic,
        scen_hor_rp_dic,
        partial_agg_flag,
        zonal_stats_method='centers',
        zonal_stats_value='mean'
):
    """
        Perform risk analysis for a vectorial exposed system (shapefile)
        under a specific hazard.

        The function:
        1. Loads exposed elements from the shapefile into a dictionary.
        2. Computes hazard impact values per element.
        3. Applies damage functions and calculates economic damage.
        4. Exports detailed results (shapefile and CSV).
        5. Returns scenario-level summary and optional partial aggregation.

        PARAMETERS
        ----------
        syst : str
            Name of the exposed system.

        scen_hor_rp : str
            Impact scenario identifier.

        expsystdic : dict
            Metadata of the exposed system shapefile
            (path, CRS, extension, etc.).

        scen_hor_rp_dic : dict
            Dictionary of hazard layers for the given scenario.

        partial_agg_flag : bool
            If True, compute and return partial aggregation results.

        zonal_stats_method : str, default 'centers'
            Method used for zonal statistics:
                - 'centers'
                - 'all touched'

        zonal_stats_value : str, default 'mean'
            Statistic extracted from hazard layer:
                - 'mean'
                - 'max'

        RETURNS
        -------
        tuple
            (scensum, partial_agg)

            scensum : dict
                Scenario summary results.

            partial_agg : dict or None
                Dictionary of partial aggregation results if partial_agg_flag is True,
                otherwise None.
        """
    # ------------------------------------------------------------------
    # 1. Load exposed system elements from shapefile into dictionary
    # ------------------------------------------------------------------
    system_dic,_ = input_reading.shp_to_dic(
        expsystdic['path'],
        [
            dics.keysdic['Elements ID'],
            dics.keysdic['Type of system'],
            dics.keysdic['Exposed value'],
            dics.keysdic['Damage function'],
            dics.keysdic['Area'],
            'geometry'])
    # Add the current impact scenario to each exposed element
    ldfun.add_value_to_dicofdics(system_dic, dics.keysdic['Impact scenario'], scen_hor_rp)

    # ------------------------------------------------------------------
    # 2. Initialize scenario summary information
    # ------------------------------------------------------------------
    scensum = {dics.keysdic['Exposed system']: syst,
               dics.keysdic['Type of system']: system_dic[next(iter(system_dic))][dics.keysdic['Type of system']],
               dics.keysdic['Exposed value']: ldfun.column_sum(system_dic, dics.keysdic['Exposed value']),
               dics.keysdic['Impact scenario']: scen_hor_rp}

    # Keep track of processed hazard names for CSV output
    haz_keys=[]
    # ------------------------------------------------------------------
    # 3. Compute hazard impact values for each hazard layer
    # ------------------------------------------------------------------
    for haz in scen_hor_rp_dic.keys():

        haz_keys.append(haz)

        # --- Raster hazard → zonal statistics over polygons ---
        if scen_hor_rp_dic[haz]['extension'] == '.tif':

            zonal_stats = compute_zonal_stats.shape_raster_zonal_stats(
                expsystdic['path'],
                scen_hor_rp_dic[haz]['path'],
                dics.keysdic['Elements ID'],
                zonal_stats_method,
                zonal_stats_value
            )
            ldfun.add_dic_to_dicofdics(system_dic, zonal_stats, haz)
        # --- Vector hazard → zonal statistics of overlapping hazard polygons into exposure polygons ---
        elif scen_hor_rp_dic[haz]['extension'] == '.shp':
            zonal_stats = compute_zonal_stats.shape_shape_zonal_stats(
                expsystdic['path'],
                scen_hor_rp_dic[haz]['path'],
                dics.keysdic['Elements ID'],
                dics.keysdic['Impact value'],
                zonal_stats_value
            )

            ldfun.add_dic_to_dicofdics(system_dic, zonal_stats, dics.keysdic['Impact value'])

    # ------------------------------------------------------------------
    # 4. Apply vulnerability/damage functions
    # ------------------------------------------------------------------
    # Convert impact value → damage fraction using predefined curves
    dmfun.apply_damage_fun_shp(system_dic)

    # ------------------------------------------------------------------
    # 5. Export detailed outputs
    # ------------------------------------------------------------------

    # Export enriched dictionary as shapefile (and if partial_agg_flag='external', assigning a section identificator
    # to each element from the external zoning file. (It is done here because it's the only place in the code where
    # there is a gdf with geometries.))

    outputs.shapefile_output(syst+scen_hor_rp, system_dic, expsystdic['crs'],partial_agg_flag)

    # Prepare CSV field structure
    fields=[
        'Elements ID',
        'Type of system',
        'Exposed value',
        'Impact scenario',
        'Damage function',
        'Damage fraction',
        'Relative damage',
        'Impact damage'
    ]
    # Map internal keys → output field names
    new_field_names = level_3_analysis.output_fields_keys(fields, system_dic)
    # Insert hazard columns before damage-related fields
    new_field_names[4:4] = haz_keys
    fields[4:4] = haz_keys
    # Export CSV table
    outputs.csv_output(syst + scen_hor_rp,fields,new_field_names,system_dic)

    # ------------------------------------------------------------------
    # 6. Update scenario summary with aggregated impact damage
    # ------------------------------------------------------------------
    scensum[dics.keysdic['Impact damage']] = ldfun.column_sum(
        system_dic,
        dics.keysdic['Impact damage']
    )

    # ------------------------------------------------------------------
    # 7. Optional partial aggregation by section/category
    # ------------------------------------------------------------------
    if partial_agg_flag:
        return scensum,partial_aggregates(system_dic,syst,scen_hor_rp)


    return scensum,None


def partial_aggregates(
        system_dic,
        syst,
        scen_hor_rp
):
    """
        Aggregate exposed system results by Section identificator.

        For each unique section identifier in the system dictionary, this function:
        1. Creates a new entry in the partial dictionary (if it doesn’t exist yet).
        2. Initializes the aggregation fields.
        3. Sums 'Exposed value' and 'Impact damage' across all elements belonging to that section.

        PARAMETERS
        ----------
        system_dic : dict of dict
            Dictionary of system elements where each value contains:
            - Exposed value
            - Impact damage
            - Section identificator
            - Type of system
            - Other metadata

        syst : str
            Name of the exposed system.

        scen_hor_rp : str
            Scenario identifier.

        RETURNS
        -------
        dict
            Dictionary keyed by Section identificator, each containing
            aggregated values and metadata.
        """

    partial_dic = {}

    # Iterate through all elements in the system
    for value in system_dic.values():

        # Identify the section for this element
        sec_ind=value[dics.keysdic['Section identificator']]

        # Initialize aggregation entry if not present
        if sec_ind not in partial_dic:
            partial_dic[sec_ind] = {
                dics.keysdic['Exposed system']:syst,
                dics.keysdic['Type of system']:
                    system_dic[next(iter(system_dic))][dics.keysdic['Type of system']],
                dics.keysdic['Exposed value']:0,
                dics.keysdic['Impact scenario']:scen_hor_rp,
                dics.keysdic['Impact damage']:0
            }
        # Aggregate values into the section entry
        partial_dic[sec_ind][dics.keysdic['Exposed value']]+=value[dics.keysdic['Exposed value']]
        partial_dic[sec_ind][dics.keysdic['Impact damage']] += value[dics.keysdic['Impact damage']]

    return partial_dic


