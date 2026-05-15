import pandas as pd
import numpy as np
from pathlib import Path
import csv

def calculate_aed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Annual Expected Damage (AED) para cada combinación de
    Exposed system, escenario, horizonte y percentil.

    El campo 'Impact scenario' tiene el formato: {escen}_{hor}_{rp}_{percentil}
    AED = integral de damage(p) dp ≈ suma de damage_i * delta_p_i
    usando la regla del trapecio sobre las probabilidades de excedencia (1/rp).
    """


    # Descomponer el campo Impact scenario en sus partes
    parts = df['Impact scenario'].str.split('_', expand=True)
    df = df.copy()
    df['escen']      = parts[0]
    df['hor']        = parts[1]
    df['rp']         = parts[2].astype(float)
    df['percentil']  = parts[3]
    df['prob']       = 1.0 / df['rp']   # probabilidad de excedencia anual

    results = []
    group_keys = ['Exposed system', 'Type of element', 'Exposed value',
                  'escen', 'hor', 'percentil']

    for group_vals, group_df in df.groupby(group_keys):
        # Ordenar por probabilidad descendente (rp ascendente)
        group_df = group_df.sort_values('prob', ascending=False)

        probs   = group_df['prob'].values
        damages = group_df['Impact damage'].values

        # AED por regla del trapecio
        aed = np.trapezoid(damages, probs) * -1

        row = dict(zip(group_keys, group_vals))
        row['AED'] = aed
        results.append(row)

    return pd.DataFrame(results)


def export_aed_csv(filename: str, csv_path) -> None:
    """
    Calcula el AED y exporta el resultado a CSV con separador punto y coma.

    Parameters:
    - filename: nombre del archivo de salida (sin extensión).
    - df_input: DataFrame con las columnas del results_summary.
    """

    df_input = pd.read_csv(csv_path, sep=';')

    df_aed = calculate_aed(df_input)

    # Reordenar columnas para el CSV
    col_order = [
        'Exposed system', 'Type of element', 'Exposed value',
        'escen', 'hor', 'percentil', 'AED'
    ]
    df_aed = df_aed[col_order]

    # Exportar
    csv_path = Path(csv_path)
    path = csv_path.parent/ (filename + '.csv')

    df_aed.to_csv(path, sep=';', index=False)
    print(f"AED exportado en: {path}")

    return df_aed