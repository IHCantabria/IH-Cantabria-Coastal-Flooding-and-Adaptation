import os
import re
from scipy.spatial import cKDTree
from scipy.io import loadmat
import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math

from twl_components import build_index

def load_extremes_from_mat(
    path_save: str | Path,
    mesh: str,
    subdomain: str,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Lee todos los Eventos_Lat_*_Lon_*.mat y devuelve un GeoDataFrame con
    una fila por punto y columnas Hs_1, Hs_10, ..., Tp_1, Tp_10, ..., SS_1, SS_10, ...
    """
    search_dir = Path(path_save) / mesh / subdomain
    mat_files = sorted(search_dir.glob("**/Eventos_Lat_*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No se encontraron .mat en {search_dir}")

    rows = []
    for mat_path in mat_files:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        eventos = mat["Eventos"]

        lat = float(eventos.lat)
        lon = float(eventos.lon)

        row = {"geometry": Point(lon, lat)}

        for r in np.atleast_1d(eventos.results):
            tret = r.Tret
            if tret is None or (isinstance(tret, float) and np.isnan(tret)):
                continue
            # Formatea el RP: 1.0 → "1", 100.0 → "100"
            rp = str(int(tret)) if float(tret).is_integer() else str(tret)
            row[f"Hs_{rp}"]  = float(r.mpp.Hs)
            row[f"Tp_{rp}"]  = float(r.mpp.Tp)
            row[f"SS_{rp}"]  = float(r.mpp.SS)

        rows.append(row)

    return gpd.GeoDataFrame(rows, crs=crs)

def slr_preprocess(PERCENTILES,YEARS,SSPS,slr_points,data_dir,output_path):

    coords_csvs, archivos_csvs, tree_csvs = build_index(data_dir, '.csv')

    for ssp in SSPS.values():
        for p in PERCENTILES.keys():
            for y in YEARS.keys():
                slr_points[f'slr_{ssp}_p{p}_{y}'] = np.nan

    for i, row in slr_points.iterrows():
        x, y = row.geometry.x, row.geometry.y
        dist, idx = tree_csvs.query([(y, x)])
        idx = idx[0]
        point=archivos_csvs[idx]
        for archivo in point:
            print(archivo)
            ssp = re.search(r'ssp\d+', archivo).group()
            ruta = os.path.join(data_dir, archivo)
            df = pd.read_csv(ruta)
            df = df[df['quantile'].isin(PERCENTILES.values())].set_index('quantile')

            for p in PERCENTILES.keys():
                for y in YEARS.keys():
                    slr_points.at[i, f'slr_{SSPS[ssp]}_{p}_{y}'] = df.loc[PERCENTILES[p], YEARS[y]]

    slr_points.to_file(os.path.join(output_path), driver='GeoJSON')

    return slr_points

def slr_extremals(gdf_twl,gdf_slr,output_path,SSPS,PERCENTILES,YEARS,RP):

    # Construir árbol con los puntos SLR
    coords_slr = np.array([(geom.x, geom.y) for geom in gdf_slr.geometry])
    tree_slr = cKDTree(coords_slr)
    # Para cada punto TWL, buscar el SLR más cercano
    coords_twl = np.array([(geom.x, geom.y) for geom in gdf_twl.geometry])
    dist, idx = tree_slr.query(coords_twl)

    # Añadir índice del SLR más cercano
    gdf_twl['slr_idx'] = idx

    for ssp in SSPS:
        for p in PERCENTILES:
            for y in YEARS:
                slr_col = f'slr_{ssp}_{p}_{y}'
                # Coge el valor SLR del punto más cercano para cada fila TWL
                slr_vals = gdf_slr.iloc[idx][slr_col].values
                for rp in RP:
                    twl_vals=gdf_twl[rp].values
                    gdf_twl[f'{rp}_{ssp}_{p}_{y}'] = twl_vals + slr_vals

    gdf_twl = gdf_twl.drop(columns=['slr_idx'])
    gdf_twl.to_file(os.path.join(output_path), driver='GeoJSON')


    return gdf_twl

def safe_num_for_name(x: float, ndigits: int = 4) -> str:
    return f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".")

def plots_extremal_analysis(
        mesh: str,
        subdomain: str,
        lat: float,
        lon: float,
        base_dir: Path = Path(
            r"E:\Proyecto IHRAT\IH Cantabria - Coastal Flooding and Adaptation\outputs\extremal_events_analysis"),
) -> None:
    """
    Muestra todas las imágenes PNG de un punto (mesh, subdomain, lat, lon).
    """
    lat_s = safe_num_for_name(lat)
    lon_s = safe_num_for_name(lon)

    point_dir = base_dir / mesh / subdomain / f"Lat_{lat_s}_Lon_{lon_s}"

    if not point_dir.exists():
        print(f"No existe el directorio: {point_dir}")
        return

    images = sorted(point_dir.glob("*.png"))

    if not images:
        print(f"No se encontraron imágenes PNG en: {point_dir}")
        return

    n = len(images)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)  # aplanar siempre a 1D

    for ax, img_path in zip(axes, images):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")

    # ocultar ejes sobrantes si n no es múltiplo de ncols
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()
