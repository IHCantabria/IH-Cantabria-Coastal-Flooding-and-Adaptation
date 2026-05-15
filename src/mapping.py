import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.colors import Normalize
import pandas as pd


def pintar_inundacion_edificios(
    gdf_buildings: gpd.GeoDataFrame,
    raster_path: str,
    zoom: int,
    gdf_study_area: gpd.GeoDataFrame | None = None,
    buildings_color: str = "#f4a261",
    buildings_alpha: float = 0.5,
    raster_cmap: str = "Blues",
    raster_alpha: float = 0.7,
    raster_vmin: float | None = None,
    raster_vmax: float | None = None,
    nodata_val: float | None = None,
    titulo: str | None = None,
    scale_label: str | None = "Profundidad (m)",
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery,
):
    """
    Mapa satelital + raster de inundación translúcido + edificios translúcidos.

    Parameters
    ----------
    gdf_buildings   : GeoDataFrame con polígonos de edificios
    raster_path     : ruta al raster de inundación (.tif)
    zoom            : nivel de zoom del mapa base
    gdf_study_area  : GeoDataFrame opcional con polígono de zona de estudio
    buildings_color : color de los edificios
    buildings_alpha : transparencia de los edificios (0-1)
    raster_cmap     : colormap del raster
    raster_alpha    : transparencia del raster (0-1)
    raster_vmin/max : rango de valores del raster (None = automático)
    nodata_val      : valor nodata del raster (None = leer del fichero)
    """

    # ------------------------------------------------------------------
    # 1. Reproyectar raster a EPSG:3857 en memoria
    # ------------------------------------------------------------------
    with rasterio.open(raster_path) as src:
        nd = nodata_val if nodata_val is not None else src.nodata

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds
        )
        data = np.empty((src.count, height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data[0],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear,
        )

    band = data[0].astype(np.float64)

    # Enmascarar nodata y valores <= 0
    mask = np.zeros_like(band, dtype=bool)
    if nd is not None:
        mask |= np.isclose(band, nd)
    mask |= (band <= 0)
    band = np.ma.masked_where(mask, band)

    # Extent del raster en 3857
    left   = transform.c
    top    = transform.f
    right  = left + transform.a * width
    bottom = top  + transform.e * height
    raster_extent = [left, right, bottom, top]  # para imshow

    # ------------------------------------------------------------------
    # 2. Reproyectar geodataframes
    # ------------------------------------------------------------------
    gdf_buildings = gdf_buildings.to_crs(3857)
    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)

    # ------------------------------------------------------------------
    # 3. Calcular extent conjunto
    # ------------------------------------------------------------------
    all_bounds = [gdf_buildings.total_bounds]
    if gdf_study_area is not None:
        all_bounds.append(gdf_study_area.total_bounds)

    all_bounds = np.array(all_bounds)
    x0 = all_bounds[:, 0].min()
    y0 = all_bounds[:, 1].min()
    x1 = all_bounds[:, 2].max()
    y1 = all_bounds[:, 3].max()

    dx = (x1 - x0) * 0.10
    dy = (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Mapa base satelital
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    # Zona de estudio (opcional)
    if gdf_study_area is not None:
        gdf_study_area.plot(
            ax=ax, color="white", edgecolor="white",
            linewidth=1.5, alpha=0.2, zorder=2,
        )

    # Raster de inundación
    vmin = raster_vmin if raster_vmin is not None else float(band.min())
    vmax = raster_vmax if raster_vmax is not None else float(band.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        band,
        extent=raster_extent,
        cmap=raster_cmap,
        norm=norm,
        alpha=raster_alpha,
        zorder=3,
        interpolation="bilinear",
    )

    # Edificios
    gdf_buildings.plot(
        ax=ax,
        color=buildings_color,
        edgecolor="black",
        linewidth=0.3,
        alpha=buildings_alpha,
        zorder=4,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    # Colorbar raster
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if scale_label:
        cbar.set_label(scale_label, fontsize=10)

    # Leyenda
    legend_handles = [
        mpatches.Patch(color=buildings_color, alpha=buildings_alpha, label="Edificios"),
    ]
    if gdf_study_area is not None:
        legend_handles.insert(0, mpatches.Patch(color="white", alpha=0.4, label="Zona de estudio"))
    ax.legend(handles=legend_handles, loc="best", fontsize=10)

    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()

def pintar_edificios_por_campo(
    gdf_buildings: gpd.GeoDataFrame,
    value_col: str,
    zoom: int,
    gdf_study_area: gpd.GeoDataFrame | None = None,
    cmap: str = "Reds",
    buildings_alpha: float = 0.8,
    vmin: float | None = None,
    vmax: float | None = None,
    titulo: str | None = None,
    scale_label: str | None = None,
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery,
):
    if value_col not in gdf_buildings.columns:
        raise ValueError(f"La columna '{value_col}' no existe. Columnas disponibles: {list(gdf_buildings.columns)}")

    gdf_buildings = gdf_buildings.to_crs(3857)
    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)

    # Extent
    all_bounds = [gdf_buildings.total_bounds]
    if gdf_study_area is not None:
        all_bounds.append(gdf_study_area.total_bounds)
    all_bounds = np.array(all_bounds)

    x0, y0 = all_bounds[:, 0].min(), all_bounds[:, 1].min()
    x1, y1 = all_bounds[:, 2].max(), all_bounds[:, 3].max()
    dx, dy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Mapa base
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    # Zona de estudio
    if gdf_study_area is not None:
        gdf_study_area.plot(
            ax=ax, color="white", edgecolor="white",
            linewidth=1.5, alpha=0.2, zorder=2,
        )

    # Edificios coloreados por campo
    _vmin = vmin if vmin is not None else gdf_buildings[value_col].min()
    _vmax = vmax if vmax is not None else gdf_buildings[value_col].max()

    gdf_buildings.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        vmin=_vmin,
        vmax=_vmax,
        edgecolor="black",
        linewidth=0.3,
        alpha=buildings_alpha,
        zorder=3,
        legend=False,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=_vmin, vmax=_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    if scale_label:
        cbar.set_label(scale_label, fontsize=10)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()

def pintar_diferencia_edificios(
    gdf_1: gpd.GeoDataFrame,
    gdf_2: gpd.GeoDataFrame,
    value_col: str,
    zoom: int,
    join_col: str = None,
    gdf_study_area: gpd.GeoDataFrame | None = None,
    cmap: str = "Reds",
    buildings_alpha: float = 0.8,
    vmin: float | None = 0,
    vmax: float | None = 100,
    titulo: str | None = None,
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery,
):
    """
    Pinta edificios coloreados por diferencia relativa absoluta (%) entre dos GeoDataFrames.

    Parameters
    ----------
    gdf_1, gdf_2  : GeoDataFrames con los mismos polígonos
    value_col     : campo sobre el que calcular la diferencia
    join_col      : columna id para hacer el merge (None = por índice)
    vmin, vmax    : rango de la escala de color (% por defecto 0 a 100)
    """
    gdf_1 = gdf_1.copy().to_crs(3857)
    gdf_2 = gdf_2.copy().to_crs(3857)

    if value_col not in gdf_1.columns or value_col not in gdf_2.columns:
        raise ValueError(f"La columna '{value_col}' debe existir en ambos GeoDataFrames.")

    # Calcular diferencia relativa absoluta
    if join_col:
        merged = gdf_1[[join_col, "geometry", value_col]].merge(
            gdf_2[[join_col, value_col]].rename(columns={value_col: f"{value_col}_2"}),
            on=join_col,
        )
    else:
        merged = gdf_1[["geometry", value_col]].copy()
        merged[f"{value_col}_2"] = gdf_2[value_col].values

    denom = merged[value_col].replace(0, np.nan)
    merged["diff_pct"] = ((merged[f"{value_col}_2"] - merged[value_col]) / denom * 100).abs()

    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)

    # Extent
    all_bounds = [merged.total_bounds]
    if gdf_study_area is not None:
        all_bounds.append(gdf_study_area.total_bounds)
    all_bounds = np.array(all_bounds)

    x0, y0 = all_bounds[:, 0].min(), all_bounds[:, 1].min()
    x1, y1 = all_bounds[:, 2].max(), all_bounds[:, 3].max()
    dx, dy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Mapa base
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    # Zona de estudio
    if gdf_study_area is not None:
        gdf_study_area.plot(
            ax=ax, color="white", edgecolor="white",
            linewidth=1.5, alpha=0.2, zorder=2,
        )

    # Edificios coloreados por diferencia relativa absoluta
    _vmin = vmin if vmin is not None else merged["diff_pct"].min()
    _vmax = vmax if vmax is not None else merged["diff_pct"].max()

    merged.plot(
        ax=ax,
        column="diff_pct",
        cmap=cmap,
        vmin=_vmin,
        vmax=_vmax,
        edgecolor="black",
        linewidth=0.3,
        alpha=buildings_alpha,
        zorder=3,
        legend=False,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=_vmin, vmax=_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Diferencia relativa absoluta (%)", fontsize=10)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()

def pintar_diferencia_rasters(
    raster_path_1: str,
    raster_path_2: str,
    zoom: int,
    gdf_study_area: gpd.GeoDataFrame | None = None,
    raster_cmap: str = "RdYlGn_r",
    raster_alpha: float = 0.7,
    raster_vmin: float | None = None,
    raster_vmax: float | None = None,
    nodata_val: float | None = None,
    titulo: str | None = None,
    scale_label: str | None = "Diferencia (m)",
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery,
):
    """
    Pinta la diferencia (raster_2 - raster_1) sobre mapa satelital.
    Los dos rasters deben tener la misma resolución y extensión.
    """

    def _reproject_band(raster_path, nd_override):
        with rasterio.open(raster_path) as src:
            nd = nd_override if nd_override is not None else src.nodata
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:3857", src.width, src.height, *src.bounds
            )
            data = np.empty((1, height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data[0],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.bilinear,
            )
        return data[0].astype(np.float64), transform, width, height, nd

    band1, transform, width, height, nd = _reproject_band(raster_path_1, nodata_val)
    band2, _, _, _, _                   = _reproject_band(raster_path_2, nodata_val)

    # Máscara nodata en cualquiera de los dos
    mask = np.zeros_like(band1, dtype=bool)
    if nd is not None:
        mask |= np.isclose(band1, nd) | np.isclose(band2, nd)
    mask |= ~(np.isfinite(band1) & np.isfinite(band2))

    diff = np.ma.masked_where(mask, band2 - band1)

    left   = transform.c
    top    = transform.f
    right  = left + transform.a * width
    bottom = top  + transform.e * height
    raster_extent = [left, right, bottom, top]

    # Extent
    x0, y0, x1, y1 = left, bottom, right, top

    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)
        sb = gdf_study_area.total_bounds
        x0 = min(x0, sb[0])
        y0 = min(y0, sb[1])
        x1 = max(x1, sb[2])
        y1 = max(y1, sb[3])

    dx, dy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    if gdf_study_area is not None:
        gdf_study_area.plot(
            ax=ax, color="white", edgecolor="white",
            linewidth=1.5, alpha=0.2, zorder=2,
        )

    vmin = raster_vmin if raster_vmin is not None else float(diff.min())
    vmax = raster_vmax if raster_vmax is not None else float(diff.max())

    # Escala simétrica alrededor de 0 si no se especifica
    if raster_vmin is None and raster_vmax is None:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(
        diff,
        extent=raster_extent,
        cmap=raster_cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        alpha=raster_alpha,
        zorder=3,
        interpolation="bilinear",
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if scale_label:
        cbar.set_label(scale_label, fontsize=10)

    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()

def pintar_diferencia_poligonos(
    gdf_1: gpd.GeoDataFrame,
    gdf_2: gpd.GeoDataFrame,
    zoom: int,
    gdf_study_area: gpd.GeoDataFrame | None = None,
    alpha: float = 0.7,
    titulo: str | None = None,
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery,
):
    """
    Pinta en verde los polígonos presentes en ambos GeoDataFrames (por solapamiento)
    y en rojo los que solo están en uno de ellos.

    Parameters
    ----------
    gdf_1, gdf_2 : GeoDataFrames a comparar
    zoom         : nivel de zoom del mapa base
    """
    gdf_1 = gdf_1.copy().to_crs(3857)
    gdf_2 = gdf_2.copy().to_crs(3857)

    # Spatial join para detectar solapamiento
    joined = gpd.sjoin(gdf_1, gdf_2, how="left", predicate="intersects", lsuffix="l", rsuffix="r")

    # Los que tienen match en gdf_2 (index_r no es NaN) -> en ambos
    in_both_idx = joined[joined["index_r"].notna()].index.unique()

    gdf_1["_status"] = "solo_gdf1"
    gdf_1.loc[gdf_1.index.isin(in_both_idx), "_status"] = "ambos"

    # Polígonos de gdf_2 que no solapan con gdf_1
    joined_2 = gpd.sjoin(gdf_2, gdf_1, how="left", predicate="intersects", lsuffix="l", rsuffix="r")
    only_gdf2_idx = joined_2[joined_2["index_r"].isna()].index.unique()
    gdf_2_only = gdf_2.loc[gdf_2.index.isin(only_gdf2_idx)].copy()
    gdf_2_only["_status"] = "solo_gdf2"

    gdf_all = pd.concat([gdf_1, gdf_2_only], ignore_index=True)

    color_map = {
        "ambos":    "#2a9d8f",  # verde
        "solo_gdf1": "#e63946",  # rojo
        "solo_gdf2": "#e63946",  # rojo
    }
    gdf_all["_color"] = gdf_all["_status"].map(color_map)

    # Extent
    all_bounds = [gdf_all.total_bounds]
    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)
        all_bounds.append(gdf_study_area.total_bounds)
    all_bounds = np.array(all_bounds)

    x0, y0 = all_bounds[:, 0].min(), all_bounds[:, 1].min()
    x1, y1 = all_bounds[:, 2].max(), all_bounds[:, 3].max()
    dx, dy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Mapa base
    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    # Zona de estudio
    if gdf_study_area is not None:
        gdf_study_area.plot(
            ax=ax, color="white", edgecolor="white",
            linewidth=1.5, alpha=0.2, zorder=2,
        )

    # Polígonos coloreados
    gdf_all.plot(
        ax=ax,
        color=gdf_all["_color"],
        edgecolor="black",
        linewidth=0.3,
        alpha=alpha,
        zorder=3,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    # Leyenda
    legend_handles = [
        mpatches.Patch(color="#2a9d8f", alpha=alpha, label="En ambos"),
        mpatches.Patch(color="#e63946", alpha=alpha, label="Solo en uno"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=10)

    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()