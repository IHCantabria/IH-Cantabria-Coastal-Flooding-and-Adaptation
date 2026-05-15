import os
import re
from scipy.spatial import cKDTree
import numpy as np
import h5py
from scipy.io import loadmat
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import matplotlib.patches as mpatches
from pathlib import Path



def build_index(folder, ext):
    """
    Reads all files with the given extension from a folder, extracts their coordinates
    from the filename, and builds a KDTree for proximity search.

    :param folder: Path to the folder containing the files.
    :param ext: File extension to filter by (e.g. '.mat').
    :returns: Tuple of (coords, valid_files, tree) where coords is a np.array of
              shape (n, 2), valid_files is a list of filenames, and tree is a cKDTree.
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]

    coords = []
    valid_files = []
    seen_coords={}

    def extract_coords_from_filename(filename):
        """
        Extracts two coordinates (x, y) from the filename using regex.
        Adjust this function if your actual filename pattern differs.

        :param filename: Filename to parse.
        :returns: Tuple (x, y) of floats.
        :raises ValueError: If fewer than two numeric values are found in the filename.
        """
        name_no_ext = os.path.splitext(filename)[0]
        nums = re.findall(r'-?\d+\.\d+', name_no_ext)

        if len(nums) < 2:
            raise ValueError(f"Could not extract two coordinates from: {filename}")

        return float(nums[0]), float(nums[1])

    for f in files:
        try:
            x, y = extract_coords_from_filename(f)
            if (x, y) in seen_coords:
                i = seen_coords[(x, y)]
                valid_files[i].append(f)
            else:
                seen_coords[(x, y)] = len(valid_files)
                coords.append((x, y))
                valid_files.append([f])
        except Exception as e:
            print(f"Saltando archivo {f}: {e}")

    coords = np.array(coords)
    tree = cKDTree(coords)

    return coords, valid_files, tree

def load_nearest_mat(x, y, folder, tree, files):
    """
    Finds the closest file to (x, y) based on coordinates encoded in the filename
    and loads it.

    :param x: Longitude of the query point.
    :param y: Latitude of the query point.
    :param folder: Path to the folder containing the .mat files.
    :param tree: cKDTree built from file coordinates.
    :param files: List of filenames corresponding to the tree points.
    :returns: Tuple of (data, filename, dist) where data is the loaded .mat content,
              filename is the matched file, and dist is the distance to the query point.
    """
    # Query tree with (y, x) since coordinates are stored as (lat, lon)
    dist, idx = tree.query([(y, x)])
    idx = idx[0]
    dist = dist[0]

    filename = files[idx][0]
    path = os.path.join(folder, filename)

    def read_hdf5(item):
        """
        Recursively reads an h5py item (dataset or group)
        and converts it into Python / numpy structures.

        :param item: h5py.Dataset or h5py.Group to read.
        :returns: numpy array, dict, or decoded string depending on item type.
        """
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if isinstance(data, bytes):
                return data.decode("utf-8")
            return np.array(data)

        elif isinstance(item, h5py.Group):
            return {key: read_hdf5(item[key]) for key in item.keys()}

        else:
            return item

    def load_mat(path):
        """
        Loads a classic or v7.3 .mat file, falling back to h5py for v7.3 format.

        :param path: Full path to the .mat file.
        :returns: Dict-like structure with the file contents.
        """
        try:
            return loadmat(path)
        except NotImplementedError:
            # v7.3 .mat files are HDF5-based and not supported by scipy.io.loadmat
            with h5py.File(path, "r") as f:
                return read_hdf5(f['data'])

    data = load_mat(path)

    return data, filename, dist

def find_closest_point(gdf,gdf_ref):
    # Build a KDTree to map reference points to this component's grid
    coords = np.array([(g.x, g.y) for g in gdf.geometry])
    tree = cKDTree(coords)
    coords_ref = np.array([(g.x, g.y) for g in gdf_ref.geometry])
    _, idx = tree.query(coords_ref)
    return idx

def save_to_nc(data_got, data_gos, data_gow,values):

    data_gow["tp"] = 1 / data_gow["fp"]
    data_gow["tm02"] = data_gow["t02"]
    data_gow = data_gow.drop(columns=["fp", "t02"])
    data_gos['surge']=data_gos['zeta']

    # Merge de los 3 DataFrames
    df_all = data_got.merge(data_gos, on=["x", "y", "time"]) \
        .merge(data_gow, on=["x", "y", "time"])

    points = df_all[["x", "y"]].drop_duplicates().reset_index(drop=True)
    times = np.sort(df_all["time"].unique())

    # ── ÚNICO CAMBIO: convertir datetime64 → horas desde 1979-01-01 ──
    base = pd.Timestamp("1979-01-01 00:00:00")
    times_hours = (pd.DatetimeIndex(times) - base).total_seconds() / 3600.0
    # ─────────────────────────────────────────────────────────────────

    n_times = len(times)
    n_points = len(points)

    df_all = df_all.set_index(["x", "y", "time"])

    def to_2d(col):
        arr = np.full((n_times, n_points), np.nan)
        for j, (x, y) in enumerate(zip(points["x"], points["y"])):
            arr[:, j] = df_all.loc[(x, y), col].values
        return arr

    ds = xr.Dataset(
        {
            value: (["time", "points"], to_2d(value)) for value in values
        },
        coords={
            "time": times_hours,          # ahora es float64, horas desde 1979
            "latitude": ("points", points["y"].values),
            "longitude": ("points", points["x"].values),
        }
    )

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_nc = output_dir / "TWL.nc"

    ds.to_netcdf(out_nc, format="NETCDF4")

def values_extraction(values_grid_path,values_dir,values,ref_grid_path):
    gdf_points_ref = gpd.read_file(ref_grid_path)
    gdf_points_values = gpd.read_file(values_grid_path)

    gdf_points_ref["idx_closest"] = find_closest_point(gdf_points_values, gdf_points_ref)

    _, files_mat, tree_mat = build_index(values_dir, '.mat')

    all_results = []
    for _, row in gdf_points_ref.iterrows():

        x_point, y_point = row.geometry.x, row.geometry.y

        value_point = gdf_points_values.iloc[row["idx_closest"]]

        x_closest, y_closest = value_point.geometry.x, value_point.geometry.y
        data, _, _ = load_nearest_mat(
            x_closest, y_closest,
            values_dir, tree_mat, files_mat
        )

        results_dict = {
            "time": pd.to_datetime(
                np.squeeze(data["time"]) - 719529,
                unit='D',
                origin='unix'
            )
        }
        for var in values:
            results_dict[var] = np.squeeze(data[var])

        results = pd.DataFrame(results_dict)

        results["x"] = x_point
        results["y"] = y_point
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)

def mean_value_gdf(data,crs):
    gdf_mean = data.groupby(["x", "y"])["tide"].mean().reset_index()
    gdf_mean = gpd.GeoDataFrame(
        gdf_mean,
        geometry=gpd.points_from_xy(gdf_mean["x"], gdf_mean["y"]),
        crs=crs
    )
    return gdf_mean

def trim_to_common_period(*dfs):
    t_0 = max(df["time"].min() for df in dfs)
    t_f = min(df["time"].max() for df in dfs)
    return [
        df[(df["time"] > t_0) & (df["time"] < t_f)].reset_index(drop=True)
        for df in dfs
    ]


def show_grids_satelite_map(
    grids: dict,
    zoom: int,
    gdf_study_area: gpd.GeoDataFrame | None = None,   # <--
    titulo: str | None = None,
    save_path=None,
    figsize=(9, 7),
    provider=cx.providers.Esri.WorldImagery
):
    colors = ["#e63946", "#2a9d8f", "#f4a261"]

    fig, ax = plt.subplots(figsize=figsize)
    all_bounds = []

    # Zona de estudio translúcida (debajo de los grids)
    if gdf_study_area is not None:
        gdf_study_area = gdf_study_area.to_crs(3857)
        gdf_study_area.plot(
            ax=ax,
            color="white",
            edgecolor="white",
            linewidth=1.5,
            alpha=0.2,
            zorder=1,
            label="Zona de estudio",
        )
        all_bounds.append(gdf_study_area.total_bounds)

    for (name, (_, gdf_polygons)), color in zip(grids.items(), colors):
        gdf_polygons = gdf_polygons.to_crs(3857)
        gdf_polygons.plot(
            ax=ax,
            color=color,
            linewidth=0.3,
            edgecolor="black",
            alpha=0.6,
            zorder=2,
            label=name,
        )
        all_bounds.append(gdf_polygons.total_bounds)

    all_bounds = np.array(all_bounds)
    x0 = all_bounds[:, 0].min()
    y0 = all_bounds[:, 1].min()
    x1 = all_bounds[:, 2].max()
    y1 = all_bounds[:, 3].max()

    dx = (x1 - x0)
    dy = (y1 - y0) * 0.10
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_xticks([])
    ax.set_yticks([])

    try:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom, attribution=False)
    except TypeError:
        cx.add_basemap(ax, crs="EPSG:3857", source=provider, zoom=zoom)
        for txt in ax.texts:
            txt.set_visible(False)

    legend_handles = [
        mpatches.Patch(color=color, alpha=0.6, label=name)
        for (name, _), color in zip(grids.items(), colors)
    ]

    if gdf_study_area is not None:
        legend_handles.insert(0, mpatches.Patch(color="white", alpha=0.4, label="Analysis zone"))

    ax.legend(handles=legend_handles, loc="best", fontsize=10)
    if titulo:
        ax.set_title(titulo, fontsize=14, pad=15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✔ Guardado en: {save_path}")

    plt.show()