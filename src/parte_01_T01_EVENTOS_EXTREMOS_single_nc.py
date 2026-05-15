from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.io import savemat
import os

sys.path.append(os.path.dirname(__file__))
from dependencias.POT_extremos_v2 import pot_extremos
from dependencias.contorno_bivariante_avanzado_v2 import contorno_bivariante_avanzado_v2
from dependencias.ajuste_potencial_v2 import ajuste_potencial_v2

import time
import traceback
import csv

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)  # resetea a valores por defecto


# =========================================================
# CONFIG  (sobreescritos por argparse en tiempo de ejecución)
# =========================================================
PATH_SAVE = Path(r"C:\Users\santamariace\Desktop\TODO")

TRET = np.array([1, 5, 10, 30, 50, 70, 100, 200, 500], dtype=np.float64)

# como idx_maxX está en estilo MATLAB -> 1; sino 0
POT_IDX_BASE = 1

# generar figuras
MAKE_PLOTS = True

# outputs de prints
DEBUG_INFO = False


# =========================================================
# HELPERS  (sin cambios respecto a v4)
# =========================================================

def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _structured_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace(" ", "_")


def emit_structured_log(tag: str, **fields: Any) -> None:
    tokens = [tag]
    for key, value in fields.items():
        if value is None:
            continue
        tokens.append(f"{key}={_structured_value(value)}")
    print(" ".join(tokens), flush=True)


def build_point_log_fields(
    mesh: str,
    subdomain: str,
    j: int,
    lat: float,
    lon: float,
    point_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if point_meta:
        fields.update(point_meta)
    fields.update({"j": j, "mesh": mesh, "subdomain": subdomain, "lat": lat, "lon": lon})
    return fields


def emit_point_stage(
    mesh: str,
    subdomain: str,
    j: int,
    lat: float,
    lon: float,
    stage: str,
    point_meta: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    fields = build_point_log_fields(mesh, subdomain, j, lat, lon, point_meta)
    fields["stage"] = stage
    fields.update(extra)
    emit_structured_log("POINT_STAGE", **fields)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def estimate_eta(elapsed: float, done: int, total: int) -> float | None:
    if done <= 0 or total <= 0:
        return None
    rate = elapsed / done
    remaining = total - done
    return rate * remaining


def matlab_prctile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x = np.sort(x)

    if x.size == 0:
        raise ValueError("No hay datos finitos para calcular percentil.")

    p = q / 100.0
    n = x.size
    h = n * p + 0.5

    if h <= 1:
        return float(x[0])
    if h >= n:
        return float(x[-1])

    hf = int(np.floor(h))
    hc = int(np.ceil(h))

    if hf == hc:
        return float(x[hf - 1])

    x0 = x[hf - 1]
    x1 = x[hc - 1]
    return float(x0 + (h - hf) * (x1 - x0))


def matlab_datenum_from_hours_since_1979(hours_from_1979: np.ndarray) -> np.ndarray:
    """datenum(datetime(1979,1,1,0,0,0) + hours(time))"""
    base = pd.Timestamp("1979-01-01 00:00:00")
    dt = base + pd.to_timedelta(np.asarray(hours_from_1979, dtype=np.float64), unit="h")

    ordinal = np.array([d.to_pydatetime().toordinal() for d in dt], dtype=np.float64)
    frac = (
        dt.hour.to_numpy()
        + dt.minute.to_numpy() / 60.0
        + dt.second.to_numpy() / 3600.0
        + dt.microsecond.to_numpy() / 3.6e9
    ) / 24.0

    return ordinal + frac + 366.0


def safe_num_for_name(x: float, ndigits: int = 4) -> str:
    return f"{float(x):.{ndigits}f}".rstrip("0").rstrip(".")


def save_mat(filepath: Path, var_name: str, obj: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    savemat(filepath, {var_name: obj}, do_compression=True)


def event_field_list(eventos: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.array([ev[field] for ev in eventos], dtype=np.float64)


def maybe_remove_figs(eventos_out: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(eventos_out)
    out.pop("fig_full", None)
    out.pop("fig_zoom", None)
    return out


def save_eventos_figures(
    eventos_out: dict[str, Any],
    out_dir: Path,
    mesh: str,
    subdomain: str,
    lat: float,
    lon: float,
) -> None:
    if not MAKE_PLOTS:
        return

    lat_s = safe_num_for_name(lat)
    lon_s = safe_num_for_name(lon)

    fig_full = eventos_out.get("fig_full", None)
    fig_zoom = eventos_out.get("fig_zoom", None)

    title_txt = f"{mesh}_{subdomain}, Lat={lat:.4f}° ; Lon={lon:.4f}°"

    if fig_full is not None:
        ax = fig_full.axes[0]
        ax.set_title(title_txt)
        ax.set_xlabel("max Hs (m)")
        ax.set_ylabel("max SS (m)")
        fig_full.savefig(
            out_dir / f"Eventos_Extremos_{mesh}_{subdomain}_Lat_{lat_s}_Lon_{lon_s}.png",
            dpi=600,
            bbox_inches="tight",
        )

    if fig_zoom is not None:
        ax = fig_zoom.axes[0]
        ax.set_title(title_txt)
        ax.set_xlabel("max Hs (m)")
        ax.set_ylabel("max SS (m)")
        fig_zoom.savefig(
            out_dir / f"Eventos_Extremos_{mesh}_{subdomain}_Lat_{lat_s}_Lon_{lon_s}_zoom.png",
            dpi=600,
            bbox_inches="tight",
        )


def save_modelo_tp_figure(
    modelo_tp: dict[str, Any],
    out_dir: Path,
    mesh: str,
    subdomain: str,
    lat: float,
    lon: float,
) -> None:
    if not MAKE_PLOTS:
        return

    lat_s = safe_num_for_name(lat)
    lon_s = safe_num_for_name(lon)

    fig = modelo_tp.get("fig", None) if isinstance(modelo_tp, dict) else None

    if fig is None or len(fig.axes) == 0:
        print("WARNING: no se encontró una figura válida para Modelo_Tp.")
        return

    ax = fig.axes[0]
    ax.set_title(f"{mesh}_{subdomain}, Lat={lat:.4f}° ; Lon={lon:.4f}°")
    ax.set_xlabel("Hs (m)")
    ax.set_ylabel("Tp (s)")

    fig.savefig(
        out_dir / f"Modelo_Tp_{mesh}_{subdomain}_Lat_{lat_s}_Lon_{lon_s}.png",
        dpi=600,
        bbox_inches="tight",
    )


def save_identificacion_eventos_figure(
    muestra_eventos: list[dict[str, Any]],
    umbral1: float,
    umbral2: float,
    out_dir: Path,
    mesh: str,
    subdomain: str,
    lat: float,
    lon: float,
) -> None:
    if not MAKE_PLOTS:
        return

    lat_s = safe_num_for_name(lat)
    lon_s = safe_num_for_name(lon)

    maxX = np.array([ev["maxX"] for ev in muestra_eventos], dtype=np.float64)
    maxSurge = np.array([ev["maxSurge"] for ev in muestra_eventos], dtype=np.float64)

    fig, ax = plt.subplots(num=4, clear=True)
    ax.plot(maxX, maxSurge, "k.", markersize=4)
    ax.set_title(f"{mesh}_{subdomain}, Lat={lat:.4f}° ; Lon={lon:.4f}°")
    ax.set_xlabel("max WP (kW/m)")
    ax.set_ylabel("max SS (m)")
    ax.grid(True)

    ymin = float(np.min(maxSurge) - 0.1)
    ymax = float(np.max(maxSurge) + 0.1)
    xmax = float(np.max(maxX) + 10.0)

    ax.plot([umbral1, umbral1], [ymin, ymax], "-r", linewidth=2)
    ax.plot([0, xmax], [umbral2, umbral2], "-r", linewidth=2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)

    fig.savefig(
        out_dir / f"Identificacion_Eventos_{mesh}_{subdomain}_Lat_{lat_s}_Lon_{lon_s}.png",
        dpi=600,
        bbox_inches="tight",
    )


# =========================================================
# PROCESAMIENTO POR PUNTO  (sin cambios respecto a v4)
# =========================================================

def process_one_point(
    data: dict[str, np.ndarray],
    mesh: str,
    subdomain: str,
    j: int,
    point_meta: dict[str, Any] | None = None,
) -> None:
    lat = float(data["latitude"][j])
    lon = float(data["longitude"][j])

    if DEBUG_INFO:
        print(f"j={j}")
        print("WP row shape:", data["WP"][j, :].shape)
        print("WP finite count:", np.isfinite(data["WP"][j, :]).sum())
        print("hs finite count:", np.isfinite(data["hs"][j, :]).sum())
        print("tp finite count:", np.isfinite(data["tp"][j, :]).sum())
        print("surge finite count:", np.isfinite(data["surge"][j, :]).sum())

    emit_point_stage(mesh, subdomain, j, lat, lon, "thresholds_start", point_meta)
    umbral1 = matlab_prctile(data["WP"][j, :], 98.0)
    umbral2 = matlab_prctile(data["surge"][j, :], 98.0)

    emit_point_stage(mesh, subdomain, j, lat, lon, "pot_start", point_meta)
    muestra_eventos = pot_extremos(
        data["time_dnum"],
        data["WP"][j, :],
        data["surge"][j, :],
        umbral1,
        umbral2,
        3,
    )
    emit_point_stage(
        mesh, subdomain, j, lat, lon, "pot_done", point_meta, n_events=len(muestra_eventos)
    )

    for ev in muestra_eventos:
        idx = int(ev["idx_maxX"])
        if POT_IDX_BASE == 1:
            idx -= 1
        ev["Tp"] = float(data["tp"][j, idx])
        ev["Hs"] = float(data["hs"][j, idx])

    emit_point_stage(mesh, subdomain, j, lat, lon, "contorno_start", point_meta)
    eventos = contorno_bivariante_avanzado_v2(
        muestra_eventos,
        data["time_dnum"],
        TRET,
        make_plots=MAKE_PLOTS,
    )
    emit_point_stage(mesh, subdomain, j, lat, lon, "contorno_done", point_meta)

    eventos["lat"] = lat
    eventos["lon"] = lon

    hs_event = np.array([ev["Hs"] for ev in muestra_eventos], dtype=np.float64)
    tp_event = np.array([ev["Tp"] for ev in muestra_eventos], dtype=np.float64)
    hs_mpp = np.array(
        [r["mpp"]["Hs"] for r in eventos["results"] if r["Tret"] is not None],
        dtype=np.float64,
    )

    emit_point_stage(
        mesh, subdomain, j, lat, lon, "modelo_tp_start", point_meta, mpp_count=hs_mpp.size
    )
    modelo_tp = ajuste_potencial_v2(hs_event, tp_event, hs_mpp, 0.1)
    emit_point_stage(mesh, subdomain, j, lat, lon, "modelo_tp_done", point_meta)

    ypred_high = np.asarray(modelo_tp["Ypred_high"], dtype=np.float64).reshape(-1)
    kk = 0
    for r in eventos["results"]:
        if r["Tret"] is not None:
            r["mpp"]["Tp"] = float(ypred_high[kk])
            kk += 1

    lat_s = safe_num_for_name(lat)
    lon_s = safe_num_for_name(lon)

    out_dir = PATH_SAVE / mesh / subdomain / f"Lat_{lat_s}_Lon_{lon_s}"
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_point_stage(mesh, subdomain, j, lat, lon, "save_start", point_meta)

    save_eventos_figures(eventos, out_dir, mesh, subdomain, lat, lon)
    save_modelo_tp_figure(modelo_tp, out_dir, mesh, subdomain, lat, lon)
    if "fig" in modelo_tp and modelo_tp["fig"] is not None:
        plt.close(modelo_tp["fig"])
    save_identificacion_eventos_figure(
        muestra_eventos, umbral1, umbral2, out_dir, mesh, subdomain, lat, lon
    )

    eventos_no_fig = maybe_remove_figs(eventos)
    save_mat(
        out_dir / f"Eventos_Lat_{lat_s}_Lon_{lon_s}.mat",
        "Eventos",
        eventos_no_fig,
    )
    save_mat(
        out_dir / f"Muestra_Eventos_Lat_{lat_s}_Lon_{lon_s}.mat",
        "Muestra_eventos",
        muestra_eventos,
    )

    modelo_tp_no_fig = copy.deepcopy(modelo_tp)
    modelo_tp_no_fig.pop("fig", None)
    save_mat(
        out_dir / f"Modelo_Tm02_Lat_{lat_s}_Lon_{lon_s}.mat",
        "Modelo_Tp",
        modelo_tp_no_fig,
    )

    emit_point_stage(mesh, subdomain, j, lat, lon, "save_done", point_meta)
    plt.close("all")


# =========================================================
# ARGPARSE  (simplificado: un único --nc-file en lugar de
#            --path-root + --start-i + --end-i + --csv-name)
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecuta el análisis de eventos extremos sobre un único fichero NetCDF.",
    )
    parser.add_argument(
        "--nc-file",
        type=Path,
        required=True,
        help="Ruta al fichero NetCDF de entrada (p. ej. TemporalSeries_Hindcast_MALLA_SUB.nc).",
    )
    parser.add_argument(
        "--path-save",
        type=Path,
        default=PATH_SAVE,
        help="Ruta donde guardar los resultados.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help=(
            "Etiqueta 'Mesh' usada en nombres de ficheros de salida. "
            "Si se omite se usa el stem del fichero .nc."
        ),
    )
    parser.add_argument(
        "--subdomain",
        type=str,
        default=None,
        help=(
            "Etiqueta 'Subdomain' usada en nombres de ficheros de salida. "
            "Si se omite se usa 'default'."
        ),
    )
    parser.add_argument(
        "--j-start",
        type=int,
        default=0,
        help="Índice Python inicial (0-based, inclusive) de los puntos a procesar.",
    )
    parser.add_argument(
        "--j-end",
        type=int,
        default=None,
        help="Índice Python final (0-based, inclusive) de los puntos a procesar. Por defecto, todos.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emitir una línea de progreso cada N puntos procesados.",
    )
    parser.add_argument(
        "--pot-idx-base",
        type=int,
        choices=(0, 1),
        default=POT_IDX_BASE,
        help="Base del idx_maxX devuelto por POT_extremos_v2 (0 ó 1).",
    )
    parser.add_argument(
        "--debug-info",
        action="store_true",
        default=DEBUG_INFO,
        help="Activa trazas extra por consola.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--make-plots",
        dest="make_plots",
        action="store_true",
        help="Genera PNGs durante la ejecución.",
    )
    plot_group.add_argument(
        "--no-make-plots",
        dest="make_plots",
        action="store_false",
        help="No genera PNGs durante la ejecución.",
    )
    parser.set_defaults(make_plots=MAKE_PLOTS)
    return parser


def configure_runtime(args: argparse.Namespace) -> None:
    global PATH_SAVE, MAKE_PLOTS, DEBUG_INFO, POT_IDX_BASE
    PATH_SAVE = args.path_save
    MAKE_PLOTS = args.make_plots
    DEBUG_INFO = args.debug_info
    POT_IDX_BASE = args.pot_idx_base


def point_index_range(j_start: int, j_end: int | None, npoints: int) -> range:
    if npoints <= 0:
        raise ValueError("No hay puntos disponibles para procesar.")
    if j_start < 0:
        raise ValueError("--j-start debe ser >= 0.")
    if j_start >= npoints:
        raise ValueError(f"--j-start={j_start} excede el número de puntos ({npoints}).")

    j_end_eff = npoints - 1 if j_end is None else j_end
    if j_end_eff < j_start:
        raise ValueError("--j-end debe ser >= --j-start.")
    if j_end_eff >= npoints:
        raise ValueError(f"--j-end={j_end_eff} excede el número de puntos ({npoints}).")

    return range(j_start, j_end_eff + 1)


# =========================================================
# MAIN
# =========================================================

def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = build_arg_parser().parse_args(argv)
    configure_runtime(args)

    nc_path: Path = args.nc_file
    if not nc_path.exists():
        print(f"ERROR: No se encuentra el fichero NetCDF: {nc_path}", file=sys.stderr)
        return 2

    # Derivar etiquetas de naming si no se han proporcionado
    mesh: str = args.mesh if args.mesh is not None else nc_path.stem
    subdomain: str = args.subdomain if args.subdomain is not None else "default"

    t0_global = time.perf_counter()
    all_failed_points: list[dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))

    print(f"Fichero NetCDF: {nc_path}")
    print(f"mesh={mesh}  subdomain={subdomain}")

    # ------------------------------------------------------------------
    # Carga del NetCDF (idéntica a la lógica del bucle en v4)
    # ------------------------------------------------------------------
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        latitude = ds["latitude"].values
        longitude = ds["longitude"].values
        hs = ds["hs"].values
        tm02 = ds["tm02"].values
        tp = ds["tp"].values
        direc = ds["dir"].values
        tide = ds["tide"].values
        surge = ds["surge"].values
        time_hours = ds["time"].values

        hs = hs.T
        tm02 = tm02.T
        tp = tp.T
        direc = direc.T
        tide = tide.T
        surge = surge.T

        if DEBUG_INFO:
            print("latitude shape:", np.shape(latitude))
            print("longitude shape:", np.shape(longitude))
            print("hs shape:", np.shape(hs))
            print("tp shape:", np.shape(tp))
            print("surge shape:", np.shape(surge))
            print("time shape:", np.shape(time_hours))
            print("time dtype:", time_hours.dtype)
            print("time min/max:", np.min(time_hours), np.max(time_hours))

    time_dnum = matlab_datenum_from_hours_since_1979(time_hours)

    # Compatibilizar: usar columnas donde surge(1,:) no es NaN
    pos = np.where(~np.isnan(surge[0, :]))[0]
    hs = hs[:, pos]
    tm02 = tm02[:, pos]
    tp = tp[:, pos]
    direc = direc[:, pos]
    tide = tide[:, pos]
    surge = surge[:, pos]
    time_dnum = time_dnum[pos]

    WP = 0.49 * tp * hs ** 2

    data = {
        "latitude": np.asarray(latitude, dtype=np.float64),
        "longitude": np.asarray(longitude, dtype=np.float64),
        "hs": np.asarray(hs, dtype=np.float64),
        "tm02": np.asarray(tm02, dtype=np.float64),
        "tp": np.asarray(tp, dtype=np.float64),
        "dir": np.asarray(direc, dtype=np.float64),
        "tide": np.asarray(tide, dtype=np.float64),
        "surge": np.asarray(surge, dtype=np.float64),
        "time_dnum": np.asarray(time_dnum, dtype=np.float64),
        "WP": np.asarray(WP, dtype=np.float64),
    }

    j_range = point_index_range(args.j_start, args.j_end, len(data["latitude"]))
    total_j = len(j_range)
    points_done = 0

    print(f"Rango j Python: {j_range.start}:{j_range.stop - 1}  (total={total_j})")
    print(f"START_TIME total timer launched")

    # ------------------------------------------------------------------
    # Bucle sobre puntos
    # ------------------------------------------------------------------
    for j_pos, j in enumerate(j_range, start=1):
        t0_point = time.perf_counter()
        lat = float(data["latitude"][j])
        lon = float(data["longitude"][j])
        point_meta = {"j_pos": j_pos, "total_j": total_j}

        try:
            emit_structured_log(
                "POINT_START",
                **build_point_log_fields(mesh, subdomain, j, lat, lon, point_meta),
            )
            process_one_point(data, mesh, subdomain, j, point_meta=point_meta)
            point_elapsed = time.perf_counter() - t0_point
            points_done += 1
            emit_structured_log(
                "POINT_DONE",
                **build_point_log_fields(mesh, subdomain, j, lat, lon, point_meta),
                elapsed_point=format_seconds(point_elapsed),
            )

            if j_pos == 1 or j_pos == total_j or (j_pos % progress_every == 0):
                elapsed_global = time.perf_counter() - t0_global
                eta_seconds = estimate_eta(elapsed_global, points_done, total_j)
                pct = 100.0 * points_done / total_j if total_j > 0 else 100.0
                eta_txt = format_seconds(eta_seconds) if eta_seconds is not None else "N/A"
                print(
                    f"PROGRESS points_done={points_done} total_points={total_j} "
                    f"pct={pct:.2f} elapsed={format_seconds(elapsed_global)} eta={eta_txt} "
                    f"last_point={format_seconds(point_elapsed)} "
                    f"j_pos={j_pos} total_j={total_j} mesh={mesh} subdomain={subdomain}"
                )

        except Exception as exc:
            point_elapsed = time.perf_counter() - t0_point
            lat_txt, lon_txt = "nan", "nan"
            try:
                lat_txt = f"{float(data['latitude'][j]):.6f}"
                lon_txt = f"{float(data['longitude'][j]):.6f}"
            except Exception:
                pass

            err_info = {
                "j": j,
                "j_pos": j_pos,
                "total_j": total_j,
                "mesh": mesh,
                "subdomain": subdomain,
                "lat": lat_txt,
                "lon": lon_txt,
                "elapsed_point": format_seconds(point_elapsed),
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
            }
            all_failed_points.append(err_info)

            print(
                f"POINT_ERROR j={j} j_pos={j_pos} total_j={total_j} "
                f"mesh={mesh} subdomain={subdomain} lat={lat_txt} lon={lon_txt} "
                f"elapsed_point={format_seconds(point_elapsed)} "
                f"error_type={type(exc).__name__} error_msg={str(exc)}"
            )
            print("POINT_ERROR_TRACEBACK_START")
            print(traceback.format_exc().rstrip())
            print("POINT_ERROR_TRACEBACK_END")
            continue

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    n_failed = len(all_failed_points)
    n_attempted = points_done + n_failed
    total_elapsed = time.perf_counter() - t0_global

    print(
        f"SUMMARY ok_points={points_done} failed_points={n_failed} "
        f"attempted_points={n_attempted} "
        f"pct_ok={(100.0 * points_done / total_j) if total_j > 0 else 100.0:.2f} "
        f"total_elapsed={format_seconds(total_elapsed)}"
    )

    if n_failed > 0:
        for k, err in enumerate(all_failed_points, start=1):
            print(
                f"FAILED_POINT idx={k} j={err['j']} j_pos={err['j_pos']} total_j={err['total_j']} "
                f"mesh={err['mesh']} subdomain={err['subdomain']} "
                f"lat={err['lat']} lon={err['lon']} "
                f"error_type={err['error_type']} error_msg={err['error_msg']}"
            )

        error_dir = PATH_SAVE / "_worker_errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_file = error_dir / f"errors_points_{mesh}_{subdomain}.csv"
        with error_file.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "j", "j_pos", "total_j", "mesh", "subdomain",
                    "lat", "lon", "elapsed_point", "error_type", "error_msg",
                ],
            )
            writer.writeheader()
            writer.writerows(all_failed_points)
        print(f"ERROR_REPORT path={error_file} n_failed={n_failed}")

    print(
        f"DONE points_done={points_done} failed_points={n_failed} "
        f"total_points={total_j} total_elapsed={format_seconds(total_elapsed)}"
    )

    return 1 if n_failed > 0 else 0

def run(
    nc_file: str | Path,
    path_save: str | Path = Path(r"C:\Users\santamariace\Desktop\TODO"),
    mesh: str | None = None,
    subdomain: str | None = None,
    j_start: int = 0,
    j_end: int | None = None,
    tret: np.ndarray | list | None = None,
    make_plots: bool = True,
    debug_info: bool = False,
    progress_every: int = 25,
    pot_idx_base: int = 1,
) -> int:
    """Punto de entrada para uso desde notebook o scripts Python."""
    global TRET

    if tret is not None:
        TRET = np.asarray(tret, dtype=np.float64)

    argv = [
        "--nc-file", str(nc_file),
        "--path-save", str(path_save),
        "--progress-every", str(progress_every),
        "--pot-idx-base", str(pot_idx_base),
        "--j-start", str(j_start),
        "--make-plots" if make_plots else "--no-make-plots",
    ]
    if mesh is not None:
        argv += ["--mesh", mesh]
    if subdomain is not None:
        argv += ["--subdomain", subdomain]
    if j_end is not None:
        argv += ["--j-end", str(j_end)]
    if debug_info:
        argv += ["--debug-info"]

    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
