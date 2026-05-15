from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def extract_monthly_max_high_tide_windows(
    t: pd.DatetimeIndex,
    eta: np.ndarray,
    half_window: int = 300,
    min_peak_dist: int = 8,
    max_nan_frac: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Port de MATLAB:
    [windows, timeRel, t_peak, h_peak, idx_peak, is_valid, year_month] =
        extract_monthly_max_high_tide_windows(t, eta, halfWindow, minPeakDist, maxNanFrac)

    Parámetros
    ----------
    t : pd.DatetimeIndex
        Vector temporal datetime.
    eta : np.ndarray
        Serie temporal Nt x Np. Si se pasa 1D, se interpreta como Nt x 1.
    half_window : int, default=300
        Semiventana en número de muestras.
    min_peak_dist : int, default=8
        Distancia mínima entre picos.
    max_nan_frac : float, default=0.05
        Fracción máxima de NaN permitida dentro de la ventana.

    Devuelve
    --------
    windows : np.ndarray
        Array de forma (2*half_window+1, Nm, Np)
    time_rel : np.ndarray
        Vector relativo [-half_window, ..., half_window]
    t_peak : np.ndarray
        Tiempo del pico mensual máximo, forma (Nm, Np)
    h_peak : np.ndarray
        Valor del pico mensual máximo, forma (Nm, Np)
    idx_peak : np.ndarray
        Índice 1-based estilo MATLAB del pico mensual máximo, forma (Nm, Np)
    is_valid : np.ndarray
        Booleano de validez de cada ventana, forma (Nm, Np)
    year_month : pd.DataFrame
        Tabla con columnas Year y Month
    """

    if not isinstance(t, (pd.DatetimeIndex, pd.Series, np.ndarray, list)):
        raise TypeError("t debe ser un vector datetime.")

    t = pd.DatetimeIndex(t)

    if t.ndim != 1:
        t = pd.DatetimeIndex(np.ravel(t))

    eta = np.asarray(eta)

    if eta.ndim == 1:
        eta = eta.reshape(-1, 1)

    nt = len(t)

    if eta.shape[0] != nt:
        raise ValueError("eta debe tener Nt filas, igual a numel(t).")

    npuntos = eta.shape[1]

    if nt < (2 * half_window + 1):
        raise ValueError("La serie es más corta que la ventana pedida.")

    ym = t.year * 100 + t.month
    months_unique = pd.Series(ym).drop_duplicates().to_numpy()
    month_to_id = {m: i for i, m in enumerate(months_unique)}
    month_id = np.array([month_to_id[m] for m in ym], dtype=int)

    nm = len(months_unique)

    year_month = pd.DataFrame({
        "Year": np.floor_divide(months_unique, 100),
        "Month": np.remainder(months_unique, 100),
    })

    win_length = 2 * half_window + 1
    time_rel = np.arange(-half_window, half_window + 1)

    windows = np.full((win_length, nm, npuntos), np.nan, dtype=float)
    t_peak = np.full((nm, npuntos), np.datetime64("NaT"), dtype="datetime64[ns]")
    h_peak = np.full((nm, npuntos), np.nan, dtype=float)
    idx_peak = np.full((nm, npuntos), np.nan, dtype=float)
    is_valid = np.zeros((nm, npuntos), dtype=bool)

    for j in range(npuntos):
        x = eta[:, j].astype(float)

        if np.all(np.isnan(x)):
            continue

        valid = ~np.isnan(x)

        peak_mask = np.zeros(nt, dtype=bool)
        peak_val = np.full(nt, np.nan, dtype=float)

        d = np.diff(np.r_[False, valid, False].astype(int))
        i_start = np.where(d == 1)[0]
        i_end = np.where(d == -1)[0] - 1

        for b in range(len(i_start)):
            ii = np.arange(i_start[b], i_end[b] + 1)

            if len(ii) < 3:
                continue

            peaks, _ = find_peaks(x[ii], distance=min_peak_dist)

            if len(peaks) > 0:
                locs_abs = ii[peaks]
                peak_mask[locs_abs] = True
                peak_val[locs_abs] = x[locs_abs]

        peak_idx_all = np.where(peak_mask)[0]

        if len(peak_idx_all) == 0:
            continue

        peak_val_all = peak_val[peak_idx_all]
        peak_month_all = month_id[peak_idx_all]

        for m in range(nm):
            indm = peak_month_all == m

            if not np.any(indm):
                continue

            idxm = peak_idx_all[indm]
            pkm = peak_val_all[indm]

            k = int(np.argmax(pkm))
            hmax = float(pkm[k])
            imax = int(idxm[k])

            t_peak[m, j] = np.datetime64(t[imax].to_datetime64())
            h_peak[m, j] = hmax
            idx_peak[m, j] = imax + 1  # 1-based, como MATLAB

            i1 = imax - half_window
            i2 = imax + half_window

            if i1 < 0 or i2 >= nt:
                continue

            seg = x[i1:i2 + 1]
            nan_frac = np.mean(np.isnan(seg))

            if nan_frac <= max_nan_frac:
                windows[:, m, j] = seg
                is_valid[m, j] = True

    return windows, time_rel, t_peak, h_peak, idx_peak, is_valid, year_month