from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from dependencias.seleccionar_umbral_gpd import seleccionar_umbral_gpd
from dependencias.hybrid_cdf import hybrid_cdf
from dependencias.hybrid_pdf import hybrid_pdf
from dependencias.comparar_copulas import comparar_copulas
from dependencias.hybrid_icdf import hybrid_icdf
from dependencias.make_strictly_increasing import make_strictly_increasing
from dependencias.eval_copulacdf import eval_copulacdf
from dependencias.contour_matrix_to_xy_longest import contour_matrix_to_xy_longest
from dependencias.regularizar_contorno_monotono import regularizar_contorno_monotono
from dependencias.eval_copulapdf import eval_copulapdf


def _gpd_negloglik(params: np.ndarray, exc: np.ndarray) -> float:
    k, sigma = params

    if sigma <= 0:
        return np.inf

    exc = np.asarray(exc, dtype=np.float64)

    if abs(k) < 1e-10:
        return exc.size * np.log(sigma) + np.sum(exc) / sigma

    t = 1.0 + (k * exc) / sigma
    if np.any(t <= 0):
        return np.inf

    return exc.size * np.log(sigma) + (1.0 + 1.0 / k) * np.sum(np.log(t))


def gpfit_like_matlab(exc: np.ndarray) -> tuple[float, float]:

    exc = np.asarray(exc, dtype=np.float64).reshape(-1)
    exc = exc[np.isfinite(exc)]

    if exc.size == 0:
        raise ValueError("No hay excesos válidos para ajustar GPD.")

    x0 = np.array([0.1, max(np.mean(exc), 1e-8)], dtype=np.float64)

    res = minimize(
        _gpd_negloglik,
        x0=x0,
        args=(exc,),
        method="Nelder-Mead",
        options={
            "maxiter": 10000,
            "xatol": 1e-10,
            "fatol": 1e-10,
        },
    )

    if not res.success:
        raise RuntimeError(f"Falló gpfit_like_matlab: {res.message}")

    k, sigma = res.x
    return float(k), float(sigma)


def _extract_event_field(eventos: Any, field: str) -> np.ndarray:

    vals = []
    for ev in eventos:
        if isinstance(ev, dict):
            vals.append(ev[field])
        else:
            vals.append(getattr(ev, field))
    return np.asarray(vals, dtype=np.float64).reshape(-1)


def _datenum_like(timevec: Any) -> np.ndarray:

    t = np.asarray(timevec)

    if np.issubdtype(t.dtype, np.number):
        return t.astype(np.float64).reshape(-1)

    # datetime64 -> días
    t = t.astype("datetime64[ns]").reshape(-1)
    t0 = t.astype("datetime64[ns]").astype(np.int64) / 1e9 / 86400.0
    return t0.astype(np.float64)


def _matlab_lines(n: int) -> np.ndarray:

    return plt.cm.tab10(np.linspace(0, 1, max(n, 3)))[:, :3]


def _build_contour_matrix_from_matplotlib(x: np.ndarray, y: np.ndarray, z: np.ndarray, level: float) -> np.ndarray:

    fig = plt.figure()
    try:
        cs = plt.contour(x, y, z, levels=[level])
        segments = []

        if len(cs.allsegs) == 0 or len(cs.allsegs[0]) == 0:
            return np.empty((2, 0), dtype=np.float64)

        for seg in cs.allsegs[0]:
            if seg.shape[0] == 0:
                continue
            npts = seg.shape[0]
            block = np.empty((2, npts + 1), dtype=np.float64)
            block[0, 0] = level
            block[1, 0] = npts
            block[0, 1:] = seg[:, 0]
            block[1, 1:] = seg[:, 1]
            segments.append(block)

        if not segments:
            return np.empty((2, 0), dtype=np.float64)

        return np.hstack(segments)

    finally:
        plt.close(fig)


def contorno_bivariante_avanzado_v2(
    eventos: Any,
    timevec: Any,
    Tret: Any = None,
    make_plots: bool = True,
) -> dict[str, Any]:


    if Tret is None or (hasattr(Tret, "__len__") and len(np.atleast_1d(Tret)) == 0):
        Tret = 1

    Tret = np.asarray(Tret, dtype=np.float64).reshape(-1)
    nT = Tret.size

    X = _extract_event_field(eventos, "Hs")
    Y = _extract_event_field(eventos, "maxSurge")

    ok = np.isfinite(X) & np.isfinite(Y)
    X = X[ok]
    Y = Y[ok]

    N = X.size
    if N < 20:
        print(f"Warning: pocos eventos ({N}). El ajuste puede ser inestable.")

    tnum = _datenum_like(timevec)
    nyears = (np.max(tnum) - np.min(tnum)) / 365.25
    lambda_ = N / nyears

    qgrid = np.arange(80, 100, 2, dtype=np.float64)

    fitX = seleccionar_umbral_gpd(X, qgrid)
    fitY = seleccionar_umbral_gpd(Y, qgrid)

    uX = float(fitX["u"])
    uY = float(fitY["u"])

    excX = X[X > uX] - uX
    excY = Y[Y > uY] - uY

    kX, sX = gpfit_like_matlab(excX)
    kY, sY = gpfit_like_matlab(excY)

    pX_u = float(np.mean(X > uX))
    pY_u = float(np.mean(Y > uY))

    def Fx(x):
        return hybrid_cdf(x, X, uX, pX_u, kX, sX)

    def Fy(y):
        return hybrid_cdf(y, Y, uY, pY_u, kY, sY)

    def fx(x):
        return hybrid_pdf(x, X, uX, pX_u, kX, sX)

    def fy(y):
        return hybrid_pdf(y, Y, uY, pY_u, kY, sY)

    epsu = 1e-8
    Ux = np.clip(np.asarray(Fx(X), dtype=np.float64).reshape(-1), epsu, 1 - epsu)
    Uy = np.clip(np.asarray(Fy(Y), dtype=np.float64).reshape(-1), epsu, 1 - epsu)

    U = np.column_stack([Ux, Uy])

    modelos = ["Gaussian", "t", "Gumbel", "Clayton", "Frank"]
    tablaCop = comparar_copulas(U, modelos)

    if len(tablaCop) == 0:
        raise RuntimeError("No se pudo ajustar ninguna cópula.")

    ibest = int(np.argmin(tablaCop["AIC"].to_numpy()))
    bestName = str(tablaCop.iloc[ibest]["Modelo"])
    bestPar = tablaCop.iloc[ibest]["Param"]

    p = np.unique(np.concatenate([
        np.linspace(1e-5,   1e-3,   140),
        np.linspace(1e-3,   0.02,   140),
        np.linspace(0.02,   0.94,   460),
        np.linspace(0.94,   0.999,  380),
        np.linspace(0.999,  0.99999, 340),
    ])).astype(np.float64)

    xq = np.array([hybrid_icdf(pp, X, uX, pX_u, kX, sX) for pp in p], dtype=np.float64)
    yq = np.array([hybrid_icdf(pp, Y, uY, pY_u, kY, sY) for pp in p], dtype=np.float64)

    xq = np.asarray(make_strictly_increasing(xq), dtype=np.float64).reshape(-1)
    yq = np.asarray(make_strictly_increasing(yq), dtype=np.float64).reshape(-1)

    PX, PY = np.meshgrid(p, p)
    C = eval_copulacdf(bestName, np.column_stack([PX.ravel(), PY.ravel()]), bestPar)
    C = np.asarray(C, dtype=np.float64).reshape(PX.shape)

    results: list[dict[str, Any]] = [
        {"Tret": None, "p_or": None, "clevel": None, "contour": None, "mpp": None}
        for _ in range(nT)
    ]

    cmap = _matlab_lines(max(nT, 3))

    all_xc = []
    all_yc = []
    all_mppx = []
    all_mppy = []

    for it in range(nT):
        T = float(Tret[it])

        p_or = 1.0 / (lambda_ * T)
        if p_or >= 1:
            print(f"Warning: se omite Tret = {T:.3f} porque lambda*Tret <= 1.")
            continue

        clevel = 1.0 - p_or

        Cmat = _build_contour_matrix_from_matplotlib(xq, yq, C, clevel)
        xc_raw, yc_raw = contour_matrix_to_xy_longest(Cmat)

        xc_raw = np.asarray(xc_raw, dtype=np.float64).reshape(-1)
        yc_raw = np.asarray(yc_raw, dtype=np.float64).reshape(-1)

        if xc_raw.size == 0:
            print(f"Warning: no se pudo extraer el contorno para Tret = {T:.3f}.")
            continue

        xc, yc = regularizar_contorno_monotono(xc_raw, yc_raw, 300)
        xc = np.asarray(xc, dtype=np.float64).reshape(-1)
        yc = np.asarray(yc, dtype=np.float64).reshape(-1)

        uc = np.clip(np.asarray(Fx(xc), dtype=np.float64).reshape(-1), epsu, 1 - epsu)
        vc = np.clip(np.asarray(Fy(yc), dtype=np.float64).reshape(-1), epsu, 1 - epsu)

        cden = np.asarray(
            eval_copulapdf(bestName, np.column_stack([uc, vc]), bestPar),
            dtype=np.float64
        ).reshape(-1)

        dens = (
            cden
            * np.asarray(fx(xc), dtype=np.float64).reshape(-1)
            * np.asarray(fy(yc), dtype=np.float64).reshape(-1)
        )

        imax = int(np.argmax(dens))

        mpp = {
            "Hs": float(xc[imax]),
            "SS": float(yc[imax]),
        }

        contour_struct = {
            "raw_x": xc_raw.reshape(-1, 1),
            "raw_y": yc_raw.reshape(-1, 1),
            "x": xc.reshape(-1, 1),
            "y": yc.reshape(-1, 1),
        }

        results[it] = {
            "Tret": T,
            "p_or": p_or,
            "clevel": clevel,
            "contour": contour_struct,
            "mpp": mpp,
        }

        all_xc.append(xc.reshape(-1, 1))
        all_yc.append(yc.reshape(-1, 1))
        all_mppx.append(mpp["Hs"])
        all_mppy.append(mpp["SS"])

    all_xc_cat = np.vstack(all_xc) if all_xc else np.empty((0, 1), dtype=np.float64)
    all_yc_cat = np.vstack(all_yc) if all_yc else np.empty((0, 1), dtype=np.float64)
    all_mppx_arr = np.asarray(all_mppx, dtype=np.float64).reshape(-1, 1)
    all_mppy_arr = np.asarray(all_mppy, dtype=np.float64).reshape(-1, 1)

    fig1 = None
    fig2 = None

    if make_plots:
        fig1, ax1 = plt.subplots()
        ax1.plot(X, Y, "k.", markersize=4) #lo bajo de 9 a 4 para que se asemeje al output de matlab.

        legend_handles = []
        legend_entries = []

        for it in range(nT):
            if results[it]["Tret"] is None:
                continue

            xc = results[it]["contour"]["x"].reshape(-1)
            yc = results[it]["contour"]["y"].reshape(-1)
            mpp = results[it]["mpp"]
            T = results[it]["Tret"]

            (hline,) = ax1.plot(xc, yc, linewidth=2, color=cmap[it, :])
            ax1.plot(
                mpp["Hs"], mpp["SS"], "o",
                color=cmap[it, :],
                markerfacecolor=cmap[it, :],
                markersize=7,
            )

            legend_handles.append(hline)
            legend_entries.append(f"T={T:g} años")

        if all_xc_cat.size > 0 and all_yc_cat.size > 0:
            xspan = np.concatenate([X.reshape(-1, 1), all_xc_cat], axis=0).reshape(-1)
            yspan = np.concatenate([Y.reshape(-1, 1), all_yc_cat], axis=0).reshape(-1)

            xmargin = 0.08 * np.ptp(xspan)
            ymargin = 0.08 * np.ptp(yspan)

            if xmargin == 0:
                xmargin = 0.1
            if ymargin == 0:
                ymargin = 0.1

            ax1.set_xlim(np.min(xspan) - xmargin, np.max(xspan) + xmargin)
            ax1.set_ylim(np.min(yspan) - ymargin, np.max(yspan) + ymargin)

        ax1.set_xlabel("maxX")
        ax1.set_ylabel("maxSurge")
        ax1.set_title(
            f"Contorno OR, cópula = {bestName}" if nT == 1
            else f"Contornos OR, cópula = {bestName}"
        )
        ax1.grid(True)
        if legend_handles:
            ax1.legend(legend_handles, legend_entries, loc="best")

        fig2, ax2 = plt.subplots()
        ax2.plot(X, Y, "k.", markersize=4) #lo bajo de 12 a 4 para que se asemeje al output de matlab

        legend_handles_zoom = []
        legend_entries_zoom = []

        for it in range(nT):
            if results[it]["Tret"] is None:
                continue

            xc = results[it]["contour"]["x"].reshape(-1)
            yc = results[it]["contour"]["y"].reshape(-1)
            mpp = results[it]["mpp"]
            T = results[it]["Tret"]

            (hline,) = ax2.plot(xc, yc, linewidth=2.2, color=cmap[it, :])
            ax2.plot(
                mpp["Hs"], mpp["SS"], "o",
                color=cmap[it, :],
                markerfacecolor=cmap[it, :],
                markersize=8,
            )

            legend_handles_zoom.append(hline)
            legend_entries_zoom.append(f"T={T:g} años")

        if all_xc_cat.size > 0 and all_yc_cat.size > 0:
            xzoom = np.concatenate([all_xc_cat.reshape(-1), all_mppx_arr.reshape(-1)])
            yzoom = np.concatenate([all_yc_cat.reshape(-1), all_mppy_arr.reshape(-1)])

            xmargin_zoom = 0.12 * np.ptp(xzoom)
            ymargin_zoom = 0.12 * np.ptp(yzoom)

            if xmargin_zoom == 0:
                xmargin_zoom = 0.05 * np.max(np.abs(xzoom)) if np.max(np.abs(xzoom)) > 0 else 0.1
            if ymargin_zoom == 0:
                ymargin_zoom = 0.05 * np.max(np.abs(yzoom)) if np.max(np.abs(yzoom)) > 0 else 0.1
            if xmargin_zoom == 0:
                xmargin_zoom = 0.1
            if ymargin_zoom == 0:
                ymargin_zoom = 0.1

            ax2.set_xlim(np.min(xzoom) - xmargin_zoom, np.max(xzoom) + xmargin_zoom)
            ax2.set_ylim(np.min(yzoom) - ymargin_zoom, np.max(yzoom) + ymargin_zoom)

        ax2.set_xlabel("maxX")
        ax2.set_ylabel("maxSurge")
        ax2.set_title(f"Zoom contornos OR, cópula = {bestName}")
        ax2.grid(True)
        if legend_handles_zoom:
            ax2.legend(legend_handles_zoom, legend_entries_zoom, loc="lower right")

    out: dict[str, Any] = {
        "X": X.reshape(-1, 1),
        "Y": Y.reshape(-1, 1),
        "N": int(N),
        "nyears": float(nyears),
        "lambda": float(lambda_),
        "umbralX": fitX,
        "umbralY": fitY,
        "gpdX": {"u": uX, "k": kX, "sigma": sX, "pu": pX_u},
        "gpdY": {"u": uY, "k": kY, "sigma": sY, "pu": pY_u},
        "tablaCopulas": tablaCop,
        "bestCopula": bestName,
        "bestParam": bestPar,
        "Tret": Tret.reshape(-1, 1),
        "results": results,
        "fig_full": fig1,
        "fig_zoom": fig2,
    }

    if nT == 1 and results[0]["Tret"] is not None:
        out["p_or"] = results[0]["p_or"]
        out["clevel"] = results[0]["clevel"]
        out["contour"] = results[0]["contour"]
        out["mpp"] = results[0]["mpp"]

    return out
