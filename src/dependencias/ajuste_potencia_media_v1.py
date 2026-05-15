from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def _col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def ajuste_potencia_media_v1_to_mat_struct(modelo):
    out = {}

    for k, v in modelo.items():
        if k == "fun":
            out[k] = np.array([], dtype=np.float64)
            continue

        if k == "fig":
            continue

        if isinstance(v, dict):
            dict_out = {}
            for kk, vv in v.items():
                if isinstance(vv, str):
                    dict_out[kk] = vv
                elif vv is None:
                    dict_out[kk] = np.array([], dtype=np.float64)
                elif np.isscalar(vv):
                    dict_out[kk] = float(vv)
                else:
                    dict_out[kk] = np.asarray(vv, dtype=np.float64)
            out[k] = dict_out
        elif v is None:
            out[k] = np.array([], dtype=np.float64)
        elif isinstance(v, str):
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = np.asarray(v, dtype=np.float64)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v

    return out


def ajuste_potencia_media_v1(x, y, xnew=None, p0=None, plot=True, save_path=None):
    if x is None or y is None:
        raise ValueError("Debes proporcionar x e y.")

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    pos = (x >= 0) & (y >= 0)
    x = x[pos]
    y = y[pos]

    if xnew is None:
        xnew = np.array([], dtype=np.float64)
    else:
        xnew = np.asarray(xnew, dtype=np.float64).reshape(-1)

    if p0 is None:
        p0 = np.array([], dtype=np.float64)
    else:
        p0 = np.asarray(p0, dtype=np.float64).reshape(-1)

    x = x.reshape(-1)
    y = y.reshape(-1)

    if x.size != y.size:
        raise ValueError("x e y deben tener la misma longitud.")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No hay datos validos tras eliminar NaN/Inf.")

    if np.any(x <= 0):
        raise ValueError("Todos los valores de x deben ser mayores que 0 para ajustar y = a*x^b.")

    if xnew.size > 0 and (np.any(~np.isfinite(xnew)) or np.any(xnew <= 0)):
        raise ValueError("xnew debe contener valores finitos y mayores que 0.")

    if p0.size == 0:
        if np.all(y > 0):
            pp = np.polyfit(np.log(x), np.log(y), 1)
            p0 = np.array([np.exp(pp[1]), pp[0]], dtype=np.float64)
        else:
            a0 = np.mean(y)
            if not np.isfinite(a0) or a0 == 0:
                a0 = 1.0
            b0 = 1.0
            p0 = np.array([a0, b0], dtype=np.float64)
    else:
        if p0.size != 2:
            raise ValueError("p0 debe ser un vector de dos elementos: [a0 b0].")

    def fun_model(p, xx):
        xx = np.asarray(xx, dtype=np.float64)
        return p[0] * np.power(xx, p[1])

    def residuals(p):
        return fun_model(p, x) - y

    result = least_squares(
        residuals,
        x0=p0,
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000,
    )

    p_opt = np.asarray(result.x, dtype=np.float64)

    a = float(p_opt[0])
    b = float(p_opt[1])

    def fun_final(xx):
        xx = np.asarray(xx, dtype=np.float64)
        return a * np.power(xx, b)

    yfit = fun_final(x)
    ypred = fun_final(xnew) if xnew.size > 0 else np.array([], dtype=np.float64)

    ssres = np.sum((y - yfit) ** 2)
    sstot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ssres / sstot) if sstot > 0 else np.nan

    xall = np.concatenate([x, xnew]) if xnew.size > 0 else x
    xmin = np.min(xall)
    xmax = np.max(xall)
    xx = np.full(300, xmin, dtype=np.float64) if xmin == xmax else np.linspace(xmin, xmax, 300, dtype=np.float64)
    yy = fun_final(xx)

    fig = None
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, "k.", markersize=4, label="Datos")
        ax.plot(xx, yy, "b-", linewidth=1.8, label="Ajuste")

        if xnew.size > 0:
            for i in range(xnew.size):
                ax.plot(
                    xnew[i],
                    ypred[i],
                    "o",
                    color="r",
                    markerfacecolor="r",
                    markeredgecolor="r",
                    markersize=4,
                )

        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Y = {a:.4g}·X^{{{b:.4g}}}   |   R^2 = {r2:.4f}")
        ax.legend(loc="best")

        if save_path is not None:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")

    status_to_exitflag = {
        0: 0.0,
        1: 1.0,
        2: 3.0,
        3: 3.0,
        4: 3.0,
    }
    exitflag = status_to_exitflag.get(result.status, float(result.status))

    output = {
        "firstorderopt": float(result.optimality),
        "iterations": float(result.nfev - 1) if result.nfev is not None else np.nan,
        "funcCount": float(result.nfev) if result.nfev is not None else np.nan,
        "cgiterations": 0.0,
        "algorithm": "trust-region-reflective",
        "stepsize": float(np.linalg.norm(result.jac.T @ result.fun, ord=np.inf)) if result.jac is not None else np.nan,
        "message": result.message,
    }

    return {
        "p": _col(p_opt),
        "a": a,
        "b": b,
        "fun": fun_final,
        "yfit": _col(yfit),
        "x": _col(x),
        "y": _col(y),
        "xnew": _col(xnew),
        "ypred": _col(ypred),
        "resid": _col(y - yfit),
        "R2": r2,
        "resnorm": float(ssres),
        "exitflag": exitflag,
        "output": output,
        "fig": fig,
    }
