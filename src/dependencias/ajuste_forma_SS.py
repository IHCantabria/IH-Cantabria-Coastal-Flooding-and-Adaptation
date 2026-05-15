from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def ajuste_forma_ss_to_mat_struct(modelo):
    out = {}

    for k, v in modelo.items():
        if k == "fun":
            out[k] = np.array([], dtype=np.float64)
            continue

        if k == "fig":
            continue

        if isinstance(v, dict):
            out[k] = {
                kk: float(vv) if np.isscalar(vv) else np.asarray(vv, dtype=np.float64)
                for kk, vv in v.items()
            }
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


def ajuste_forma_ss(Xdata, Ydata, Xnew=None, plot=True, save_path=None):
    if Xdata is None or Ydata is None:
        raise ValueError("Debes proporcionar Xdata e Ydata.")

    if Xnew is None:
        Xnew = np.array([], dtype=np.float64)

    Xdata = np.asarray(Xdata, dtype=np.float64).reshape(-1)
    Ydata = np.asarray(Ydata, dtype=np.float64).reshape(-1)

    if Xdata.size != Ydata.size:
        raise ValueError("Xdata e Ydata deben tener la misma longitud.")

    mask = np.isfinite(Xdata) & np.isfinite(Ydata)
    Xdata = Xdata[mask]
    Ydata = Ydata[mask]

    if Xdata.size == 0:
        raise ValueError("No hay datos validos tras eliminar NaN/Inf.")

    Xnew = np.asarray(Xnew, dtype=np.float64).reshape(-1)
    if Xnew.size > 0 and np.any(~np.isfinite(Xnew)):
        raise ValueError("Xnew debe contener solo valores finitos.")

    usar_potencial = np.all(Xdata > 0) and np.all(Ydata > 0)

    if usar_potencial:
        xT = np.log(Xdata)
        yT = np.log(Ydata)

        if np.unique(xT).size < 2:
            raise ValueError("No hay variabilidad suficiente en Xdata para ajustar el modelo.")

        p = np.polyfit(xT, yT, 1)
        b = float(p[0])
        intercepto = float(p[1])
        a = float(np.exp(intercepto))

        def fun(x):
            x = np.asarray(x, dtype=np.float64)
            return a * np.power(x, b)

        Yfit = fun(Xdata)
        yT_fit = np.polyval(p, xT)
        SSres = np.sum((yT - yT_fit) ** 2)
        SStot = np.sum((yT - np.mean(yT)) ** 2)
        R2 = float(1 - SSres / SStot) if SStot > 0 else np.nan

        if Xnew.size > 0:
            if np.any(Xnew <= 0):
                raise ValueError("En modo potencial, Xnew debe contener valores mayores que 0.")
            Ypred = fun(Xnew)
        else:
            Ypred = np.array([], dtype=np.float64)

        tipo = "potencial"
        parametros = {"a": a, "b": b}
        ylabel_txt = "Y"
        ecuacion_txt = f"Y = {a:.4g}·X^{{{b:.4g}}}"
        titulo_txt = f"{ecuacion_txt}   |   R^2 = {R2:.4f}"
    else:
        xT = np.arcsinh(Xdata)
        yT = np.arcsinh(Ydata)

        if np.unique(xT).size < 2:
            raise ValueError("No hay variabilidad suficiente en Xdata para ajustar el modelo.")

        p = np.polyfit(xT, yT, 1)
        m = float(p[0])
        c = float(p[1])

        def fun(x):
            x = np.asarray(x, dtype=np.float64)
            return np.sinh(np.polyval(p, np.arcsinh(x)))

        Yfit = fun(Xdata)
        yT_fit = np.polyval(p, xT)
        SSres = np.sum((yT - yT_fit) ** 2)
        SStot = np.sum((yT - np.mean(yT)) ** 2)
        R2 = float(1 - SSres / SStot) if SStot > 0 else np.nan
        Ypred = fun(Xnew) if Xnew.size > 0 else np.array([], dtype=np.float64)

        tipo = "asinh"
        parametros = {"m": m, "c": c}
        ylabel_txt = "Y"
        ecuacion_txt = f"asinh(Y) = {m:.4g}·asinh(X) + {c:.4g}"
        titulo_txt = f"{ecuacion_txt}   |   R^2 = {R2:.4f}"

    Xall = np.concatenate([Xdata, Xnew]) if Xnew.size > 0 else Xdata
    xmin = np.min(Xall)
    xmax = np.max(Xall)
    xx = np.full(300, xmin, dtype=np.float64) if xmin == xmax else np.linspace(xmin, xmax, 300, dtype=np.float64)
    yy = fun(xx)

    fig = None
    if plot:
        fig, ax = plt.subplots()
        ax.plot(Xdata, Ydata, "k.", markersize=4, label="Datos")
        ax.plot(xx, yy, "b-", linewidth=1.8, label="Ajuste")

        if Xnew.size > 0:
            ax.plot(
                Xnew,
                Ypred,
                "ro",
                markerfacecolor="r",
                markersize=4,
                label="Prediccion",
            )

        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel(ylabel_txt)
        ax.set_title(titulo_txt)
        ax.legend(loc="best")

        if save_path is not None:
            fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return {
        "tipo": tipo,
        "parametros": parametros,
        "Yfit": _col(Yfit),
        "R2": float(R2),
        "Xdata": _col(Xdata),
        "Ydata": _col(Ydata),
        "Xnew": _col(Xnew),
        "Ypred": _col(Ypred),
        "fun": fun,
        "fig": fig,
    }
