import numpy as np
from scipy.stats import t as student_t
import matplotlib.pyplot as plt


import numpy as np
from scipy.stats import t as student_t


def _col(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 1)


def ajuste_potencial_v2_to_mat_struct(modelo):
    """
    Guardado en .mat compatible excluyendo 'fun'
    """
    out = {}

    for k, v in modelo.items():
        if k == "fun":
            continue

        if v is None:
            out[k] = np.array([], dtype=np.float64)
        elif isinstance(v, np.ndarray):
            out[k] = np.asarray(v, dtype=np.float64)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v

    return out

def ajuste_potencial_v2(Xdata, Ydata, Xnew=None, alpha=0.05, plot=True, save_path=None):

    if Xdata is None or Ydata is None:
        raise ValueError("Debes proporcionar Xdata e Ydata.")

    if alpha is None:
        alpha = 0.05

    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0 or alpha >= 1:
        raise ValueError("alpha debe ser un escalar entre 0 y 1.")

    Xdata = np.asarray(Xdata, dtype=np.float64).reshape(-1)
    Ydata = np.asarray(Ydata, dtype=np.float64).reshape(-1)

    if len(Xdata) != len(Ydata):
        raise ValueError("Xdata e Ydata deben tener la misma longitud.")

    mask = np.isfinite(Xdata) & np.isfinite(Ydata)
    Xdata = Xdata[mask]
    Ydata = Ydata[mask]

    if np.any(Xdata <= 0) or np.any(Ydata <= 0):
        raise ValueError("Xdata e Ydata deben ser mayores que 0 para ajustar Y = a*x^b.")

    if Xnew is None:
        Xnew = np.array([], dtype=np.float64)
    else:
        Xnew = np.asarray(Xnew, dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(Xnew)) or np.any(Xnew <= 0):
            raise ValueError("Xnew debe contener valores finitos y mayores que 0.")

    xlog = np.log(Xdata)
    ylog = np.log(Ydata)

    p = np.polyfit(xlog, ylog, 1)
    b = float(p[0])
    alpha0 = float(p[1])
    a = float(np.exp(alpha0))

    fun = lambda x: a * np.power(x, b)

    Yfit = fun(Xdata).astype(np.float64)

    ylog_fit = np.polyval(p, xlog)
    SSres = np.sum((ylog - ylog_fit) ** 2)
    SStot = np.sum((ylog - np.mean(ylog)) ** 2)
    R2log = float(1 - SSres / SStot)

    n = len(Xdata)
    gl = n - 2

    if gl <= 0:
        raise ValueError("Se necesitan al menos 3 puntos para calcular el intervalo de predicción.")

    xbar = np.mean(xlog)
    Sxx = np.sum((xlog - xbar) ** 2)

    if Sxx <= 0:
        raise ValueError("No hay variabilidad suficiente en Xdata para ajustar el modelo.")

    s = np.sqrt(SSres / gl)
    tval = student_t.ppf(1 - alpha / 2, gl)

    if Xnew.size > 0:
        xnew_log = np.log(Xnew)
        ynew_log_hat = np.polyval(p, xnew_log)

        se_pred_new = s * np.sqrt(1 + 1 / n + (xnew_log - xbar) ** 2 / Sxx)

        ynew_log_low = ynew_log_hat - tval * se_pred_new
        ynew_log_high = ynew_log_hat + tval * se_pred_new

        Ypred = np.exp(ynew_log_hat).astype(np.float64)
        Ypred_low = np.exp(ynew_log_low).astype(np.float64)
        Ypred_high = np.exp(ynew_log_high).astype(np.float64)
    else:
        Ypred = np.array([], dtype=np.float64)
        Ypred_low = np.array([], dtype=np.float64)
        Ypred_high = np.array([], dtype=np.float64)

    if Xnew.size > 0:
        Xall = np.concatenate([Xdata, Xnew])
    else:
        Xall = Xdata

    Xplot_min = np.min(Xall)
    Xplot_max = np.max(Xall)

    xx = np.linspace(Xplot_min, Xplot_max, 300).astype(np.float64)
    xx_log = np.log(xx)

    yy_log = np.polyval(p, xx_log)
    yy = np.exp(yy_log).astype(np.float64)

    se_pred_xx = s * np.sqrt(1 + 1 / n + (xx_log - xbar) ** 2 / Sxx)

    yy_log_low = yy_log - tval * se_pred_xx
    yy_log_high = yy_log + tval * se_pred_xx

    yy_low = np.exp(yy_log_low).astype(np.float64)
    yy_high = np.exp(yy_log_high).astype(np.float64)

    modelo = {
        "a": float(a),
        "b": float(b),
        "Yfit": _col(Yfit),
        "R2log": float(R2log),
        "Xdata": _col(Xdata),
        "Ydata": _col(Ydata),
        "Xnew": _col(Xnew),
        "Ypred": _col(Ypred),
        "Ypred_low": _col(Ypred_low),
        "Ypred_high": _col(Ypred_high),
        "PI_xx": np.column_stack([yy_low, yy_high]).astype(np.float64),
        "PI_new": np.column_stack([Ypred_low, Ypred_high]).astype(np.float64),
        "alpha": float(alpha),
        "confLevel": float(1 - alpha),
        "fun": fun, #
    }

    if plot:
        fig, ax = plt.subplots(num=3, clear=True)

        # Scatter datos
        ax.plot(Xdata, Ydata, 'k.', markersize=4, label='Tp|maxHs', zorder=1) #cambio markersize a 4 de 9, y pongo zorder, para que se parezca al output de matlab

        # Banda de predicción
        ax.fill_between(
            xx.flatten(),
            yy_low.flatten(),
            yy_high.flatten(),
            color=(0.85, 0.85, 0.85),
            alpha=0.6,
            label=f'IC {100 * (1 - alpha):.1f}%',
            zorder=2 #pongo zorder para que se ponga lo gris por delante de los puntos negros, la banda va en medio
        )

        ax.plot(xx, yy, 'b-', linewidth=1.8, zorder=3)#zorder para que gris por delante de puntos


        if Xnew is not None and len(Xnew) > 0:
            cmap = plt.get_cmap('tab10')

            for i in range(len(Xnew)):
                ax.errorbar(
                    Xnew[i], Ypred[i],
                    yerr=[[Ypred[i] - Ypred_low[i]],
                          [Ypred_high[i] - Ypred[i]]],
                    fmt='o',
                    color=cmap(i),
                    markerfacecolor=cmap(i),
                    markeredgecolor=cmap(i),
                    linewidth=1.2,
                    capsize=0,
                    zorder=4 #zorder para estetica similar a matlab
                )

        ax.grid(True)
        ax.set_xlabel('Hs (m)')
        ax.set_ylabel('Tp (s)')
        ax.set_title(f'Y = {a:.4g}·X^{b:.4g}   |   R²_log = {R2log:.4f}')
        ax.legend(loc='lower right')


        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')


    modelo["fig"] = fig if plot else None
    return modelo