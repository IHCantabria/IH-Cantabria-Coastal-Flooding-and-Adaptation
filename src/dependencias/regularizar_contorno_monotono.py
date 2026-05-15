import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


def regularizar_contorno_monotono(x, y, npts):

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("x e y deben tener la misma longitud.")

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    isort = np.argsort(x)
    x = x[isort]
    y = y[isort]

    xu, inv = np.unique(x, return_inverse=True)
    yu = np.zeros_like(xu, dtype=np.float64)
    counts = np.zeros_like(xu, dtype=np.int64)

    np.add.at(yu, inv, y)
    np.add.at(counts, inv, 1)
    yu = yu / counts

    if xu.size < 5:
        xout = xu.reshape(-1, 1)
        yout = yu.reshape(-1, 1)
        return xout, yout

    xg = np.linspace(np.min(xu), np.max(xu), int(npts), dtype=np.float64)

    pchip = PchipInterpolator(xu, yu)
    yg = np.asarray(pchip(xg), dtype=np.float64)

    yg_mono = yg.copy()
    for i in range(1, yg_mono.size):
        if yg_mono[i] > yg_mono[i - 1]:
            yg_mono[i] = yg_mono[i - 1]

    if yg_mono.size >= 15:
        from scipy.io import savemat
        #savemat("debug_regularizar_preSmooth_py.mat", {"yg_mono_py": yg_mono.reshape(-1, 1)})
        yg_s = savgol_filter(yg_mono, window_length=15, polyorder=2, mode="interp")
        yg_s = np.asarray(yg_s, dtype=np.float64)
    else:
        yg_s = yg_mono.copy()

    for i in range(1, yg_s.size):
        if yg_s[i] > yg_s[i - 1]:
            yg_s[i] = yg_s[i - 1]

    xout = xg.reshape(-1, 1)
    yout = yg_s.reshape(-1, 1)

    return xout, yout