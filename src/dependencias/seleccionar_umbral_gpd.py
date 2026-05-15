import numpy as np
from scipy.optimize import minimize


def matlab_prctile(x, q):

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x = np.sort(x)

    if x.size == 0:
        raise ValueError("x está vacío tras filtrar no finitos.")

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


def gpd_negloglik(params, exc):

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


def gpfit_like_matlab(exc):

    exc = np.asarray(exc, dtype=np.float64).reshape(-1)
    exc = exc[np.isfinite(exc)]

    if exc.size == 0:
        raise ValueError("No hay excesos válidos.")


    x0 = np.array([0.1, max(np.mean(exc), 1e-8)], dtype=np.float64)

    res = minimize(
        gpd_negloglik,
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
        raise RuntimeError(f"Falló la optimización GPD: {res.message}")

    k, sigma = res.x
    return float(k), float(sigma)


def seleccionar_umbral_gpd(z, qgrid):

    best_score = np.inf
    best = None

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]

    qgrid = np.asarray(qgrid, dtype=np.float64).reshape(-1)

    for q in qgrid:
        try:
            u = matlab_prctile(z, q)
            exc = z[z > u] - u

            if exc.size < 12:
                continue

            k, sigma = gpfit_like_matlab(exc)
            score = abs(k) + 5.0 / exc.size

            if score < best_score:
                best_score = score
                best = {
                    "u": float(u),
                    "q": float(q),
                    "k": float(k),
                    "sigma": float(sigma),
                    "nexc": int(exc.size),
                }

        except Exception:
            pass

    if best is None:
        raise ValueError("No se pudo seleccionar umbral para la GPD.")

    return best