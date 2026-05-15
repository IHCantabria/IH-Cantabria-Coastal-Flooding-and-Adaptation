import numpy as np
from scipy.stats import genpareto


def _gaussian_kernel_pdf_1d(x_eval, sample, bandwidth):

    x_eval = np.asarray(x_eval, dtype=np.float64).reshape(-1)
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)

    z = (x_eval[:, None] - sample[None, :]) / bandwidth
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return np.mean(phi, axis=1) / bandwidth


def _matlab_like_bandwidth(sample):

    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    n = sample.size

    if n < 2:
        raise ValueError("Se necesitan al menos 2 datos para KDE.")

    std = np.std(sample, ddof=1)
    iqr = np.subtract(*np.percentile(sample, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std

    h = 0.9 * sigma * n ** (-1.0 / 5.0)

    if not np.isfinite(h) or h <= 0:
        h = max(std, 1e-6) * n ** (-1.0 / 5.0)

    return float(max(h, 1e-12))


def hybrid_pdf(x, sample, u, pu, k, sigma, bandwidth=None):

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]

    if sample.size == 0:
        raise ValueError("sample no contiene valores finitos.")
    if not np.isfinite(u):
        raise ValueError("u debe ser finito.")
    if not np.isfinite(pu) or not (0.0 <= pu <= 1.0):
        raise ValueError("pu debe estar en [0,1].")
    if not np.isfinite(k):
        raise ValueError("k debe ser finito.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser > 0 y finito.")

    f = np.zeros_like(x, dtype=np.float64)

    body = x <= u
    if np.any(body):
        xb = x[body]

        try:
            h = _matlab_like_bandwidth(sample) if bandwidth is None else float(bandwidth)
            fb = _gaussian_kernel_pdf_1d(xb, sample, h)

            pbody_emp = np.mean(sample <= u)
            if pbody_emp > 0:
                fb = fb * ((1.0 - pu) / pbody_emp)
            else:
                fb = np.full_like(xb, 1e-12, dtype=np.float64)

            f[body] = fb

        except Exception:
            f[body] = 1e-12

    tail = x > u
    if np.any(tail):
        xt = x[tail] - u
        gt = genpareto.pdf(xt, c=k, loc=0.0, scale=sigma)
        f[tail] = pu * np.asarray(gt, dtype=np.float64)

    f = np.maximum(f, 1e-12)

    return f.reshape(-1, 1)