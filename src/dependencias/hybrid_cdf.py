from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def hybrid_cdf(x, sample, u, pu, k, sigma):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)

    # MATLAB aquí no filtra sample. Conviene dejar solo finitos para evitar contaminar mean(sample <= z) con NaN.
    sample = sample[np.isfinite(sample)]

    if sample.size == 0:
        raise ValueError("sample no contiene valores finitos.")

    if not np.isfinite(u):
        raise ValueError("u debe ser finito.")
    if not np.isfinite(pu):
        raise ValueError("pu debe ser finito.")
    if not np.isfinite(k):
        raise ValueError("k debe ser finito.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser finito y > 0.")

    F = np.zeros(x.shape, dtype=np.float64)

    body = x <= u
    if np.any(body):
        xb = x[body]

        F_body = np.mean(sample[None, :] <= xb[:, None], axis=1)
        F[body] = F_body.astype(np.float64)

    tail = x > u
    if np.any(tail):
        xt = x[tail] - u

        G = genpareto.cdf(xt, c=k, loc=0.0, scale=sigma)
        F[tail] = (1.0 - pu) + pu * G

    F = np.clip(F, 0.0, 1.0)

    return F.reshape(-1, 1)