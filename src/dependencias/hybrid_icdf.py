import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import genpareto


def hybrid_icdf(p, sample, u, pu, k, sigma):
    p_in = np.asarray(p, dtype=np.float64)
    scalar_input = (p_in.ndim == 0)

    p_flat = p_in.reshape(-1)
    p_flat = np.clip(p_flat, 0.0, 1.0)

    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]

    if sample.size == 0:
        raise ValueError("sample no contiene valores finitos.")
    if not np.isfinite(u):
        raise ValueError("u debe ser finito.")
    if not np.isfinite(pu) or pu < 0 or pu > 1:
        raise ValueError("pu debe estar en [0,1].")
    if not np.isfinite(k):
        raise ValueError("k debe ser finito.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma debe ser > 0 y finito.")

    xs = np.sort(sample)
    n = xs.size
    pbody = 1.0 - pu

    x_flat = np.empty_like(p_flat, dtype=np.float64)

    body = p_flat <= pbody
    if np.any(body):
        pp = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n

        f_interp = interp1d(
            pp,
            xs,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        x_flat[body] = np.asarray(f_interp(p_flat[body]), dtype=np.float64)

    tail = ~body
    if np.any(tail):
        if pu == 0:
            x_flat[tail] = u
        else:
            q = (p_flat[tail] - pbody) / pu
            q = np.clip(q, 0.0, 1.0)

            xt = genpareto.ppf(q, c=k, loc=0.0, scale=sigma)
            x_flat[tail] = u + np.asarray(xt, dtype=np.float64)

    x_out = x_flat.reshape(p_in.shape)

    if scalar_input:
        return float(x_out)

    return x_out