import numpy as np


def make_strictly_increasing(z):


    z = np.asarray(z, dtype=np.float64).reshape(-1)

    if z.size == 0:
        return z.reshape(1, -1)

    if np.any(~np.isfinite(z)):
        raise ValueError("La malla contiene NaN o Inf.")

    z = np.maximum.accumulate(z)

    span = np.max(z) - np.min(z)
    if span == 0:
        span = 1.0

    eps0 = 1e-10 * span

    for i in range(1, z.size):
        if z[i] <= z[i - 1]:
            z[i] = z[i - 1] + eps0

    return z.reshape(1, -1)