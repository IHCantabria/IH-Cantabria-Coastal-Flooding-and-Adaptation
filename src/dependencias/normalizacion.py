from __future__ import annotations

import numpy as np


def normalizacion(x):
    x = np.asarray(x, dtype=np.float64)
    mu_storm = np.mean(x)
    std_storm = np.std(x, ddof=1)
    if not np.isfinite(std_storm) or std_storm == 0:
        xnorm = np.full_like(x, np.nan, dtype=np.float64)
    else:
        xnorm = (x - mu_storm) / std_storm
    return xnorm, mu_storm, std_storm
