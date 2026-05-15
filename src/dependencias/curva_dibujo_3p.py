from __future__ import annotations

import numpy as np


def curva_dibujo_3p(x, y, xx, m1, m3):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xx = np.asarray(xx, dtype=np.float64)

    if x.size != 3 or y.size != 3:
        raise ValueError("x e y deben tener 3 elementos.")

    x1, x2, x3 = x
    y1, y2, y3 = y

    if not (x1 < x2 < x3):
        raise ValueError("Debe cumplirse x1 < x2 < x3.")

    m2 = 0.0

    h1 = x2 - x1
    a1_mat = np.array([[h1 ** 3, h1 ** 2], [3.0 * h1 ** 2, 2.0 * h1]], dtype=np.float64)
    b1_vec = np.array([y2 - (m1 * h1 + y1), m2 - m1], dtype=np.float64)
    u1 = np.linalg.solve(a1_mat, b1_vec)
    a1 = u1[0]
    b1c = u1[1]
    c1 = float(m1)
    d1 = y1

    h2 = x3 - x2
    a2_mat = np.array([[h2 ** 3, h2 ** 2], [3.0 * h2 ** 2, 2.0 * h2]], dtype=np.float64)
    b2_vec = np.array([y3 - (m2 * h2 + y2), m3 - m2], dtype=np.float64)
    u2 = np.linalg.solve(a2_mat, b2_vec)
    a2 = u2[0]
    b2c = u2[1]
    c2 = m2
    d2 = y2

    yy = np.empty_like(xx, dtype=np.float64)

    mask1 = xx <= x2
    t1 = xx[mask1] - x1
    yy[mask1] = a1 * t1 ** 3 + b1c * t1 ** 2 + c1 * t1 + d1

    mask2 = ~mask1
    t2 = xx[mask2] - x2
    yy[mask2] = a2 * t2 ** 3 + b2c * t2 ** 2 + c2 * t2 + d2

    return yy
