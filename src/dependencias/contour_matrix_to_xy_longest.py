import numpy as np


def contour_matrix_to_xy_longest(M):

    x = np.array([], dtype=np.float64).reshape(-1, 1)
    y = np.array([], dtype=np.float64).reshape(-1, 1)

    M = np.asarray(M, dtype=np.float64)

    if M.size == 0:
        return x, y

    if M.ndim != 2 or M.shape[0] != 2:
        raise ValueError("M debe ser una matriz 2xN estilo contourc de MATLAB.")

    k = 0
    best_len = 0
    bestx = np.array([], dtype=np.float64).reshape(-1, 1)
    besty = np.array([], dtype=np.float64).reshape(-1, 1)

    # MATLAB: while k < size(M,2) con k empezando en 1
    # Python: columnas 0..N-1
    while k < M.shape[1]:
        n = int(M[1, k])
        cols = slice(k + 1, k + 1 + n)

        xx = M[0, cols].reshape(-1, 1)
        yy = M[1, cols].reshape(-1, 1)

        if xx.size > best_len:
            best_len = xx.size
            bestx = xx
            besty = yy

        k = k + n + 1

    x = bestx
    y = besty

    return x, y