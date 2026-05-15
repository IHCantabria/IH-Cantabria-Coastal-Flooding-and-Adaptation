import numpy as np

def _row_vector(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(1, -1)


def pot_extremos_to_mat_struct(eventos):


    dtype = [
        ("ini", "O"),
        ("fin", "O"),
        ("idx", "O"),
        ("t", "O"),
        ("X", "O"),
        ("surge", "O"),
        ("t_exc_X", "O"),
        ("t_exc_surge", "O"),
        ("exc_X", "O"),
        ("exc_surge", "O"),
        ("maxX", "O"),
        ("idx_maxX", "O"),
        ("maxSurge", "O"),
        ("Tm02", "O"),
    ]

    out = np.empty((1, len(eventos)), dtype=dtype)

    for i, ev in enumerate(eventos):
        out[0, i] = (
            float(ev["ini"]),
            float(ev["fin"]),
            _row_vector(ev["idx"]),
            _row_vector(ev["t"]),
            _row_vector(ev["X"]),
            _row_vector(ev["surge"]),
            _row_vector(ev["t_exc_X"]),
            _row_vector(ev["t_exc_surge"]),
            _row_vector(ev["exc_X"]),
            _row_vector(ev["exc_surge"]),
            float(ev["maxX"]),
            float(ev["idx_maxX"]),
            float(ev["maxSurge"]),
            np.array([], dtype=np.float64).reshape(0, 0),
        )

    return out

def pot_extremos(t, X, surge, umbral1, umbral2, minsep):

    t = np.asarray(t, dtype=np.float64).flatten()
    X = np.asarray(X, dtype=np.float64).flatten()
    surge = np.asarray(surge, dtype=np.float64).flatten()

    if not (len(t) == len(X) == len(surge)):
        raise ValueError("t, X y surge deben tener misma longitud")


    exc_X = X > umbral1
    exc_surge = surge > umbral2
    exc_all = exc_X | exc_surge

    idx_ext = np.where(exc_all)[0]

    if len(idx_ext) == 0:
        return []


    dt = np.diff(t[idx_ext])

    nuevo_evento = np.concatenate([[True], dt > minsep])
    id_evento = np.cumsum(nuevo_evento)

    eventos = []

    for k in range(1, id_evento[-1] + 1):
        ii = idx_ext[id_evento == k]

        t_ev = t[ii].astype(np.float64)
        X_ev = X[ii].astype(np.float64)
        surge_ev = surge[ii].astype(np.float64)

        excX_ev = exc_X[ii]
        excSurge_ev = exc_surge[ii]

        # máximos
        idx_local_maxX = np.argmax(X_ev)
        idx_global_maxX = float(ii[idx_local_maxX] + 1)

        maxX = X_ev[idx_local_maxX]
        maxSurge = np.max(surge_ev)

        evento = {
            "ini": float(t_ev[0]),
            "fin": float(t_ev[-1]),
            "idx": (ii + 1).astype(np.float64),
            "t": t_ev,
            "X": X_ev,
            "surge": surge_ev,
            "t_exc_X": t_ev[excX_ev].astype(np.float64),
            "t_exc_surge": t_ev[excSurge_ev].astype(np.float64),
            "exc_X": X_ev[excX_ev].astype(np.float64),
            "exc_surge": surge_ev[excSurge_ev].astype(np.float64),
            "maxX": float(maxX),
            "idx_maxX": float(idx_global_maxX),
            "maxSurge": float(maxSurge),
            "Tm02": np.array([], dtype=np.float64),
        }

        eventos.append(evento)

    return eventos