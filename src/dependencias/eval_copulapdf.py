import numpy as np
from scipy.stats import multivariate_t, t as student_t
from statsmodels.distributions.copula.api import (
    GaussianCopula,
    GumbelCopula,
    ClaytonCopula,
    FrankCopula,
)


def eval_copulapdf(name, U, par):

    U = np.asarray(U, dtype=np.float64)

    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("U debe tener shape (n, 2).")

    eps = 1e-12
    U = np.clip(U, eps, 1.0 - eps)

    if name == "Gaussian":
        par_arr = np.asarray(par, dtype=np.float64)

        if par_arr.ndim == 0 or par_arr.size == 1:
            rho = float(par_arr.reshape(-1)[0])

        elif par_arr.shape == (2, 2):
            if not np.allclose(par_arr, par_arr.T, atol=1e-12):
                raise ValueError("Para la copula Gaussian, la matriz no es simétrica.")
            if not np.allclose(np.diag(par_arr), 1.0, atol=1e-12):
                raise ValueError("Para la copula Gaussian, la diagonal de la matriz debe ser 1.")
            rho = float(par_arr[0, 1])

        else:
            raise ValueError(
                f"Para la copula Gaussian, par debe ser escalar o matriz 2x2. "
                f"Recibido shape={par_arr.shape}, value={par}"
            )

        cop = GaussianCopula(corr=rho, k_dim=2)
        c = cop.pdf(U)

    elif name == "t":
        if not isinstance(par, dict):
            raise ValueError("Para la copula t, par debe ser un dict con 'Rho' y 'nu'.")

        rho = np.asarray(par["Rho"], dtype=np.float64)
        if rho.shape != (2, 2):
            raise ValueError("Para la copula t, par['Rho'] debe ser una matriz 2x2.")

        nu = float(par["nu"])
        q = student_t.ppf(U, df=nu)
        joint = multivariate_t.pdf(q, shape=rho, df=nu)
        marg0 = student_t.pdf(q[:, 0], df=nu)
        marg1 = student_t.pdf(q[:, 1], df=nu)
        c = joint / (marg0 * marg1)

    elif name == "Gumbel":
        theta = float(par)
        cop = GumbelCopula(theta=theta, k_dim=2)
        c = cop.pdf(U)

    elif name == "Clayton":
        theta = float(par)
        cop = ClaytonCopula(theta=theta, k_dim=2)
        c = cop.pdf(U)

    elif name == "Frank":
        theta = float(par)
        cop = FrankCopula(theta=theta, k_dim=2)
        c = cop.pdf(U)

    else:
        raise ValueError("Copula no reconocida.")

    c = np.asarray(c, dtype=np.float64).reshape(-1, 1)
    return np.maximum(c, np.finfo(np.float64).tiny)
