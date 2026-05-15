import numpy as np
import pandas as pd

from scipy.optimize import minimize, minimize_scalar
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student_t

_EPS_U = np.finfo(np.float64).eps

_REALMIN = np.finfo(np.float64).tiny

_RHO_MAX = 0.999999999
_NU_MIN = 2.001
_NU_MAX = 1e6
_BIG = 1e100


# ============================================================
# Utilities
# ============================================================

def _prepare_U(U):
    U = np.asarray(U, dtype=np.float64)

    if U.ndim != 2 or U.shape[1] != 2:
        raise ValueError("U debe tener shape (n, 2).")

    if np.any(~np.isfinite(U)):
        raise ValueError("U contiene NaN o Inf.")

    U = np.clip(U, _EPS_U, 1.0 - _EPS_U)
    return U


def _safe_sum_log(pdfv):
    pdfv = np.asarray(pdfv, dtype=np.float64).ravel()

    if pdfv.size == 0:
        return -np.inf
    if np.any(~np.isfinite(pdfv)):
        return -np.inf

    return float(np.sum(np.log(pdfv + _REALMIN)))


def _aic_bic(ll, k, n):
    aic = 2.0 * k - 2.0 * ll
    bic = k * np.log(n) - 2.0 * ll
    return float(aic), float(bic)


def _rho_to_R(rho):
    return np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)


def _kendall_to_rho(U):
    tau = pd.Series(U[:, 0]).corr(pd.Series(U[:, 1]), method="kendall")
    if not np.isfinite(tau):
        return 0.0
    return float(np.clip(np.sin(np.pi * tau / 2.0), -0.95, 0.95))


# ============================================================
# PDFs de cópulas
# ============================================================

def _pdf_gaussian(U, rho):
    if not (-_RHO_MAX < rho < _RHO_MAX):
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    z = norm.ppf(U)
    R = _rho_to_R(rho)

    try:
        joint = multivariate_normal.pdf(z, mean=np.zeros(2), cov=R)
        marg1 = norm.pdf(z[:, 0])
        marg2 = norm.pdf(z[:, 1])
        pdfv = joint / (marg1 * marg2)
    except Exception:
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    return np.asarray(pdfv, dtype=np.float64)


def _pdf_t(U, rho, nu):
    if not (-_RHO_MAX < rho < _RHO_MAX):
        return np.full(U.shape[0], np.nan, dtype=np.float64)
    if not (_NU_MIN < nu <= _NU_MAX):
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    x = student_t.ppf(U, df=nu)
    R = _rho_to_R(rho)

    try:
        joint = multivariate_t.pdf(x, shape=R, df=nu)
        marg1 = student_t.pdf(x[:, 0], df=nu)
        marg2 = student_t.pdf(x[:, 1], df=nu)
        pdfv = joint / (marg1 * marg2)
    except Exception:
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    pdfv = np.asarray(pdfv, dtype=np.float64)
    return np.maximum(pdfv, _REALMIN)


def _pdf_frank(U, theta):
    if abs(theta) < 1e-14:
        return np.ones(U.shape[0], dtype=np.float64)

    u = U[:, 0]
    v = U[:, 1]

    A = np.expm1(-theta)
    Au = np.expm1(-theta * u)
    Av = np.expm1(-theta * v)

    num = -theta * A * np.exp(-theta * (u + v))
    den = (A + Au * Av) ** 2

    return np.asarray(num / den, dtype=np.float64)


def _pdf_clayton(U, theta):
    if theta < 0.0:
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    if theta < 1e-14:
        return np.ones(U.shape[0], dtype=np.float64)

    u = U[:, 0]
    v = U[:, 1]

    s = u ** (-theta) + v ** (-theta) - 1.0
    pdfv = (
        (theta + 1.0)
        * u ** (-theta - 1.0)
        * v ** (-theta - 1.0)
        * s ** (-2.0 - 1.0 / theta)
    )
    return np.asarray(pdfv, dtype=np.float64)


def _pdf_gumbel(U, theta):
    if theta < 1.0:
        return np.full(U.shape[0], np.nan, dtype=np.float64)

    if abs(theta - 1.0) < 1e-14:
        return np.ones(U.shape[0], dtype=np.float64)

    u = U[:, 0]
    v = U[:, 1]

    lu = -np.log(u)
    lv = -np.log(v)

    x = lu ** theta
    y = lv ** theta
    A = x + y

    C = np.exp(-(A ** (1.0 / theta)))
    term = ((lu * lv) ** (theta - 1.0)) / (u * v)

    pdfv = C * term * (A ** (2.0 / theta - 2.0)) * (
        1.0 + (theta - 1.0) * (A ** (-1.0 / theta))
    )
    return np.asarray(pdfv, dtype=np.float64)


# ============================================================
# Log-likelihoods
# ============================================================

def _ll_gaussian(U, rho):
    pdfv = _pdf_gaussian(U, rho)
    return _safe_sum_log(pdfv)


def _ll_t(U, rho, nu):
    pdfv = _pdf_t(U, rho, nu)
    return _safe_sum_log(pdfv)


def _ll_frank(U, theta):
    pdfv = _pdf_frank(U, theta)
    return _safe_sum_log(pdfv)


def _ll_clayton(U, theta):
    pdfv = _pdf_clayton(U, theta)
    return _safe_sum_log(pdfv)


def _ll_gumbel(U, theta):
    pdfv = _pdf_gumbel(U, theta)
    return _safe_sum_log(pdfv)


# ============================================================
# Fits
# ============================================================

def _fit_gaussian(U):
    rho0 = _kendall_to_rho(U)

    def obj(x):
        rho = float(x[0])
        ll = _ll_gaussian(U, rho)
        return -ll if np.isfinite(ll) else _BIG

    res = minimize(
        obj,
        x0=np.array([rho0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(-_RHO_MAX, _RHO_MAX)],
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not res.success:
        raise RuntimeError(f"Fallo Gaussian: {res.message}")

    rho = float(res.x[0])
    pdfv = _pdf_gaussian(U, rho)
    ll = _safe_sum_log(pdfv)

    if not np.isfinite(ll):
        raise RuntimeError("Gaussian: log-likelihood no finita.")

    par = _rho_to_R(rho)
    return ll, 1, par


def _fit_t(U):
    # Ojo con esta, nu no acaba de salir bien; aunque no todas las copulas ajustan como en matlab
    rho_kendall = _kendall_to_rho(U)
    rho_seeds = [rho_kendall, 0.0]

    for nu_seed in (3.0, 5.0, 10.0, 30.0):
        try:
            z0 = student_t.ppf(U, df=nu_seed)
            rho_guess = float(np.corrcoef(z0[:, 0], z0[:, 1])[0, 1])
            if np.isfinite(rho_guess):
                rho_seeds.append(rho_guess)
        except Exception:
            continue

    clean_rho_seeds = []
    for rho0 in rho_seeds:
        rho0 = float(np.clip(rho0, -0.95, 0.95))
        if not any(abs(rho0 - r) < 1e-10 for r in clean_rho_seeds):
            clean_rho_seeds.append(rho0)

    def obj(theta):
        rho = float(np.tanh(theta[0]))
        nu = float(2.0 + np.exp(theta[1]))
        ll = _ll_t(U, rho, nu)
        return -ll if np.isfinite(ll) else _BIG

    best = None

    for rho0 in clean_rho_seeds:
        for nu0 in (3.0, 5.0, 10.0, 30.0):
            x0 = np.array([np.arctanh(rho0), np.log(nu0 - 2.0)], dtype=np.float64)

            for method in ("L-BFGS-B", "Powell"):
                try:
                    res = minimize(
                        obj,
                        x0=x0,
                        method=method,
                        bounds=[
                            (-8.0, 8.0),
                            (np.log(_NU_MIN - 2.0), np.log(_NU_MAX - 2.0)),
                        ],
                        options={"maxiter": 3000, "ftol": 1e-12},
                    )
                except Exception:
                    continue

                theta = np.asarray(res.x, dtype=np.float64)
                rho = float(np.tanh(theta[0]))
                rho = float(np.clip(rho, -_RHO_MAX, _RHO_MAX))
                nu = float(2.0 + np.exp(theta[1]))
                ll = _ll_t(U, rho, nu)

                if not np.isfinite(ll):
                    continue

                candidate = (ll, rho, nu)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best is None:
        raise RuntimeError("Fallo t: no se encontro ningun ajuste con log-likelihood finita.")

    ll, rho, nu = best
    par = {"Rho": _rho_to_R(rho), "nu": float(nu)}
    return float(ll), 2, par


def _fit_frank(U):
    candidates = []

    for bounds in [(-50.0, -1e-10), (1e-10, 50.0)]:
        res = minimize_scalar(
            lambda th: -_ll_frank(U, th) if np.isfinite(_ll_frank(U, th)) else _BIG,
            bounds=bounds,
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 2000},
        )
        if res.success and np.isfinite(res.fun):
            candidates.append(res)

    if not candidates:
        raise RuntimeError("Fallo Frank.")

    res = min(candidates, key=lambda r: r.fun)
    theta = float(res.x)
    pdfv = _pdf_frank(U, theta)
    ll = _safe_sum_log(pdfv)

    if not np.isfinite(ll):
        raise RuntimeError("Frank: log-likelihood no finita.")

    return ll, 1, theta


def _fit_clayton(U): #tambien diverge con respecto a matlab
    res = minimize_scalar(
        lambda th: -_ll_clayton(U, th) if np.isfinite(_ll_clayton(U, th)) else _BIG,
        bounds=(1e-10, 50.0),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 2000},
    )

    if not res.success:
        raise RuntimeError(f"Fallo Clayton: {res.message}")

    theta = float(res.x)
    pdfv = _pdf_clayton(U, theta)
    ll = _safe_sum_log(pdfv)

    if not np.isfinite(ll):
        raise RuntimeError("Clayton: log-likelihood no finita.")

    return ll, 1, theta


def _fit_gumbel(U):
    res = minimize_scalar(
        lambda th: -_ll_gumbel(U, th) if np.isfinite(_ll_gumbel(U, th)) else _BIG,
        bounds=(1.0 + 1e-10, 50.0),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 2000},
    )

    if not res.success:
        raise RuntimeError(f"Fallo Gumbel: {res.message}")

    theta = float(res.x)
    pdfv = _pdf_gumbel(U, theta)
    ll = _safe_sum_log(pdfv)

    if not np.isfinite(ll):
        raise RuntimeError("Gumbel: log-likelihood no finita.")

    return ll, 1, theta


# ============================================================
# Main
# ============================================================

def comparar_copulas(U, modelos, verbose=False):
    U = _prepare_U(U)
    n = U.shape[0]

    Modelo = []
    LogLik = []
    AIC = []
    BIC = []
    Param = []

    for name in modelos:
        try:
            if name == "Gaussian":
                ll, k, par = _fit_gaussian(U)

            elif name == "t":
                ll, k, par = _fit_t(U)

            elif name == "Gumbel":
                ll, k, par = _fit_gumbel(U)

            elif name == "Clayton":
                ll, k, par = _fit_clayton(U)

            elif name == "Frank":
                ll, k, par = _fit_frank(U)

            else:
                continue

            aic, bic = _aic_bic(ll, k, n)

            Modelo.append(name)
            LogLik.append(float(ll))
            AIC.append(float(aic))
            BIC.append(float(bic))
            Param.append(par)

            if verbose:
                print(f"{name}: ll={ll:.12f}, AIC={aic:.12f}, BIC={bic:.12f}")

        except Exception as e:
            if verbose:
                print(f"FALLO {name}: {e}")

    if len(Modelo) == 0:
        return pd.DataFrame(columns=["Modelo", "LogLik", "AIC", "BIC", "Param"])

    tab = pd.DataFrame(
        {
            "Modelo": Modelo,
            "LogLik": LogLik,
            "AIC": AIC,
            "BIC": BIC,
            "Param": Param,
        }
    )

    tab = tab.sort_values("AIC", kind="mergesort").reset_index(drop=True)
    return tab
