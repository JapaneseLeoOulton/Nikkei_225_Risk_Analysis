"""Models. All inputs are daily log returns in percent.

- OLS AR(p) walk-forward point forecasts (scikit-learn LinearRegression)
- Gaussian: i.i.d. normal, MLE mean and sd
- Merton jump-diffusion: i.i.d. Poisson mixture of normals, own MLE (scipy Nelder-Mead)
- GARCH(1,1) family: own MLE for GARCH-N (cross-checked against `arch`),
  `arch` for GARCH-t, GJR-N and GJR-t; own variance filter for forecasting
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# OLS AR(p) walk-forward (Task 2)
# ---------------------------------------------------------------------------
def lag_table(r, max_lag=5):
    return pd.DataFrame({"y": r, **{f"lag{k}": r.shift(k) for k in range(1, max_lag + 1)}}).dropna()


def ols_walk_forward(table, p, test_dates):
    """For each test day t, refit OLS of y on lags 1..p using rows strictly before t."""
    cols = [f"lag{k}" for k in range(1, p + 1)]
    X, y = table[cols].to_numpy(), table["y"].to_numpy()
    pos = table.index.get_indexer(test_dates)
    preds = np.empty(len(pos))
    for j, i in enumerate(pos):
        model = LinearRegression().fit(X[:i], y[:i])
        preds[j] = model.predict(X[i:i + 1])[0]
    return pd.Series(preds, index=test_dates)


def expanding_mean_forecast(table, test_dates):
    """Naive benchmark: mean of y over rows strictly before t."""
    csum = table["y"].cumsum().shift(1)
    n = pd.Series(np.arange(len(table)), index=table.index)
    return (csum / n).loc[test_dates]


# ---------------------------------------------------------------------------
# Gaussian (Tasks 3-4)
# ---------------------------------------------------------------------------
def fit_gaussian(r):
    """MLE: sample mean and sd with ddof=0."""
    return {"mu": float(np.mean(r)), "sigma": float(np.std(r))}


def gaussian_var(params, a):
    """One-day VaR at tail probability a, as a positive loss in %."""
    return -(params["mu"] + stats.norm.ppf(a) * params["sigma"])


# ---------------------------------------------------------------------------
# Merton jump-diffusion (Tasks 4-5)
#   r = mu + sigma*Z + sum_{j=1..N} J_j,  N ~ Poisson(lam),  J ~ N(m, delta^2)
#   density = sum_k P(N=k) * N(r; mu + k m, sigma^2 + k delta^2), truncated at K
# ---------------------------------------------------------------------------
MERTON_K = 10


def _merton_components(lam, K=MERTON_K):
    k = np.arange(K + 1)
    return k, -lam + k * np.log(lam) - gammaln(k + 1)


def merton_logpdf(r, mu, sigma, lam, m, delta, K=MERTON_K):
    k, log_w = _merton_components(lam, K)
    sd = np.sqrt(sigma**2 + k * delta**2)
    return logsumexp(log_w + stats.norm.logpdf(np.asarray(r)[:, None], mu + k * m, sd), axis=1)


def merton_cdf(x, mu, sigma, lam, m, delta, K=MERTON_K):
    k, log_w = _merton_components(lam, K)
    sd = np.sqrt(sigma**2 + k * delta**2)
    return float(np.sum(np.exp(log_w) * stats.norm.cdf(x, mu + k * m, sd)))


def _merton_unpack(theta):
    mu, log_sigma, log_lam, m, log_delta = theta
    return dict(mu=mu, sigma=np.exp(log_sigma), lam=np.exp(log_lam), m=m, delta=np.exp(log_delta))


def fit_merton(r):
    """MLE from three starting points (log-parameterised so sigma, lam, delta > 0); keeps the best."""
    mu0 = np.mean(r)
    starts = [
        [mu0, np.log(1.2), np.log(0.10), -0.5, np.log(3.0)],
        [mu0, np.log(1.0), np.log(0.05), -1.0, np.log(4.0)],
        [mu0, np.log(1.4), np.log(0.02), -1.0, np.log(5.0)],
    ]
    nll = lambda th: -np.sum(merton_logpdf(r, **_merton_unpack(th)))
    fits = [minimize(nll, x0, method="Nelder-Mead",
                     options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8}) for x0 in starts]
    best = min(fits, key=lambda f: f.fun)
    params = {k: float(v) for k, v in _merton_unpack(best.x).items()}
    return params, -best.fun, [-f.fun for f in fits]


def merton_var(params, a):
    return -brentq(lambda x: merton_cdf(x, **params) - a, -30, 0)


# ---------------------------------------------------------------------------
# GARCH family (Task 6)
#   eps_t = r_t - mu (mu = training sample mean, held fixed)
#   sigma_t^2 = omega + (alpha + gamma*1[eps_{t-1}<0]) eps_{t-1}^2 + beta sigma_{t-1}^2
# ---------------------------------------------------------------------------
GARCH_SPECS = {
    "GARCH-N": dict(o=0, dist="normal"),
    "GARCH-t": dict(o=0, dist="t"),
    "GJR-N": dict(o=1, dist="normal"),
    "GJR-t": dict(o=1, dist="t"),
}


def arch_backcast(eps, tau=75):
    """arch's start-up value: 0.94-weighted mean of the first 75 squared residuals."""
    w = 0.94 ** np.arange(min(tau, len(eps)))
    return float(np.sum(w / w.sum() * eps[:len(w)] ** 2))


def garch_filter(eps, omega, alpha, beta, gamma=0.0, backcast=None):
    """Conditional variance path. v[t] uses eps[0..t-1] only.

    Start-up follows arch: the backcast stands in for both eps_{-1}^2 and sigma_{-1}^2."""
    bc = arch_backcast(eps) if backcast is None else backcast
    v = np.empty(len(eps))
    v[0] = omega + (alpha + gamma / 2 + beta) * bc
    for t in range(1, len(eps)):
        e = eps[t - 1]
        v[t] = omega + (alpha + gamma * (e < 0)) * e**2 + beta * v[t - 1]
    return v


def fit_garch_own(eps):
    """Own Gaussian GARCH(1,1) MLE with SLSQP; enforces alpha + beta < 1."""
    def nll(p):
        v = garch_filter(eps, *p)
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(v) + eps**2 / v)

    with warnings.catch_warnings():
        # SLSQP occasionally probes just outside the bounds and clips; harmless
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        res = minimize(nll, [0.1 * eps.var(), 0.1, 0.8], method="SLSQP",
                       bounds=((1e-8, None), (0, 1), (0, 1)),
                       constraints={"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},
                       options={"ftol": 1e-10, "maxiter": 1000})
    return dict(zip(["omega", "alpha", "beta"], res.x)), -res.fun


def fit_garch_arch(eps, spec):
    return arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, **GARCH_SPECS[spec]).fit(disp="off")


def garch_params(fit):
    p = fit.params
    return {"omega": p["omega"], "alpha": p["alpha[1]"], "gamma": p.get("gamma[1]", 0.0),
            "beta": p["beta[1]"], "nu": p.get("nu", np.nan)}


def persistence(gp):
    """alpha + gamma/2 + beta (gamma term weighted by P(eps<0) = 1/2 for symmetric innovations)."""
    return gp["alpha"] + gp["gamma"] / 2 + gp["beta"]


def std_quantile(a, nu=np.nan):
    """a-quantile of the unit-variance innovation (normal, or t rescaled to variance 1)."""
    if np.isnan(nu):
        return stats.norm.ppf(a)
    return stats.t.ppf(a, nu) * np.sqrt((nu - 2) / nu)
