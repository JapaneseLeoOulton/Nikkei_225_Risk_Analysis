import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gamma

df = yf.download("^N225", start="2000-01-01", auto_adjust=True)
print(df.tail())

df["Log_Return"]= np.log(df['Close'] / df['Close'].shift(1))
print(df.tail())

split_date = "2012-01-01"

train = df.loc[df.index < split_date]
test  = df.loc[df.index >= split_date]

r_train = train["Log_Return"].dropna().values
r_test = test["Log_Return"].dropna().values


r_pct_train = r_train * 100 # due to decimal scale optiiser could stop early

r_pct_test = r_test * 100

def neg_ll_merton(theta, r, K=10):
    mu, log_sigma, log_lam, m, log_delta = theta
    sigma, lam, delta = np.exp(log_sigma), np.exp(log_lam), np.exp(log_delta)

    k = np.arange(K + 1)

    log_w = -lam + k * log_lam - gammaln(k + 1)      # shape (K+1,)
    means = mu + k * m                               # shape (K+1,)
    sd = np.sqrt(sigma**2 + k * delta**2)            # shape (K+1,)

    log_phi = stats.norm.logpdf(r[:, None], loc=means, scale=sd)   # shape (n, K+1)
    log_f = logsumexp(log_w + log_phi, axis=1)                     # shape (n,)

    return -np.sum(log_f)


mu0 = np.mean(r_pct_train)
starts = [
    [mu0, np.log(1.2), np.log(0.10), -0.5, np.log(3.0)],   #implied var about 2.4
    [mu0, np.log(1.0), np.log(0.05), -1.0, np.log(4.0)],   # about 1.9
    [mu0, np.log(1.4), np.log(0.02), -1.0, np.log(5.0)],   # about 2.5
]

for x0 in starts:
    res2 = minimize(neg_ll_merton, x0=x0, args=(r_pct_train,), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    print(res2.success, res2.fun, res2.x)

params = {
    "mu": 0.01848,
    "sigma": np.exp(0.2291),
    "lam": np.exp(-2.340),
    "m": -0.4775,
    "delta": np.exp(1.1424),
}

def merton_density(r, mu, sigma, lam, m, delta, K=10):
    r = np.asarray(r)
    components = []
    for k in range(K + 1):
        weight = np.exp(-lam) * lam**k / gamma(k + 1)
        std_k = np.sqrt(sigma**2 + k * delta**2)
        mean_k = mu + k * m
        components.append(weight * norm.pdf(r, loc=mean_k, scale=std_k))
    return np.sum(components, axis=0)

dens = merton_density(r_pct_test, **params)


from scipy.optimize import brentq

def merton_cdf(r, mu, sigma, lam, m, delta, K=10):
    r = np.asarray(r)
    components = []
    for k in range(K + 1):
        weight = np.exp(-lam) * lam**k / gamma(k + 1)
        std_k = np.sqrt(sigma**2 + k * delta**2)
        mean_k = mu + k * m
        components.append(weight * norm.cdf(r, loc=mean_k, scale=std_k))
    return np.sum(components, axis=0)


alpha = 0.05
q = brentq(lambda x: merton_cdf(x, **params) - alpha, -30, 0)
var_merton = -q


viol = (r_pct_test < q)
print(viol.sum(), viol.mean())

np.percentile(r_pct_test, [0.5, 1, 5])


import matplotlib.pyplot as plt

u = merton_cdf(r_pct_test, **params)   

plt.hist(u, bins=20, density=True)
plt.axhline(1, color="k", ls="--")     
plt.xlabel("u = F(r)"); plt.ylabel("density")
plt.show()


from scipy.stats import chi2



def christoffersen_ind_test(I):
    
    i = pd.Series(I).values

    t_0 = i[:-1]  # Day t
    t_1 = i[1:]  # Day t+1

    m00 = np.sum((t_0 == 0) & (t_1 == 0))
    m01 = np.sum((t_0 == 0) & (t_1 == 1))
    m10 = np.sum((t_0 == 1) & (t_1 == 0))
    m11 = np.sum((t_0 == 1) & (t_1 == 1))

    pi_01 = m01 / (m00 + m01) if (m00 + m01) > 0 else 0
    pi_11 = m11 / (m10 + m11) if (m10 + m11) > 0 else 0

    pi_overall = (m01 + m11) / (m00 + m01 + m10 + m11)

    L_null = ((1 - pi_overall) ** (m00 + m10)) * (pi_overall ** (m01 + m11))

    L_alt = (
        ((1 - pi_01) ** m00)
        * (pi_01**m01)
        * ((1 - pi_11) ** m10)
        * (pi_11**m11)
    )

    lr_ind = -2 * np.log(L_null) + 2 * np.log(L_alt)

    p_value = 1 - chi2.cdf(lr_ind, df=1)

    
    return {
        "Matrix": {"M00": m00, "M01": m01, "M10": m10, "M11": m11},
        "Probabilities": {"pi_01": pi_01, "pi_11": pi_11},
        "LR_ind": lr_ind,
        "p-value": p_value,
    }


def merton_var(alpha, mu, sigma, lam, m, delta, K=10):
    q = brentq(lambda x: merton_cdf(x, mu, sigma, lam, m, delta, K) - alpha, -30, 0)
    return -q

var_gauss  = -(r_pct_train.mean() + norm.ppf(0.01) * r_pct_train.std())
var_merton = merton_var(0.01, **params)

I_gauss  = (r_pct_test < -var_gauss)
I_merton = (r_pct_test < -var_merton)

christoffersen_ind_test(I_gauss)
christoffersen_ind_test(I_merton)


from scipy.special import xlogy
from scipy.stats import chi2

def kupiec_uc(x, n, alpha):
    p_hat = x / n
    ll_null = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll_alt  = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (ll_null - ll_alt)
    return lr, chi2.sf(lr, df=1)


print(kupiec_uc(I_gauss.sum(), len(I_gauss), 0.01))
print(kupiec_uc(I_merton.sum(), len(I_merton), 0.01))

