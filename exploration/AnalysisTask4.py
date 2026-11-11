import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize


df = yf.download("^N225", start="2000-01-01", auto_adjust=True)
df["Log_Return"] = np.log(df['Close'] / df['Close'].shift(1))

split_date = "2012-01-01"

train = df.loc[df.index < split_date]
test  = df.loc[df.index >= split_date]

r_train = train["Log_Return"].dropna().values
r_test = test["Log_Return"].dropna().values

def neg_ll_gauss(theta, r):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    return -np.sum(stats.norm.logpdf(r, loc=mu, scale=sigma))

r_pct = r_train * 100 # due to decimal scale optiiser could stop early

res =minimize(neg_ll_gauss, x0=[0.0, np.log(0.01)], args=(r_pct,), method="Nelder-Mead")
print(res.success, res.fun, res.x)

print(np.mean(r_train), np.std(r_train))


from scipy.special import logsumexp, gammaln

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

r_pct = r_train * 100 # due to decimal scale optiiser could stop early

mu0 = np.mean(r_pct)
starts = [
    [mu0, np.log(1.2), np.log(0.10), -0.5, np.log(3.0)],   # implied var about 2.4
    [mu0, np.log(1.0), np.log(0.05), -1.0, np.log(4.0)],   # about 1.9
    [mu0, np.log(1.4), np.log(0.02), -1.0, np.log(5.0)],   # about 2.5
]


from scipy.stats import norm



for x0 in starts:
    res2 = minimize(neg_ll_merton, x0=x0, args=(r_pct,), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    print(res2.success, res2.fun, res2.x)




stats.kurtosis(r_train)



r_pct_test = r_train * 100




def merton_log_density(r, mu, sigma, lam, m, delta, K=25):
    r = np.asarray(r)
    log_components = []
    for k in range(K + 1):
        log_weight = -lam + k * np.log(lam) - gammaln(k + 1)
        std_dev_k = np.sqrt(sigma**2 + k * delta**2)
        mean_k = mu + k * m
        log_components.append(log_weight + norm.logpdf(r, loc=mean_k, scale=std_dev_k))
    return logsumexp(np.array(log_components), axis=0)

params = {
    "mu": 0.01848,
    "sigma": np.exp(0.2291),
    "lam": np.exp(-2.340),
    "m": -0.4775,
    "delta": np.exp(1.1424),
}

r_test_pct = test["Log_Return"].dropna().values * 100
s_M = merton_log_density(r_test_pct, **params)
s_G = norm.logpdf(r_test_pct, loc=-0.0275, scale=np.exp(0.4717))
d = s_M - s_G
print(d.mean(), np.median(d))



s_M = merton_log_density(r_test_pct, **params)
d = s_M - s_G
print(d.mean(), np.median(d), d.min(), d.max())

print(test["Log_Return"].isna().sum())



print(test["Log_Return"].isna().sum())

import numpy as np
import pandas as pd

# Dates aligned with d and r_test_pct (same NaN-free test returns)
dates = test["Log_Return"].dropna().index
assert len(dates) == len(d) == len(r_test_pct)      # all 3618

a = np.abs(r_test_pct)

# ---------- 1. Bucket sums ----------
buckets = {
    "<1%":  a < 1,
    "1-2%": (a >= 1) & (a < 2),
    "2-3%": (a >= 2) & (a < 3),
    ">=3%": a >= 3,
}

rows = []
for name, mask in buckets.items():
    rows.append({
        "bucket": name,
        "days": mask.sum(),
        "sum_d": d[mask].sum(),
        "mean_d": d[mask].mean(),
        "share_of_total": d[mask].sum() / d.sum(),
    })
bucket_table = pd.DataFrame(rows)
print(bucket_table.to_string(index=False))

# Sanity checks: buckets should cover every day exactly once
print(sum(m.sum() for m in buckets.values()), len(d))        # 3618 3618
print(sum(d[m].sum() for m in buckets.values()), d.sum())    # same number


# ---------- 2. Removal check ----------
order = np.argsort(d)[::-1]          # indices, largest d first

print("\nTop 10 days for Merton:")
print(pd.DataFrame({"date": dates[order[:10]],
                    "return_%": r_test_pct[order[:10]],
                    "d": d[order[:10]]}).to_string(index=False))

print("\nRemoval check (drop the days where Merton did best):")
print("removed  mean_d    total_d")
print(f"none     {d.mean():.4f}   {d.sum():.1f}")
for k in [1, 3, 10]:
    kept = np.delete(d, order[:k])
    print(f"top {k:<3}  {kept.mean():.4f}   {kept.sum():.1f}")

# Harsher-looking but outcome-independent version: drop the largest |return| days
order_abs = np.argsort(a)[::-1]
print("\nRemoval by largest |return| (does not pick days by model outcome):")
for k in [1, 3, 10]:
    kept = np.delete(d, order_abs[:k])
    print(f"top {k:<3}  {kept.mean():.4f}   {kept.sum():.1f}")

by_year = pd.Series(d, index=dates).groupby(dates.year).mean()
print(by_year)


rng = np.random.default_rng(42)           # fixed seed for reproducibility
n = len(d)
B = 5000

def block_boot_means(d, L, B, rng):
    n = len(d)
    nb = int(np.ceil(n / L))
    out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - L + 1, size=nb)
        idx = (starts[:, None] + np.arange(L)).ravel()[:n]
        out[b] = d[idx].mean()
    return out

for L in [10, 20, 60]:
    boots = block_boot_means(d, L, B, rng)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(L, round(lo, 4), round(hi, 4))