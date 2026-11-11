import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp, gammaln, xlogy
from scipy.optimize import minimize, brentq
from scipy.stats import norm, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SHOW = False          # True -> plt.show() each figure; False -> just save to Figures/
if not SHOW:
    matplotlib.use("Agg")


def finish(fig_name):
    plt.tight_layout()
    plt.savefig(f"Figures/{fig_name}.png", dpi=130)
    if SHOW:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 0. Data
#    Use the frozen CSV from Task 1 instead of re-downloading, so Gaussian,
#    Merton and GARCH are all backtested on exactly the same test days
#    (a fresh yf.download adds new days every time the script is run).
# ---------------------------------------------------------------------------
df = pd.read_csv("data/n225.csv", header=[0, 1], index_col=0, parse_dates=True)
df.columns = df.columns.get_level_values(0)          # drop the "Ticker" header row
df.index.name = "Date"

r_all = df["Log_Return"].dropna() * 100              # % log returns (see section 2 for why)

split_date = "2012-01-01"
r_tr = r_all[r_all.index < split_date]
r_te = r_all[r_all.index >= split_date]

r_pct_train = r_tr.values
r_pct_test = r_te.values
r_train = r_pct_train / 100                           # decimal, as in earlier tasks
print(f"train {r_tr.index[0].date()} -> {r_tr.index[-1].date()}  n={len(r_tr)}")
print(f"test  {r_te.index[0].date()} -> {r_te.index[-1].date()}  n={len(r_te)}")


# ---------------------------------------------------------------------------
# 1. Ljung-Box on raw returns (training data)
# ---------------------------------------------------------------------------
def sample_acf(x, k):
    x = np.asarray(x) - np.mean(x)
    return np.sum(x[k:] * x[:-k]) / np.sum(x**2)

def ljung_box(x, h):
    n = len(x)
    ks = np.arange(1, h + 1)
    rho = np.array([sample_acf(x, k) for k in ks])
    Q = n * (n + 2) * np.sum(rho**2 / (n - ks))
    p_value = 1 - chi2.cdf(Q, df=h)
    return Q, p_value, rho


hs = [5, 10, 20, 40]

rows = []
for h in hs:
    Q_r,  p_r,  _ = ljung_box(r_train,    h)
    Q_r2, p_r2, _ = ljung_box(r_train**2, h)
    rows.append({"h": h, "Q returns": Q_r, "p returns": p_r,
                 "Q squared": Q_r2, "p squared": p_r2})
lb_raw = pd.DataFrame(rows).set_index("h")
print("\n=== 1. Ljung-Box, raw training returns ===")
print(lb_raw.round(4))

# cross-check own function against statsmodels
sm_lb = acorr_ljungbox(r_train**2, lags=hs, return_df=True)
assert np.allclose(sm_lb["lb_stat"].values, lb_raw["Q squared"].values)

# seems to be volatility clustering in the returns (squared returns massively
# autocorrelated), so fit a GARCH(1,1) to the demeaned returns


# ---------------------------------------------------------------------------
# 2. Own GARCH(1,1) MLE  vs  the arch package
# ---------------------------------------------------------------------------
def garch_variance(params, eps, v0=None):
    omega, alpha, beta = params
    v = np.empty(len(eps))
    if v0 == "arch":
        # arch's convention: the backcast stands in for BOTH eps_{-1}^2 and sigma_{-1}^2
        v[0] = omega + (alpha + beta) * arch_backcast(eps)
    else:
        v[0] = eps.var() if v0 is None else v0         # starting value
    for t in range(1, len(eps)):
        v[t] = omega + alpha * eps[t-1]**2 + beta * v[t-1]
    return v

def negative_log_likelihood(params, eps, v0=None):
    v = garch_variance(params, eps, v0)
    # full Gaussian log-likelihood, including the 2*pi constant, so the number
    # is directly comparable with arch's .loglikelihood
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(v) + eps**2 / v)

def fit_garch(eps, v0=None, ftol=1e-10):
    X0 = [0.1 * eps.var(), 0.1, 0.8]  # Initial guess for omega, alpha, beta
    return minimize(
        negative_log_likelihood, X0, args=(eps, v0), method="SLSQP",  # SLSQP handles bounds and inequality constraints
        bounds=((1e-8, None), (0, 1), (0, 1)),
        constraints={"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]},  # alpha + beta < 1
        options={"ftol": ftol, "maxiter": 1000})

def arch_backcast(eps, tau=75):
    # arch's default starting variance: exponentially weighted (0.94) mean of
    # the first 75 squared residuals
    w = 0.94 ** np.arange(min(tau, len(eps)))
    return np.sum(w / w.sum() * eps[:len(w)]**2)


mu_train = r_pct_train.mean()                  # constant mean, training data only
eps = r_pct_train - mu_train                   # demeaned returns (the old code fitted raw r_pct_train by mistake)

fits = {
    "own, v0 = sample var, ftol 1e-10": fit_garch(eps),
    "own, v0 = sample var, ftol default": fit_garch(eps, ftol=1e-6),
    "own, arch start-up (backcast)": fit_garch(eps, v0="arch"),
}
arch_zero = arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
arch_const = arch_model(r_pct_train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

rows = [{"fit": k, "mu": mu_train, "omega": f.x[0], "alpha": f.x[1], "beta": f.x[2], "loglik": -f.fun}
        for k, f in fits.items()]
rows.append({"fit": "arch ZeroMean on eps", "mu": mu_train, **dict(zip(["omega", "alpha", "beta"], arch_zero.params.values)),
             "loglik": arch_zero.loglikelihood})
rows.append({"fit": "arch ConstantMean on r (mu estimated)", "mu": arch_const.params["mu"],
             **dict(zip(["omega", "alpha", "beta"], arch_const.params.values[1:])), "loglik": arch_const.loglikelihood})

# Why scale by 100: same model on decimal returns. Convert the log-lik back to
# % units (+ n*ln(100)) so it is comparable; omega scales by 100^2.
eps_dec = eps / 100
f_dec = fit_garch(eps_dec, ftol=1e-6)
rows.append({"fit": "own on DECIMAL returns (omega x1e4)", "mu": mu_train,
             "omega": f_dec.x[0] * 1e4, "alpha": f_dec.x[1], "beta": f_dec.x[2],
             "loglik": -f_dec.fun - len(eps) * np.log(100)})

cmp = pd.DataFrame(rows).set_index("fit")
print("\n=== 2. Own MLE vs arch (training 2000-2011, % returns) ===")
print(cmp.round(5))
print("Gap explanation:")
print(" - own(arch start-up) vs arch ZeroMean: same likelihood, only optimiser tolerance differs")
print(" - own(v0 = sample var) vs arch: different starting variance v0; affects early terms of the likelihood only")
print(" - arch ConstantMean estimates mu jointly (GARCH-weighted) instead of using the sample mean -> higher log-lik")
print(" - decimal returns: omega ~ 1e-5 is tiny next to alpha,beta ~ 0.1, so the optimiser's steps/tolerances are")
print("   badly scaled and it stops early at a worse log-lik. Scaling by 100 fixes this.")

omega, alpha, beta = fits["own, v0 = sample var, ftol 1e-10"].x


# ---------------------------------------------------------------------------
# 3. Parameter sanity checks
# ---------------------------------------------------------------------------
def summarise_vol(name, omega, persistence):
    lrv = omega / (1 - persistence)
    return {"model": name, "persistence": persistence,
            "long-run var (%^2)": lrv, "long-run daily vol %": np.sqrt(lrv),
            "annualised vol %": np.sqrt(252 * lrv),
            "half-life (days)": np.log(0.5) / np.log(persistence)}

print("\n=== 3. Sanity checks (own GARCH(1,1)-N) ===")
print(f"alpha + beta = {alpha + beta:.4f}  (< 1: {alpha + beta < 1})")
print(pd.Series(summarise_vol("GARCH-N own", omega, alpha + beta)).to_string())
print(f"sample variance of training returns = {eps.var():.4f} %^2  (sd {eps.std():.4f} %)")


# ---------------------------------------------------------------------------
# 4. Residual diagnostics: z_t = eps_t / sigma_t
# ---------------------------------------------------------------------------
sigma_tr = np.sqrt(garch_variance((omega, alpha, beta), eps))
z = eps / sigma_tr

rows = []
for h in hs:
    Qz, pz, _ = ljung_box(z, h)
    Qz2, pz2, _ = ljung_box(z**2, h)
    rows.append({"h": h, "Q r^2 (raw)": lb_raw.loc[h, "Q squared"], "p r^2 (raw)": lb_raw.loc[h, "p squared"],
                 "Q z^2": Qz2, "p z^2": pz2, "Q z": Qz, "p z": pz})
lb_z = pd.DataFrame(rows).set_index("h")
print("\n=== 4. Ljung-Box: raw squared returns vs squared standardised residuals ===")
print(lb_z.round(4))
print("(strictly, for z^2 from a fitted GARCH(1,1) the df should be h-2 (Li & Mak); it does not change any conclusion here)")

def tail_stats(name, x):
    jb = stats.jarque_bera(x)
    return {"series": name, "skew": stats.skew(x), "kurtosis": stats.kurtosis(x, fisher=False),
            "JB p-value": jb.pvalue,
            "share |x|>3 sd": np.mean(np.abs((x - x.mean()) / x.std()) > 3)}

tails = pd.DataFrame([tail_stats("raw returns", eps), tail_stats("z (GARCH-N)", z)]).set_index("series")
print("\nTails (Gaussian: kurtosis 3, share beyond 3 sd = 0.0027)")
print(tails.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
stats.probplot(eps / eps.std(), dist="norm", plot=axes[0])
axes[0].set_title("Raw training returns vs Gaussian")
stats.probplot(z, dist="norm", plot=axes[1])
axes[1].set_title("GARCH-N standardised residuals vs Gaussian")
finish("Task6 QQ raw vs GARCH residuals")


# ---------------------------------------------------------------------------
# 5. Innovation distribution / leverage: change one thing at a time
#    GARCH-N -> GARCH-t   (distribution only)
#    GARCH-N -> GJR-N     (leverage only)
#    -> GJR-t             (both)
#    Model choice is made on TRAINING evidence only (LR / BIC), before looking
#    at the backtest, so the backtest stays an honest out-of-sample test.
# ---------------------------------------------------------------------------
specs = {
    "GARCH-N": dict(o=0, dist="normal"),
    "GARCH-t": dict(o=0, dist="t"),
    "GJR-N":   dict(o=1, dist="normal"),
    "GJR-t":   dict(o=1, dist="t"),
}
garch_fits = {name: arch_model(eps, mean="Zero", vol="GARCH", p=1, q=1, **s).fit(disp="off")
              for name, s in specs.items()}

rows, vol_rows = [], []
for name, f in garch_fits.items():
    p = f.params
    gamma_ = p.get("gamma[1]", 0.0)
    pers = p["alpha[1]"] + gamma_ / 2 + p["beta[1]"]        # symmetric innovations -> P(eps<0) = 1/2
    zf = f.std_resid
    rows.append({"model": name, "omega": p["omega"], "alpha": p["alpha[1]"], "gamma": gamma_,
                 "beta": p["beta[1]"], "nu": p.get("nu", np.nan), "loglik": f.loglikelihood,
                 "AIC": f.aic, "BIC": f.bic, "kurt(z)": stats.kurtosis(zf, fisher=False),
                 "LB z^2 p(20)": ljung_box(zf**2, 20)[1]})
    vol_rows.append(summarise_vol(name, p["omega"], pers))
model_tab = pd.DataFrame(rows).set_index("model")
print("\n=== 5. Innovation / leverage comparison (training only) ===")
print(model_tab.round(4))
print(pd.DataFrame(vol_rows).set_index("model").round(4))

ll = model_tab["loglik"]
lr_t = 2 * (ll["GARCH-t"] - ll["GARCH-N"])
lr_gjr = 2 * (ll["GJR-N"] - ll["GARCH-N"])
lr_gjr_t = 2 * (ll["GJR-t"] - ll["GARCH-t"])
print(f"\nLR GARCH-N -> GARCH-t : {lr_t:.2f}  (nu=inf on the boundary; 50:50 chi2(0)/chi2(1) p = {0.5 * chi2.sf(lr_t, 1):.2e})")
print(f"LR GARCH-N -> GJR-N   : {lr_gjr:.2f}  p = {chi2.sf(lr_gjr, 1):.2e}")
print(f"LR GARCH-t -> GJR-t   : {lr_gjr_t:.2f}  p = {chi2.sf(lr_gjr_t, 1):.2e}")
CHOSEN = model_tab["BIC"].idxmin()
print(f"Lowest BIC on training data: {CHOSEN}")

# QQ of z against the distribution each model assumes
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
stats.probplot(garch_fits["GARCH-N"].std_resid, dist="norm", plot=axes[0])
axes[0].set_title("GARCH-N residuals vs Gaussian")
nu_c = garch_fits[CHOSEN].params.get("nu", np.inf)
z_c = garch_fits[CHOSEN].std_resid
if np.isfinite(nu_c):
    stats.probplot(z_c / np.sqrt((nu_c - 2) / nu_c), dist=stats.t, sparams=(nu_c,), plot=axes[1])
    axes[1].set_title(f"{CHOSEN} residuals vs Student-t (nu={nu_c:.1f})")
else:
    stats.probplot(z_c, dist="norm", plot=axes[1])
    axes[1].set_title(f"{CHOSEN} residuals vs Gaussian")
finish("Task6 QQ GARCH-N vs chosen")


# ---------------------------------------------------------------------------
# 6. Rolling one-day-ahead VaR
#    Scheme (same for all three models): parameters estimated ONCE on
#    2000-2011 and held fixed. Gaussian and Merton therefore give a constant
#    VaR; GARCH's VaR still moves because sigma_t^2 is updated each day from
#    the recursion, using returns up to t-1 only. The only thing that differs
#    between Gaussian and GARCH is the time-varying variance.
# ---------------------------------------------------------------------------
LEVELS = [0.05, 0.01, 0.005]                       # 95%, 99%, 99.5%
eps_all = (r_all - mu_train).values                # test days demeaned with the TRAINING mean
is_test = (r_all.index >= split_date)

def sigma_path(fit, eps_all):
    """Filter sigma_t through train+test with fixed training parameters.
    sigma[t] uses eps[0..t-1] only."""
    p = fit.params
    o_, a_, g_, b_ = p["omega"], p["alpha[1]"], p.get("gamma[1]", 0.0), p["beta[1]"]
    v = np.empty(len(eps_all))
    bc = arch_backcast(eps_all[:len(r_tr)])          # from the first 75 TRAINING days only
    v[0] = o_ + (a_ + g_ / 2 + b_) * bc
    for t in range(1, len(eps_all)):
        e = eps_all[t-1]
        v[t] = o_ + (a_ + g_ * (e < 0)) * e**2 + b_ * v[t-1]
    return np.sqrt(v)

def std_quantile(fit, a):
    nu_ = fit.params.get("nu", None)
    if nu_ is None:
        return norm.ppf(a)
    return stats.t.ppf(a, nu_) * np.sqrt((nu_ - 2) / nu_)     # unit-variance t

sig = {}
for name, f in garch_fits.items():
    s = sigma_path(f, eps_all)
    # check against arch with the same fixed parameters on the full sample
    fixed = arch_model(eps_all, mean="Zero", vol="GARCH", p=1, q=1, **specs[name]).fix(f.params.values)
    assert np.allclose(s[len(r_tr):], fixed.conditional_volatility[len(r_tr):], rtol=1e-6)
    # and the training part must equal the in-sample fit
    assert np.allclose(s[:len(r_tr)], f.conditional_volatility, rtol=1e-6)
    sig[name] = s[is_test]

# no-look-ahead check: change the last test return; every sigma up to and
# including that day must be unchanged
e_shock = eps_all.copy(); e_shock[-1] = -50
assert np.allclose(sigma_path(garch_fits["GARCH-N"], e_shock), sigma_path(garch_fits["GARCH-N"], eps_all))

# Gaussian (Task 3): fixed training mean / sd
sd_train = r_pct_train.std(ddof=1)

# Merton (Tasks 4-5): refit on training data, then fixed
def neg_ll_merton(theta, r, K=10):
    mu, log_sigma, log_lam, m, log_delta = theta
    sigma, lam, delta = np.exp(log_sigma), np.exp(log_lam), np.exp(log_delta)
    k = np.arange(K + 1)
    log_w = -lam + k * log_lam - gammaln(k + 1)
    means = mu + k * m
    sd = np.sqrt(sigma**2 + k * delta**2)
    log_phi = stats.norm.logpdf(r[:, None], loc=means, scale=sd)
    return -np.sum(logsumexp(log_w + log_phi, axis=1))

mres = minimize(neg_ll_merton, x0=[0.01848, 0.2291, -2.340, -0.4775, 1.1424], args=(r_pct_train,),
                method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
mp = dict(mu=mres.x[0], sigma=np.exp(mres.x[1]), lam=np.exp(mres.x[2]), m=mres.x[3], delta=np.exp(mres.x[4]))
print("\nMerton training params:", {k: round(float(v), 4) for k, v in mp.items()})

def merton_cdf(x, mu, sigma, lam, m, delta, K=10):
    k = np.arange(K + 1)
    w = np.exp(-lam + k * np.log(lam) - gammaln(k + 1))
    return np.sum(w * norm.cdf(x, loc=mu + k * m, scale=np.sqrt(sigma**2 + k * delta**2)))

VaR = {}       # VaR[model][a] -> array over test days (positive number, in %)
VaR["Gaussian"] = {a: np.full(len(r_te), -(mu_train + norm.ppf(a) * sd_train)) for a in LEVELS}
VaR["Merton"] = {a: np.full(len(r_te), -brentq(lambda x: merton_cdf(x, **mp) - a, -30, 0)) for a in LEVELS}
for name, f in garch_fits.items():
    VaR[name] = {a: -(mu_train + sig[name] * std_quantile(f, a)) for a in LEVELS}


# ---------------------------------------------------------------------------
# 7. Backtests: Kupiec, Christoffersen, LR_cc
# ---------------------------------------------------------------------------
def kupiec_uc(x, n, alpha):
    p_hat = x / n
    ll_null = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (ll_null - ll_alt)
    return lr, chi2.sf(lr, df=1)

def christoffersen_ind(I):
    # same test as Task 3, written in logs (xlogy) so 0*log(0) = 0 and no underflow
    i = np.asarray(I, dtype=int)
    t0, t1 = i[:-1], i[1:]
    m00 = np.sum((t0 == 0) & (t1 == 0)); m01 = np.sum((t0 == 0) & (t1 == 1))
    m10 = np.sum((t0 == 1) & (t1 == 0)); m11 = np.sum((t0 == 1) & (t1 == 1))
    pi01 = m01 / (m00 + m01)
    pi11 = m11 / (m10 + m11) if (m10 + m11) > 0 else 0.0
    pi = (m01 + m11) / (m00 + m01 + m10 + m11)
    ll_null = xlogy(m00 + m10, 1 - pi) + xlogy(m01 + m11, pi)
    ll_alt = xlogy(m00, 1 - pi01) + xlogy(m01, pi01) + xlogy(m10, 1 - pi11) + xlogy(m11, pi11)
    lr = 2 * (ll_alt - ll_null)
    return lr, chi2.sf(lr, df=1), m11

breaches = {}
rows = []
for model in VaR:
    for a in LEVELS:
        I = r_pct_test < -VaR[model][a]
        breaches[(model, a)] = I
        lr_uc, p_uc = kupiec_uc(I.sum(), len(I), a)
        lr_ind, p_ind, m11 = christoffersen_ind(I)
        lr_cc = lr_uc + lr_ind
        rows.append({"model": model, "level": f"{1 - a:.1%}", "expected": a * len(I), "breaches": int(I.sum()),
                     "rate %": 100 * I.mean(), "back-to-back": int(m11),
                     "LR_uc": lr_uc, "p_uc": p_uc, "LR_ind": lr_ind, "p_ind": p_ind,
                     "LR_cc": lr_cc, "p_cc": chi2.sf(lr_cc, df=2)})
bt = pd.DataFrame(rows)

print("\n=== 7. Backtest, test period", r_te.index[0].date(), "->", r_te.index[-1].date(), f"(N={len(r_te)}) ===")
three_way = ["Gaussian", "Merton", "GARCH-N", CHOSEN]
for lvl in ["95.0%", "99.0%", "99.5%"]:
    print(f"\n--- {lvl} ---")
    print(bt[(bt["level"] == lvl) & bt["model"].isin(three_way)].set_index("model").drop(columns="level").round(4))
print("\nAll GARCH variants (one change at a time):")
print(bt[bt["model"].str.contains("GARCH|GJR")].set_index(["model", "level"]).round(4))

# reproduces Task 3? (Gaussian 99%: LR_ind was 6.713, p = 0.0096 on the earlier download)
g99 = bt[(bt.model == "Gaussian") & (bt.level == "99.0%")].iloc[0]
print(f"\nGaussian 99% check: breaches={g99.breaches}, LR_ind={g99.LR_ind:.3f}, p_ind={g99.p_ind:.4f}")


# ---------------------------------------------------------------------------
# 8. Breach dates: have the 2015 / 2020 / 2024 clusters gone?
# ---------------------------------------------------------------------------
yrs = r_te.index.year
by_year = pd.DataFrame({m: pd.Series(breaches[(m, 0.01)], index=r_te.index).groupby(yrs).sum()
                        for m in three_way})
by_year["expected"] = pd.Series(1, index=r_te.index).groupby(yrs).sum() * 0.01
print("\n=== 8. 99% breaches per year ===")
print(by_year.round(1).to_string())

def clusters(I, window=10):
    """Breaches that have another breach within `window` trading days before them."""
    idx = np.flatnonzero(I)
    return [r_te.index[j].date() for k, j in enumerate(idx) if k > 0 and j - idx[k - 1] <= window]

for m in three_way:
    print(f"{m:9s} 99% breaches within 10 days of the previous one: {clusters(breaches[(m, 0.01)])}")

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for ax, m in zip(axes, ["Gaussian", "Merton", CHOSEN]):
    I = breaches[(m, 0.01)]
    ax.plot(r_te.index, r_pct_test, lw=0.5, color="0.6", label="daily log return (%)")
    ax.plot(r_te.index, -VaR[m][0.01], color="tab:blue", lw=0.9, label="-VaR 99%")
    ax.scatter(r_te.index[I], r_pct_test[I], color="red", s=14, zorder=3, label=f"breaches ({I.sum()})")
    for y in (2015, 2020, 2024):
        ax.axvspan(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"), color="orange", alpha=0.08)
    ax.set_title(f"{m} 99% VaR, test period (shaded: 2015, 2020, 2024)")
    ax.set_ylabel("%")
    ax.legend(loc="lower left", fontsize=8)
finish("Task6 99 VaR breaches Gaussian Merton GARCH")
