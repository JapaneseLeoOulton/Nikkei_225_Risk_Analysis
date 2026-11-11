"""Entry point: reproduces every result in the README and write-up from data/n225.csv.

    python run_all.py

Writes tables to results/ and figures to figures/. Runs top to bottom from a
fresh session; the only randomness (block bootstrap) is seeded.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

import backtests as bt
import data
import models
import plots

SEED = 42
LEVELS = [0.05, 0.01, 0.005]                   # tail probabilities: 95%, 99%, 99.5% VaR
LB_LAGS = [5, 10, 20, 40]
ROOT = Path(__file__).parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def save(df, name):
    df.to_csv(RESULTS / f"{name}.csv")
    return df


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    df = data.load()
    r = data.returns_pct(df)
    r_tr, r_te = data.split(r)
    x_tr, x_te = r_tr.to_numpy(), r_te.to_numpy()
    section("Data")
    print(f"Prices {df.index[0].date()} -> {df.index[-1].date()}; returns {r.index[0].date()} -> {r.index[-1].date()}")
    print(f"Train {r_tr.index[0].date()} -> {r_tr.index[-1].date()} (n={len(r_tr)}); "
          f"test {r_te.index[0].date()} -> {r_te.index[-1].date()} (n={len(r_te)})")

    # ------------------------------------------- Task 1: stylised facts
    section("Task 1: descriptive statistics (daily log returns, %)")
    desc = pd.DataFrame({name: {"n": len(s), "mean": s.mean(), "sd": s.std(), "skew": stats.skew(s),
                                "kurtosis": stats.kurtosis(s, fisher=False), "min": s.min(), "max": s.max()}
                         for name, s in [("full", r), ("train", r_tr), ("test", r_te)]}).T
    print(save(desc, "descriptives").round(4))

    lb = pd.DataFrame([{"h": h, **dict(zip(["Q r", "p r"], bt.ljung_box(x_tr, h))),
                        **dict(zip(["Q r^2", "p r^2"], bt.ljung_box(x_tr**2, h)))} for h in LB_LAGS]).set_index("h")
    print("\nLjung-Box on training returns and squared returns")
    print(save(lb, "ljung_box_raw").round(4))

    # ------------------------------------- Task 2: OLS AR(p) walk-forward
    section("Task 2: OLS AR(p) walk-forward (sklearn LinearRegression, refit daily on all earlier days)")
    table = models.lag_table(r, 5)
    test_dates = table.index[table.index >= data.SPLIT_DATE]
    y = table.loc[test_dates, "y"]
    mean_fc = models.expanding_mean_forecast(table, test_dates)
    rows = []
    for p in range(1, 6):
        f = models.ols_walk_forward(table, p, test_dates)
        row = {"p": p}
        for name, naive in [("zero", 0.0), ("mean", mean_fc)]:
            gain = (y - naive) ** 2 - (y - f) ** 2
            row[f"R2_oos vs {name}"] = 1 - ((y - f) ** 2).sum() / ((y - naive) ** 2).sum()
            row[f"t vs {name}"] = gain.mean() / (gain.std() / np.sqrt(len(gain)))
        rows.append(row)
    ols = save(pd.DataFrame(rows).set_index("p"), "ols_walk_forward")
    print(ols.round(5))

    # ------------------------------ Task 4: Gaussian vs Merton density fit
    section("Task 4: Gaussian vs Merton fitted on training data")
    gauss = models.fit_gaussian(x_tr)
    merton, ll_merton, ll_starts = models.fit_merton(x_tr)
    ll_gauss = float(np.sum(stats.norm.logpdf(x_tr, gauss["mu"], gauss["sigma"])))
    print("Gaussian:", {k: round(v, 4) for k, v in gauss.items()}, f"loglik {ll_gauss:.2f}")
    print("Merton:  ", {k: round(v, 4) for k, v in merton.items()}, f"loglik {ll_merton:.2f}",
          "(starts:", [round(float(v), 2) for v in ll_starts], ")")
    lr_m = 2 * (ll_merton - ll_gauss)
    print(f"Training LR Merton vs Gaussian = {lr_m:.1f} (3 extra params)")
    save(pd.DataFrame({"Gaussian": {**gauss, "loglik": ll_gauss},
                       "Merton": {**merton, "loglik": ll_merton, "LR vs Gaussian": lr_m}}).T, "density_fit")

    # out-of-sample log score difference d_t = log f_Merton - log f_Gauss
    d = models.merton_logpdf(x_te, **merton) - stats.norm.logpdf(x_te, gauss["mu"], gauss["sigma"])
    absr = np.abs(x_te)
    buckets = {"<1%": absr < 1, "1-2%": (absr >= 1) & (absr < 2), "2-3%": (absr >= 2) & (absr < 3), ">=3%": absr >= 3}
    bucket = pd.DataFrame({name: {"days": m.sum(), "sum d": d[m].sum(), "mean d": d[m].mean(),
                                  "share of total d": d[m].sum() / d.sum()} for name, m in buckets.items()}).T
    print(f"\nTest log score: mean d = {d.mean():.4f} per day, total {d.sum():.1f}, median {np.median(d):.4f}")
    print("Decomposition of d by |return| bucket")
    print(save(bucket, "merton_logscore_buckets").round(4))
    order_abs = np.argsort(absr)[::-1]
    print("Drop largest |return| days:",
          {k: round(float(np.delete(d, order_abs[:k]).mean()), 4) for k in [1, 3, 10]})
    ci = {L: bt.block_bootstrap_ci(d, L, 5000, rng) for L in [10, 20, 60]}
    print("Moving-block bootstrap 95% CI for mean d (B=5000):",
          {L: tuple(round(float(x), 4) for x in v) for L, v in ci.items()})
    pd.DataFrame(ci, index=["lo", "hi"]).T.rename_axis("block length").pipe(save, "merton_logscore_bootstrap")

    # ---------------------------------------------------- Task 6: GARCH
    section("Task 6: GARCH, fitted on training data (2000-2011)")
    mu = gauss["mu"]                         # constant mean = training sample mean
    eps_tr = x_tr - mu
    own, ll_own = models.fit_garch_own(eps_tr)
    fits = {s: models.fit_garch_arch(eps_tr, s) for s in models.GARCH_SPECS}
    a_n = models.garch_params(fits["GARCH-N"])
    check = pd.DataFrame({"own MLE": {**own, "loglik": ll_own},
                          "arch": {k: a_n[k] for k in ["omega", "alpha", "beta"]} | {"loglik": fits["GARCH-N"].loglikelihood}}).T
    print("GARCH(1,1)-N: own MLE vs arch (same start-up convention)")
    print(save(check, "garch_own_vs_arch").round(6))

    rows = []
    for s, f in fits.items():
        gp = models.garch_params(f)
        pers = models.persistence(gp)
        lrv = gp["omega"] / (1 - pers)
        z = np.asarray(f.std_resid)
        rows.append({"model": s, **gp, "persistence": pers, "half-life (days)": np.log(0.5) / np.log(pers),
                     "long-run vol % (annual)": np.sqrt(252 * lrv), "loglik": f.loglikelihood, "BIC": f.bic,
                     "kurtosis z": stats.kurtosis(z, fisher=False),
                     **{f"LB z^2 p({h})": bt.ljung_box(z**2, h)[1] for h in LB_LAGS}})
    gtab = save(pd.DataFrame(rows).set_index("model"), "garch_models")
    print(gtab.round(4).T)
    ll = gtab["loglik"]
    print(f"LR GARCH-N -> GARCH-t {2 * (ll['GARCH-t'] - ll['GARCH-N']):.1f}; "
          f"GARCH-N -> GJR-N {2 * (ll['GJR-N'] - ll['GARCH-N']):.1f}; "
          f"GARCH-t -> GJR-t {2 * (ll['GJR-t'] - ll['GARCH-t']):.1f}")
    chosen = gtab["BIC"].idxmin()
    print(f"Chosen on training BIC (before any backtest): {chosen}")
    print(f"Raw training returns kurtosis {stats.kurtosis(eps_tr, fisher=False):.2f}; "
          f"sample var {eps_tr.var():.3f} %^2")

    # ---------------------------------------------- VaR forecasts, test set
    # Same scheme for every model: parameters estimated once on 2000-2011 and
    # never refitted. GARCH's sigma_t^2 is still updated daily by the recursion
    # using returns up to t-1 only.
    eps_all = r.to_numpy() - mu
    n_tr = len(r_tr)
    bc = models.arch_backcast(eps_tr)        # start-up uses training data only
    sigma = {}
    for s, f in fits.items():
        gp = models.garch_params(f)
        v = models.garch_filter(eps_all, gp["omega"], gp["alpha"], gp["beta"], gp["gamma"], backcast=bc)
        assert np.allclose(np.sqrt(v[:n_tr]), f.conditional_volatility), "filter != arch in-sample"
        sigma[s] = np.sqrt(v[n_tr:])
    # no look-ahead: changing the final return leaves every forecast unchanged
    shocked = eps_all.copy(); shocked[-1] = -50.0
    gp = models.garch_params(fits[chosen])
    assert np.allclose(models.garch_filter(shocked, gp["omega"], gp["alpha"], gp["beta"], gp["gamma"], backcast=bc),
                       models.garch_filter(eps_all, gp["omega"], gp["alpha"], gp["beta"], gp["gamma"], backcast=bc))

    VaR = {"Gaussian": {a: np.full(len(x_te), models.gaussian_var(gauss, a)) for a in LEVELS},
           "Merton": {a: np.full(len(x_te), models.merton_var(merton, a)) for a in LEVELS}}
    for s, f in fits.items():
        nu = models.garch_params(f)["nu"]
        VaR[s] = {a: -(mu + sigma[s] * models.std_quantile(a, nu)) for a in LEVELS}

    section(f"Backtests, test {r_te.index[0].date()} -> {r_te.index[-1].date()} (N={len(x_te)})")
    breaches, rows = {}, []
    for m in VaR:
        for a in LEVELS:
            I = x_te < -VaR[m][a]
            breaches[(m, a)] = I
            rows.append({"model": m, "VaR level": f"{1 - a:.1%}", **bt.backtest(I, a)})
    btab = save(pd.DataFrame(rows).set_index(["model", "VaR level"]), "backtests_all_models")
    main_models = ["Gaussian", "Merton", chosen]
    print(btab.loc[main_models].round(4))
    print("\nOther GARCH variants (one change at a time from GARCH-N)")
    print(btab.loc[[s for s in models.GARCH_SPECS if s != chosen]].round(4))

    # ---------------------------------------------- breach timing
    years = r_te.index.year
    per_year = pd.DataFrame({m: pd.Series(breaches[(m, 0.01)], index=r_te.index).groupby(years).sum()
                             for m in main_models})
    per_year["expected"] = pd.Series(0.01, index=r_te.index).groupby(years).sum()
    section("99% VaR breaches per calendar year")
    print(save(per_year, "breaches_per_year_99").round(1).T)

    # ---------------------------------------------- figures
    plots.qq_figure(eps_tr / eps_tr.std(), np.asarray(fits["GARCH-N"].std_resid),
                    np.asarray(fits[chosen].std_resid), models.garch_params(fits[chosen])["nu"], chosen,
                    FIGURES / "qq_residuals.png")
    plots.result_figure(r_te, {m: VaR[m][0.01] for m in main_models},
                        {m: breaches[(m, 0.01)] for m in main_models},
                        {m: btab.loc[(m, "99.0%")] for m in main_models},
                        FIGURES / "var99_breaches.png")
    print(f"\nTables written to {RESULTS.relative_to(ROOT)}/, figures to {FIGURES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
