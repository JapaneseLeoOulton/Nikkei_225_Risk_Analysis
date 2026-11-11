"""Statistical tests: Ljung-Box, VaR backtests (Kupiec, Christoffersen, LR_cc),
moving-block bootstrap."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def ljung_box(x, h):
    """Ljung-Box Q over lags 1..h; p-value from chi2 with h degrees of freedom."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    ks = np.arange(1, h + 1)
    rho = np.array([np.sum(x[k:] * x[:-k]) for k in ks]) / np.sum(x**2)
    Q = n * (n + 2) * np.sum(rho**2 / (n - ks))
    return Q, chi2.sf(Q, h)


def kupiec_uc(I, a):
    """Unconditional coverage: is the breach rate equal to a?"""
    x, n = int(np.sum(I)), len(I)
    p_hat = x / n
    lr = -2 * ((xlogy(n - x, 1 - a) + xlogy(x, a)) - (xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)))
    return lr, chi2.sf(lr, 1)


def christoffersen_ind(I):
    """Independence: does a breach today change the chance of a breach tomorrow?
    Returns LR_ind, p-value and the number of back-to-back breaches (n11)."""
    i = np.asarray(I, dtype=int)
    t0, t1 = i[:-1], i[1:]
    n00 = np.sum((t0 == 0) & (t1 == 0)); n01 = np.sum((t0 == 0) & (t1 == 1))
    n10 = np.sum((t0 == 1) & (t1 == 0)); n11 = np.sum((t0 == 1) & (t1 == 1))
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = 2 * (ll1 - ll0)
    return lr, chi2.sf(lr, 1), int(n11)


def backtest(I, a):
    lr_uc, p_uc = kupiec_uc(I, a)
    lr_ind, p_ind, n11 = christoffersen_ind(I)
    lr_cc = lr_uc + lr_ind
    return {"expected": a * len(I), "breaches": int(np.sum(I)), "rate %": 100 * np.mean(I),
            "back-to-back": n11, "LR_uc": lr_uc, "p_uc": p_uc, "LR_ind": lr_ind, "p_ind": p_ind,
            "LR_cc": lr_cc, "p_cc": chi2.sf(lr_cc, 2)}


def block_bootstrap_ci(d, L, B, rng, q=(2.5, 97.5)):
    """Moving-block bootstrap percentile CI for the mean of d (blocks of length L)."""
    d = np.asarray(d)
    n = len(d)
    nb = int(np.ceil(n / L))
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - L + 1, size=nb)
        means[b] = d[(starts[:, None] + np.arange(L)).ravel()[:n]].mean()
    return np.percentile(means, q)
