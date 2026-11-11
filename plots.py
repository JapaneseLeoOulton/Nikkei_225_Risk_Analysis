"""Figures. Static matplotlib, written to figures/ (no plt.show, so it runs headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Colours: fixed categorical order (blue, orange, aqua), one per model, same in every figure
COLORS = {"Gaussian": "#2a78d6", "Merton": "#eb6834", "GJR-t": "#1baf7a",
          "GARCH-N": "#1baf7a", "GARCH-t": "#1baf7a", "GJR-N": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
RETURN_GREY = "#b4b3ad"
EPISODES = [("2015-08-10", "2015-09-30", "China\ndevaluation"),
            ("2020-02-20", "2020-04-15", "COVID-19"),
            ("2024-07-25", "2024-08-20", "Yen carry\nunwind")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def result_figure(r_te, var99, breaches99, tests99, path):
    """Top: test-period returns with each model's 99% VaR line.
    Bottom: breach timeline, one row per model, with count and independence p-value."""
    models = list(var99)
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.08})
    dates = r_te.index

    for a in (ax, axb):
        for start, end, _ in EPISODES:
            a.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#f1efe8", lw=0, zorder=0)

    ax.plot(dates, r_te.values, color=RETURN_GREY, lw=0.45, label="Daily log return", zorder=1)
    for m in models:
        ax.plot(dates, -var99[m], color=COLORS[m], lw=1.6 if np.ptp(var99[m]) == 0 else 1.0,
                label=f"{m} 99% VaR", zorder=3 if m == models[-1] else 2)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.set_ylabel("Daily log return (%)")
    ax.set_ylim(-14.5, 10.5)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=8.5,
                    handlelength=1.8, borderaxespad=0.2)
    leg.get_lines()[0].set_linewidth(1.2)
    for start, end, label in EPISODES:
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, 7.6, label, ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.0)
    # direct labels for the two flat lines at the right edge
    for m in models:
        if np.ptp(var99[m]) == 0:
            ax.text(dates[-1] + pd.Timedelta(days=40), -var99[m][0], f"{m} {var99[m][0]:.2f}%",
                    va="center", fontsize=7.5, color=INK2)

    # breach timeline
    for row, m in enumerate(models):
        y = len(models) - 1 - row
        I = np.asarray(breaches99[m])
        axb.scatter(dates[I], np.full(I.sum(), y), marker="|", s=110, linewidths=1.6,
                    color=COLORS[m], zorder=3)
        t = tests99[m]
        axb.text(dates[-1] + pd.Timedelta(days=40), y,
                 f"{int(t['breaches'])} breaches (exp. {t['expected']:.0f})\nindep. p = {t['p_ind']:.3f}",
                 va="center", fontsize=7.5, color=INK2, linespacing=1.05)
    axb.set_yticks(range(len(models)))
    axb.set_yticklabels(models[::-1])
    axb.set_ylim(-0.6, len(models) - 0.4)
    axb.grid(axis="y", visible=False)
    axb.set_ylabel("99% breaches")
    axb.xaxis.set_major_locator(mdates.YearLocator(2))
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axb.set_xlim(dates[0] - pd.Timedelta(days=30), dates[-1] + pd.Timedelta(days=30))

    fig.suptitle("Out-of-sample 99% one-day VaR for the Nikkei 225, 2012-2026", x=0.125, ha="left",
                 fontsize=11.5, color=INK, y=0.985)
    fig.text(0.125, 0.945, "All parameters estimated on 2000-2011 and held fixed. GJR-t VaR moves because "
             "its variance is updated daily from returns up to the previous day.",
             fontsize=8.5, color=INK2, ha="left")
    fig.savefig(path)
    plt.close(fig)


def qq_figure(raw_std, z_n, z_c, nu_c, chosen, path):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    panels = [(raw_std, "norm", (), "Raw returns vs normal"),
              (z_n, "norm", (), "GARCH-N residuals vs normal"),
              (z_c / np.sqrt((nu_c - 2) / nu_c), stats.t, (nu_c,), f"{chosen} residuals vs t({nu_c:.1f})")]
    for ax, (x, dist, sp, title) in zip(axes, panels):
        (osm, osr), (slope, icpt, _) = stats.probplot(x, dist=dist, sparams=sp)
        ax.plot(osm, osm * slope + icpt, color=MUTED, lw=1)
        ax.scatter(osm, osr, s=6, color=COLORS["Gaussian"], lw=0)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Theoretical quantile")
    axes[0].set_ylabel("Sample quantile")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
