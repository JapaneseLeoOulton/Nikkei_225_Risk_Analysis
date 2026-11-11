"""Build report/report.pdf with LaTeX.

Fills every <<name>> placeholder in report/report_template.tex from results/*.csv,
writes report/report.tex and compiles it with pdflatex (any TeX distribution;
TinyTeX is enough). Run `python run_all.py` first.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
DOWNLOAD_DATE = "30 October 2026"


def f(x, nd=3):
    return f"{x:.{nd}f}"


def pval(p, bold=True):
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    s = s.replace("<", r"$<$")
    return rf"\textbf{{{s}}}" if (bold and p < 0.05) else s


def values():
    desc = pd.read_csv(RES / "descriptives.csv", index_col=0)
    lb = pd.read_csv(RES / "ljung_box_raw.csv", index_col=0)
    ols = pd.read_csv(RES / "ols_walk_forward.csv", index_col=0)
    buckets = pd.read_csv(RES / "merton_logscore_buckets.csv", index_col=0)
    boot = pd.read_csv(RES / "merton_logscore_bootstrap.csv", index_col=0)
    garch = pd.read_csv(RES / "garch_models.csv", index_col=0)
    own = pd.read_csv(RES / "garch_own_vs_arch.csv", index_col=0)
    btab = pd.read_csv(RES / "backtests_all_models.csv", index_col=[0, 1])
    per_year = pd.read_csv(RES / "breaches_per_year_99.csv", index_col=0)
    dens = pd.read_csv(RES / "density_fit.csv", index_col=0)

    g, o, m = garch.loc["GJR-t"], own.loc["own MLE"], dens.loc["Merton"]
    bt = lambda model, lv: btab.loc[(model, lv)]
    share = lambda b: f"{abs(buckets.loc[b, 'share of total d']) * 100:.0f}"

    rows = []
    for model in ["Gaussian", "Merton", "GJR-t"]:
        for lv in ["95.0%", "99.0%", "99.5%"]:
            b = bt(model, lv)
            name = model.replace("GJR-t", "GJR-$t$") if lv == "95.0%" else ""
            rows.append(f"{name} & {lv[:-1].rstrip('0').rstrip('.')}\\% & {int(b['breaches'])} ({b['expected']:.0f}) & "
                        f"{pval(b['p_uc'])} & {pval(b['p_ind'])} & {pval(b['p_cc'])} \\\\")
        if model != "GJR-t":
            rows.append(r"\addlinespace")

    v = {
        "download_date": DOWNLOAD_DATE,
        "n_full": f"{int(desc.loc['full', 'n']):,}", "n_tr": f"{int(desc.loc['train', 'n']):,}",
        "n_te": f"{int(desc.loc['test', 'n']):,}",
        "kurt_full": f(desc.loc["full", "kurtosis"], 1), "skew_full": f(desc.loc["full", "skew"], 2),
        "lb_p5": f(lb.loc[5, "p r"], 2), "lb_p10": f(lb.loc[10, "p r"], 2), "lb_q5": f"{lb.loc[5, 'Q r^2']:,.0f}",
        "g_omega": f(o["omega"], 4), "g_alpha": f(o["alpha"], 4), "g_beta": f(o["beta"], 4), "g_ll": f(o["loglik"], 2),
        "j_omega": f(g["omega"]), "j_alpha": f(g["alpha"]), "j_gamma": f(g["gamma"]), "j_beta": f(g["beta"]),
        "j_nu": f(g["nu"], 1), "j_nu0": f"{g['nu']:.0f}", "j_pers": f(g["persistence"]),
        "j_hl": f"{g['half-life (days)']:.0f}",
        "ols_r2_zero": f(ols["R2_oos vs zero"].max(), 4), "ols_r2_mean": f(ols["R2_oos vs mean"].max(), 4),
        "ols_tmax": f(ols[["t vs zero", "t vs mean"]].abs().values.max() + 0.005, 2),
        "m_lr": f"{m['LR vs Gaussian']:.0f}", "m_sigma": f(m["sigma"], 2), "m_lam": f(m["lam"]),
        "m_m": f(m["m"], 2), "m_delta": f(m["delta"], 2),
        "d_mean": f(buckets["sum d"].sum() / buckets["days"].sum()), "d_lo": f(boot["lo"].min()),
        "d_hi": f(boot["hi"].max()),
        "b1_days": f"{int(buckets.loc['<1%', 'days']):,}", "b1_share": share("<1%"), "b3_share": share("2-3%"),
        "b4_days": f"{int(buckets.loc['>=3%', 'days'])}", "b4_share": share(">=3%"),
        "table_rows": "\n".join(rows),
        "G99_x": f"{int(bt('Gaussian', '99.0%')['breaches'])}", "G99_e": f"{bt('Gaussian', '99.0%')['expected']:.0f}",
        "G99_pind": f(bt("Gaussian", "99.0%")["p_ind"], 4),
        "M99_x": f"{int(bt('Merton', '99.0%')['breaches'])}", "M995_x": f"{int(bt('Merton', '99.5%')['breaches'])}",
        "lbz_min": f(g[[c for c in garch.columns if c.startswith("LB z^2")]].min(), 2),
        "J99_pind": f(bt("GJR-t", "99.0%")["p_ind"], 2),
        "J99_x": f"{int(bt('GJR-t', '99.0%')['breaches'])}", "J995_x": f"{int(bt('GJR-t', '99.5%')['breaches'])}",
        "J99_e": f"{bt('GJR-t', '99.0%')['expected']:.0f}", "J995_e": f"{bt('GJR-t', '99.5%')['expected']:.0f}",
        "kurt_tr": f(desc.loc["train", "kurtosis"], 1), "kurt_z": f(g["kurtosis z"], 1),
        "kurt_t": f(3 * (g["nu"] - 2) / (g["nu"] - 4), 1),
        "y2015": f"{int(per_year.loc[2015, 'GJR-t'])}", "y2020": f"{int(per_year.loc[2020, 'GJR-t'])}",
        "y2024": f"{int(per_year.loc[2024, 'GJR-t'])}", "y_exp": f(per_year.loc[2015, "expected"], 1),
        "sd_te": f(desc.loc["test", "sd"], 2), "sd_tr": f(desc.loc["train", "sd"], 2),
        "mean_te": f"{desc.loc['test', 'mean']:+.3f}", "mean_tr": f"{desc.loc['train', 'mean']:+.3f}",
        "GT99_x": f"{int(bt('GARCH-t', '99.0%')['breaches'])}", "GT99_pcc": f(bt("GARCH-t", "99.0%")["p_cc"]),
    }
    return v


def main():
    tex = (HERE / "report_template.tex").read_text()
    v = values()
    tex = re.sub(r"<<(\w+)>>", lambda mt: v[mt.group(1)], tex)
    assert "<<" not in tex, "unfilled placeholder"
    (HERE / "report.tex").write_text(tex)

    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise SystemExit("pdflatex not found: install a TeX distribution (e.g. TinyTeX) or compile report/report.tex elsewhere")
    for _ in range(2):                       # second pass resolves the Figure/Table numbers
        res = subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
                             cwd=HERE, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:])
            raise SystemExit("pdflatex failed; see report/report.log")
    for ext in ("aux", "log", "out"):
        (HERE / f"report.{ext}").unlink(missing_ok=True)
    print(f"wrote {HERE / 'report.pdf'}")


if __name__ == "__main__":
    main()
