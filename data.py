"""Data: download, load and split the Nikkei 225 series.

The analysis always runs off the committed snapshot data/n225.csv so results
are reproducible. `python data.py --refresh` re-downloads from Yahoo Finance and
overwrites that snapshot (results will then change, because new days are added).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = "^N225"
START = "2000-01-01"
SPLIT_DATE = "2012-01-01"          # train: before this date, test: on/after
CSV_PATH = Path(__file__).parent / "data" / "n225.csv"


def download(path=CSV_PATH, start=START):
    """Download daily N225 closes (auto-adjusted) and save with log returns."""
    import yfinance as yf

    df = yf.download(TICKER, start=start, auto_adjust=True, progress=False)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.to_csv(path)
    return df


def load(path=CSV_PATH):
    """Load the snapshot. Returns a DataFrame indexed by date with Close and Log_Return."""
    # yfinance writes a two-row header (Price / Ticker) plus a "Date" row
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df[["Close", "Log_Return"]]


def returns_pct(df):
    """Daily log returns in percent (x100), first NaN dropped.

    Percent scaling keeps the GARCH/Merton parameters at similar magnitudes,
    which the optimisers need (see Task 6 in the README)."""
    return df["Log_Return"].dropna() * 100


def split(r, split_date=SPLIT_DATE):
    return r[r.index < split_date], r[r.index >= split_date]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download and overwrite data/n225.csv")
    if ap.parse_args().refresh:
        download()
    r = returns_pct(load())
    tr, te = split(r)
    print(f"returns {r.index[0].date()} -> {r.index[-1].date()} ({len(r)} days)")
    print(f"train {len(tr)} days, test {len(te)} days")
