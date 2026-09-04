"""Stage 5 — asset layer (spec §6 Stage 5, D11).

Returns are monthly simple total returns on the 11-ETF universe. Every join
to regime labels goes through `align_to_available`, which uses the labels'
`available_at` column and never the label date itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE = {
    "SPY": "Equity_US", "VEA": "Equity_DevelopedExUS", "EEM": "Equity_EM",
    "AGG": "US_Aggregate_Bonds", "TLT": "US_Long_Treasury", "LQD": "Corp_IG", "HYG": "Corp_HY",
    "VNQ": "REITs", "GLD": "Gold", "DBC": "Commodities", "TIP": "TIPS",
}
W6040 = {"Equity_US": 0.6, "US_Aggregate_Bonds": 0.4}


def _download_yfinance(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px = px["Close"]
    return px


def returns_to_monthly(px: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted closes -> monthly simple returns indexed at month start."""
    monthly = px.sort_index().resample("ME").last()
    rets = monthly.pct_change().iloc[1:]
    rets.index = rets.index.to_period("M").to_timestamp()
    rets.index.name = "date"
    return rets


def load_returns(source: str = "yfinance", tickers: dict[str, str] | None = None, start: str = "2000-01-01",
                 cache: str | Path | None = None, refresh: bool = False, fetch=None) -> pd.DataFrame:
    """Monthly total returns for the universe; cached to parquet when `cache` is given."""
    tickers = dict(tickers or UNIVERSE)
    cache = Path(cache) if cache else None
    if cache is not None and cache.exists() and not refresh:
        return pd.read_parquet(cache)[list(tickers.values())]
    if source != "yfinance":
        raise ValueError(f"unknown return source {source!r}")
    px = (fetch or _download_yfinance)(list(tickers), start)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        raise ValueError(f"tickers missing from download: {missing}")
    rets = returns_to_monthly(px[list(tickers)].rename(columns=tickers))
    rets = rets.dropna(how="any")          # common history only (VEA binds at 2007-07)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        rets.to_parquet(cache)
    return rets
