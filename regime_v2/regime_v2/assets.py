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


from .placebo import block_bootstrap  # noqa: E402
from .regimes import REGIMES  # noqa: E402


def align_to_available(returns: pd.DataFrame, labels: pd.DataFrame, col: str, strict: bool = False) -> pd.DataFrame:
    """Join each return month r to the latest label whose `available_at` <= r
    (`strict=True`: < r, i.e. the decision was made before r began).  D11.

    Returns a frame indexed by the return date with columns
    [label, label_date, *assets]; return months with no available label are dropped.
    """
    lab = labels[[col, "available_at"]].dropna(subset=[col]).copy()
    lab["label_date"] = lab.index
    lab = lab.sort_values("available_at").rename(columns={col: "label"})
    ret = returns.sort_index().reset_index().rename(columns={returns.index.name or "index": "date"})
    merged = pd.merge_asof(ret, lab[["available_at", "label", "label_date"]], left_on="date",
                           right_on="available_at", direction="backward", allow_exact_matches=not strict)
    merged = merged.dropna(subset=["label"]).set_index("date")
    cols = ["label", "label_date"] + list(returns.columns)
    return merged[cols]


def _stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-asset n, ann_ret, ann_vol, sharpe, maxdd, hit for one regime's months."""
    wealth = (1 + sub).cumprod()
    maxdd = (wealth / wealth.cummax() - 1).min()
    ann_ret, ann_vol = sub.mean() * 12, sub.std() * np.sqrt(12)
    return pd.DataFrame({"n": len(sub), "ann_ret": ann_ret, "ann_vol": ann_vol,
                         "sharpe": ann_ret / ann_vol, "maxdd": maxdd, "hit": (sub > 0).mean()})


def _table_from_aligned(al: pd.DataFrame) -> pd.DataFrame:
    parts = {}
    for reg, grp in al.groupby("label"):
        parts[reg] = _stats(grp.drop(columns=["label", "label_date"]))
    out = pd.concat(parts, names=["regime", "asset"]).swaplevel().sort_index()
    return out


def regime_conditional_table(returns: pd.DataFrame, labels: pd.DataFrame, col: str = "hmm_walkforward",
                             n_boot: int = 1000, block: int = 12, seed: int = 0) -> pd.DataFrame:
    al = align_to_available(returns, labels, col)
    base = _table_from_aligned(al)
    boot = block_bootstrap(al, lambda d: _table_from_aligned(d)[["ann_ret", "sharpe"]].stack(), block=block, n=n_boot, seed=seed)
    se = boot.std(ddof=1)
    base["se_ann_ret"] = se.xs("ann_ret", level=-1).reindex(base.index)
    base["se_sharpe"] = se.xs("sharpe", level=-1).reindex(base.index)
    return base[["n", "ann_ret", "ann_vol", "sharpe", "maxdd", "hit", "se_ann_ret", "se_sharpe"]]


def conditional_corr(returns: pd.DataFrame, labels: pd.DataFrame, col: str = "hmm_walkforward") -> dict:
    al = align_to_available(returns, labels, col)
    return {reg: grp.drop(columns=["label", "label_date"]).corr() for reg, grp in al.groupby("label")}


def regime_moments(returns: pd.DataFrame, labels: pd.DataFrame, col: str, strict: bool = False,
                   min_obs: int = 1) -> tuple[dict, dict]:
    """Annualised mean vector and covariance per regime from the aligned panel."""
    al = align_to_available(returns, labels, col, strict=strict)
    mu, cov = {}, {}
    for reg, grp in al.groupby("label"):
        sub = grp.drop(columns=["label", "label_date"])
        if len(sub) >= min_obs:
            mu[reg], cov[reg] = sub.mean() * 12, sub.cov() * 12
    return mu, cov
