"""Stage 5 — asset layer (spec §6 Stage 5, D11).

Returns are monthly simple total returns on the 11-ETF universe. Every join
to regime labels goes through `align_to_available`, which uses the labels'
`available_at` column and never the label date itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .placebo import block_bootstrap, placebo

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
    """Daily adjusted closes -> monthly simple returns indexed at month start.

    A calendar month is treated as complete only once the series carries at
    least one observation in a *later* month; the last calendar month present
    is therefore always dropped. A download run mid-month (e.g. on the 4th)
    would otherwise report a 2-3-trading-day stub as a completed month's return.

    The earlier `BMonthEnd` rule (drop the last month only if its last
    observation precedes the month's last business day) is holiday-blind: a
    month ending on a market holiday — Memorial Day, Good Friday — looks
    partial and would be dropped even when complete. A calendar-day tolerance
    has the opposite failure: it keeps months that are genuinely missing
    trading days. Requiring an observation in a later month is exact.
    """
    px = px.sort_index()
    monthly = px.resample("ME").last()
    if len(monthly):
        monthly = monthly.iloc[:-1]
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
    """Per-asset n, ann_ret, ann_vol, sharpe, maxdd, hit for one regime's months.

    `maxdd` is the drawdown of the within-regime months chained together (the
    regime's months compounded back-to-back), not a calendar drawdown.
    """
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
    """Regime-conditional per-asset stats; the SEs come from block-bootstrapping the whole
    aligned panel in calendar time, so regime membership is resampled along with the blocks."""
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


def moments_by_label(aligned: pd.DataFrame, min_obs: int = 1) -> tuple[dict, dict]:
    """Annualised mean vector and covariance per label from an already-aligned panel.

    `aligned` is the output of `align_to_available` (or a `.loc[:d]` slice of it):
    a frame with `label` / `label_date` columns and one column per asset. Labels
    with fewer than `min_obs` months are omitted from both dicts.
    """
    mu, cov = {}, {}
    for reg, grp in aligned.groupby("label"):
        sub = grp.drop(columns=["label", "label_date"])
        if len(sub) >= min_obs:
            mu[reg], cov[reg] = sub.mean() * 12, sub.cov() * 12
    return mu, cov


def regime_moments(returns: pd.DataFrame, labels: pd.DataFrame, col: str, strict: bool = False,
                   min_obs: int = 1) -> tuple[dict, dict]:
    """Annualised mean vector and covariance per regime from the aligned panel."""
    return moments_by_label(align_to_available(returns, labels, col, strict=strict), min_obs)


def mixture_moments(mu_by_regime: dict, cov_by_regime: dict, probs_t: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Law of total variance: mu = sum p_k mu_k; Sigma = sum p_k [Sigma_k + (mu_k - mu)(mu_k - mu)'].

    Regimes without moments are dropped and the remaining probabilities renormalised.
    """
    p = probs_t[[k for k in probs_t.index if k in mu_by_regime]].astype(float)
    if p.sum() <= 0:
        raise ValueError("no probability mass on regimes with estimated moments")
    p = p / p.sum()
    assets = next(iter(mu_by_regime.values())).index
    mu = sum(p[k] * mu_by_regime[k].reindex(assets) for k in p.index)
    Sigma = pd.DataFrame(0.0, index=assets, columns=assets)
    for k in p.index:
        d = (mu_by_regime[k].reindex(assets) - mu).to_numpy().reshape(-1, 1)
        Sigma += p[k] * (cov_by_regime[k].reindex(index=assets, columns=assets) + d @ d.T)
    return mu, Sigma


def mixture_path(mu_by_regime: dict, cov_by_regime: dict, probs: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Expected return and vol of a fixed-weight portfolio under the mixture, month by month."""
    rows = []
    for t, row in probs.iterrows():
        mu, S = mixture_moments(mu_by_regime, cov_by_regime, row)
        w = weights.reindex(mu.index).fillna(0.0)
        rows.append((t, float(w @ mu), float(np.sqrt(max(w @ S @ w, 0.0)))))
    out = pd.DataFrame(rows, columns=["date", "mu", "sigma"]).set_index("date")
    return out


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    return (returns @ w).rename("r_port")


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(1.0 - resid.var() / y.var())


def growth_share_6040(aligned: pd.DataFrame) -> dict:
    """OLS of the 60/40 return on both gaps; LMG (Shapley) split of R^2 between growth and inflation."""
    df = aligned[["r6040", "growth_gap", "inflation_gap"]].dropna()
    y, g, p = df["r6040"].to_numpy(), df["growth_gap"].to_numpy(), df["inflation_gap"].to_numpy()
    r2_full, r2_g, r2_p = _r2(y, np.column_stack([g, p])), _r2(y, g[:, None]), _r2(y, p[:, None])
    lmg_g = 0.5 * (r2_g + (r2_full - r2_p))
    lmg_p = 0.5 * (r2_p + (r2_full - r2_g))
    share_g = lmg_g / r2_full if r2_full > 0 else float("nan")
    return {"r2": r2_full, "growth_share": share_g, "inflation_share": 1.0 - share_g if r2_full > 0 else float("nan"),
            "n": int(len(df))}


def sharpe_spread_placebo(aligned: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Max-minus-min annualised Sharpe of r6040 across regimes vs. run-preserving label shuffles."""
    r = aligned["r6040"]

    def spread(labels: pd.Series) -> float:
        s = r.groupby(labels).agg(["mean", "std"])
        sh = (s["mean"] * 12) / (s["std"] * np.sqrt(12))
        return float(sh.max() - sh.min())

    return placebo(aligned["label"], spread, n=n, seed=seed)
