"""Stage 6 — portfolios and the achievable backtest (spec §6 Stage 6, D11).

One backtest loop serves every strategy. The label a strategy acts on is the
one strictly available before the month whose return it earns; the moments
it uses are estimated on returns up to the decision date, each paired with
its own strictly-available label. `InSample_MaxSharpe_expost` is the only
strategy allowed full-sample moments and smoothed labels, and it exists only
to measure look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assets import W6040, align_to_available, mixture_moments

STRATEGIES = ["PIT_MaxSharpe", "PIT_MinVar", "ProbWeighted_MaxSharpe", "Oracle_MaxSharpe", "Static_6040", "EqualWeight"]
EXPOST = ["InSample_MaxSharpe_expost"]


def mv_weights(mu: pd.Series, Sigma: pd.DataFrame, objective: str = "max_sharpe", rf: float = 0.0,
               leverage_cap: float = 3.0) -> tuple[pd.Series, dict]:
    """Unconstrained mean-variance weights (shorts allowed), scaled to sum to 1.

    Guards (from the notebook's mv_opt): a raw weight sum near zero or negative
    means no fully-invested max-Sharpe portfolio exists; the sign of the risk
    premium is preserved and `negsum` is flagged (weights then sum to -1).
    `rank_deficient` flags a singular covariance (pinv is used regardless).

    Capping leverage: a uniform scale-down of `w` cannot cap gross exposure
    (sum |w|) without also shrinking net exposure (sum w), since the two are
    linearly coupled for any scalar multiple of the same direction. So once
    net is fixed at +-1, an over-the-cap solution is capped by rescaling the
    long and short books separately to hit gross == leverage_cap while
    leaving net exactly where it was. In the degenerate near-zero-net case
    (no finite scalar reaches net = 1), there is no net to preserve, so the
    raw direction is instead scaled up to use the full leverage headroom.
    """
    if objective not in ("max_sharpe", "min_var"):
        raise ValueError(f"unknown objective {objective!r}")
    Sig = Sigma.reindex(index=mu.index, columns=mu.index).to_numpy()
    flags = {"negsum": False, "rank_deficient": bool(np.linalg.matrix_rank(Sig) < len(mu))}
    vec = (mu - rf).to_numpy() if objective == "max_sharpe" else np.ones(len(mu))
    w = np.linalg.pinv(Sig) @ vec
    s = w.sum()
    if abs(s) < 1e-12:
        flags["negsum"] = True
        gross0 = max(np.abs(w).sum(), 1e-12)
        w = w / gross0 * leverage_cap
    else:
        if s < 0:
            flags["negsum"] = True
        net = 1.0 if s > 0 else -1.0
        w = w / abs(s)
        gross = np.abs(w).sum()
        if gross > leverage_cap:
            long, short = np.clip(w, 0, None), np.clip(-w, 0, None)
            long_sum, short_sum = long.sum(), short.sum()
            long_target, short_target = (leverage_cap + net) / 2, (leverage_cap - net) / 2
            if long_sum > 1e-12:
                long = long * (long_target / long_sum)
            if short_sum > 1e-12:
                short = short * (short_target / short_sum)
            w = long - short
    return pd.Series(w, index=mu.index), flags


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: dict
    turnover: pd.DataFrame
    perf: pd.DataFrame
    counters: dict
    params: dict


def perf_table(rets: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    wealth = (1 + rets).cumprod()
    ann, vol = rets.mean() * 12, rets.std() * np.sqrt(12)
    return pd.DataFrame({"ann_ret": ann, "ann_vol": vol, "sharpe": ann / vol,
                         "maxdd": (wealth / wealth.cummax() - 1).min(), "turnover": turnover.mean()})


def _moments(hist: pd.DataFrame, min_obs: int) -> tuple[dict, dict]:
    mu, cov = {}, {}
    for reg, grp in hist.groupby("label"):
        sub = grp.drop(columns=["label", "label_date"])
        if len(sub) >= min_obs:
            mu[reg], cov[reg] = sub.mean() * 12, sub.cov() * 12
    return mu, cov


def backtest(returns: pd.DataFrame, labels_frame: pd.DataFrame, probs_rt: pd.DataFrame, start: str = "2010-01-01",
             min_regime_obs: int = 15, cost_bp: float = 0.0, leverage_cap: float = 3.0,
             strategies: list[str] | None = None, include_expost: bool = True) -> BacktestResult:
    strategies = list(strategies or STRATEGIES) + (list(EXPOST) if include_expost else [])
    assets = list(returns.columns)
    pit = align_to_available(returns, labels_frame, "hmm_walkforward", strict=True)
    orc = align_to_available(returns, labels_frame, "hmm_smoothed_expost", strict=True)
    full_mu, full_cov = _moments(orc, min_regime_obs) if include_expost else ({}, {})
    w6040 = pd.Series(W6040).reindex(assets).fillna(0.0)
    weq = pd.Series(1.0 / len(assets), index=assets)
    months = [m for m in pit.index if m >= pd.Timestamp(start) and m in orc.index]
    counters = {"pit_fallback": 0, "oracle_fallback": 0, "pw_fallback": 0, "insample_fallback": 0,
                "negsum": 0, "rank_deficient": 0}
    rets = {s: {} for s in strategies}
    wts = {s: {} for s in strategies}
    turn = {s: {} for s in strategies}
    prev = {s: pd.Series(0.0, index=assets) for s in strategies}

    def regime_weights(hist, current, objective, key):
        sub = hist[hist["label"] == current].drop(columns=["label", "label_date"])
        if len(sub) < min_regime_obs:
            counters[key] += 1
            return None
        w, flags = mv_weights(sub.mean() * 12, sub.cov() * 12, objective, leverage_cap=leverage_cap)
        counters["negsum"] += int(flags["negsum"]); counters["rank_deficient"] += int(flags["rank_deficient"])
        return w

    for r in months:
        d = r - pd.DateOffset(months=1)                 # decision at the end of d
        hist_pit, hist_orc = pit.loc[:d], orc.loc[:d]   # returns <= d, each with its strictly-available label
        cur_pit, cur_orc = pit.loc[r, "label"], orc.loc[r, "label"]
        label_date = pit.loc[r, "label_date"]
        row = returns.loc[r].reindex(assets)
        for s in strategies:
            w = None
            if s == "PIT_MaxSharpe":
                w = regime_weights(hist_pit, cur_pit, "max_sharpe", "pit_fallback")
            elif s == "PIT_MinVar":
                w = regime_weights(hist_pit, cur_pit, "min_var", "pit_fallback")
            elif s == "Oracle_MaxSharpe":
                w = regime_weights(hist_orc, cur_orc, "max_sharpe", "oracle_fallback")
            elif s == "ProbWeighted_MaxSharpe":
                mu, cov = _moments(hist_pit, min_regime_obs)
                p = probs_rt.loc[label_date] if label_date in probs_rt.index else None
                if p is None or not any(k in mu for k in p.index[p > 0]):
                    counters["pw_fallback"] += 1
                else:
                    m, S = mixture_moments(mu, cov, p)
                    w, flags = mv_weights(m, S, "max_sharpe", leverage_cap=leverage_cap)
                    counters["negsum"] += int(flags["negsum"]); counters["rank_deficient"] += int(flags["rank_deficient"])
            elif s == "InSample_MaxSharpe_expost":
                if cur_orc in full_mu:
                    w, _ = mv_weights(full_mu[cur_orc], full_cov[cur_orc], "max_sharpe", leverage_cap=leverage_cap)
                else:
                    counters["insample_fallback"] += 1
            elif s == "Static_6040":
                w = w6040
            elif s == "EqualWeight":
                w = weq
            else:
                raise ValueError(f"unknown strategy {s}")
            if w is None:
                w = w6040
            w = w.reindex(assets).fillna(0.0)
            to = float((w - prev[s]).abs().sum())
            rets[s][r] = float(w @ row) - cost_bp / 1e4 * to
            wts[s][r] = w
            turn[s][r] = to
            prev[s] = w
    R = pd.DataFrame(rets); R.index.name = "date"
    T = pd.DataFrame(turn); T.index.name = "date"
    W = {s: pd.DataFrame(wts[s]).T.rename_axis("date") for s in strategies}
    return BacktestResult(returns=R, weights=W, turnover=T, perf=perf_table(R, T), counters=counters,
                          params=dict(start=str(start), min_regime_obs=min_regime_obs, cost_bp=cost_bp,
                                      leverage_cap=leverage_cap))
