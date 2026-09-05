"""Stage 6 — portfolios and the achievable backtest (spec §6 Stage 6, D11).

One backtest loop serves every strategy. The label a strategy acts on is the
one strictly available before the month whose return it earns; the moments
it uses are estimated on returns up to the decision date, each paired with
its own strictly-available label. The `_expost` strategies are the only ones
allowed full-sample moments and smoothed labels, and they exist only to
measure look-ahead -- one per optimiser family, so the decomposition can be
read for the unconstrained long-short optimiser and for its long-only
counterpart on the same window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .assets import W6040, align_to_available, mixture_moments, moments_by_label
from .placebo import block_shuffle

STRATEGIES = ["PIT_MaxSharpe", "PIT_MinVar", "ProbWeighted_MaxSharpe", "Oracle_MaxSharpe", "Static_6040", "EqualWeight",
              "PIT_LongOnly_MaxSharpe", "PIT_RiskParity", "Oracle_LongOnly_MaxSharpe"]
EXPOST = ["InSample_MaxSharpe_expost", "InSample_LongOnly_expost"]

_VAR_FLOOR = 1e-12   # guards the sqrt in the Sharpe objective on a degenerate covariance


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
        # Degenerate: exact-zero-net solutions do not occur on real regime-conditional
        # covariances (0 hits across 4500+ mv_weights calls in the fixture test suite);
        # this branch exists to satisfy test_mv_weights_diagonal_cases' w4 case (a
        # contrived symmetric mu/Sigma). Scaling to leverage_cap rather than the
        # conservative gross=1 is what that test requires; gross=1 remains the fallback
        # should this ever need to be more conservative in production.
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


def longonly_weights(mu: pd.Series, Sigma: pd.DataFrame, rf: float = 0.0) -> tuple[pd.Series, dict]:
    """Long-only fully-invested max-Sharpe weights: max w'(mu-rf)/sqrt(w'Sw) s.t. w >= 0, sum w = 1.

    The constrained problem has no closed form, so it is solved numerically
    (SLSQP, Kraft 1988) from the equal-weight start, which is always feasible.
    A non-convergent solve returns the equal-weight vector with
    `converged=False`; the caller (backtest) counts that and uses 60/40 for
    the month rather than trusting a half-solved direction.
    """
    idx = mu.index
    n = len(idx)
    Sig = Sigma.reindex(index=idx, columns=idx).to_numpy()
    flags = {"converged": False, "rank_deficient": bool(np.linalg.matrix_rank(Sig) < n)}
    excess = (mu - rf).to_numpy()
    weq = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        return -float(w @ excess) / np.sqrt(max(float(w @ Sig @ w), _VAR_FLOOR))

    res = minimize(neg_sharpe, weq, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   options={"maxiter": 200, "ftol": 1e-10})
    if not res.success:
        return pd.Series(weq, index=idx), flags
    flags["converged"] = True
    # SLSQP satisfies the bounds and the budget to its own tolerance only; clip and
    # renormalise so the published weights are exactly long-only and exactly fully invested.
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    s = w.sum()
    return pd.Series(w / s if s > 1e-12 else weq, index=idx), flags


def risk_parity_weights(Sigma: pd.DataFrame, tol: float = 1e-8, max_iter: int = 10_000) -> tuple[pd.Series, dict]:
    """Equal risk contribution weights (Maillard, Roncalli and Teiletche, 2010).

    Long-only, sum w = 1, every asset's risk contribution w_i (Sw)_i / (w'Sw)
    equal to 1/n. Solved by cyclical coordinate descent on the equivalent
    unconstrained problem min 0.5 y'Sy - (1/n) sum log y_i (Griveau-Billion,
    Richard and Roncalli, 2013), whose solution is positive by construction and
    scales to the ERC portfolio; each coordinate update is the positive root of
    S_ii y_i^2 + (S_i.y - S_ii y_i) y_i - 1/n = 0. No start point can be
    infeasible, so the only failure modes are iteration exhaustion and a
    non-positive variance on the diagonal; either returns the equal-weight
    vector with `converged=False`, and the backtest then falls back to 60/40 for
    that month and counts it. A rank-deficient Sigma is flagged and given a
    small ridge so the recursion stays well posed.
    """
    idx = Sigma.index
    n = len(idx)
    Sig = Sigma.reindex(index=idx, columns=idx).to_numpy(dtype=float)
    Sig = 0.5 * (Sig + Sig.T)
    flags = {"converged": False, "rank_deficient": bool(np.linalg.matrix_rank(Sig) < n)}
    if flags["rank_deficient"]:
        Sig = Sig + np.eye(n) * (1e-10 * np.trace(Sig) / n)
    diag = np.diag(Sig).copy()
    weq = np.full(n, 1.0 / n)
    if (diag <= 0).any():
        # A zero or negative variance is a corrupted covariance, not a usable degenerate
        # one: the asset has no risk contribution to equalise, and skipping its coordinate
        # would leave it at its start value and hand it a real allocation anyway. Report a
        # non-convergence instead, so the caller falls back to 60/40 and counts the month.
        return pd.Series(weq, index=idx), flags
    target = 1.0 / n
    y = weq.copy()
    for _ in range(max_iter):
        y_prev = y.copy()
        for i in range(n):
            a = diag[i]
            b = float(Sig[i] @ y) - a * y[i]
            y[i] = (-b + np.sqrt(b * b + 4.0 * a * target)) / (2.0 * a)
        if np.max(np.abs(y - y_prev)) < tol:
            flags["converged"] = True
            break
    s = y.sum()
    return pd.Series(y / s if s > 1e-12 else weq, index=idx), flags


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


def backtest(returns: pd.DataFrame, labels_frame: pd.DataFrame, probs_rt: pd.DataFrame, start: str = "2010-01-01",
             min_regime_obs: int = 15, cost_bp: float = 0.0, leverage_cap: float = 3.0,
             strategies: list[str] | None = None, include_expost: bool = True) -> BacktestResult:
    strategies = list(strategies or STRATEGIES) + (list(EXPOST) if include_expost else [])
    assets = list(returns.columns)
    pit = align_to_available(returns, labels_frame, "hmm_walkforward", strict=True)
    orc = align_to_available(returns, labels_frame, "hmm_smoothed_expost", strict=True)
    full_mu, full_cov = moments_by_label(orc, min_regime_obs) if include_expost else ({}, {})
    w6040 = pd.Series(W6040).reindex(assets).fillna(0.0)
    weq = pd.Series(1.0 / len(assets), index=assets)
    months = [m for m in pit.index if m >= pd.Timestamp(start) and m in orc.index]
    counters = {"pit_maxsharpe_fallback": 0, "pit_minvar_fallback": 0, "oracle_fallback": 0, "pw_fallback": 0,
                "insample_fallback": 0, "pit_longonly_fallback": 0, "pit_riskparity_fallback": 0,
                "oracle_longonly_fallback": 0, "insample_longonly_fallback": 0,
                "longonly_nonconverged": 0, "riskparity_nonconverged": 0, "negsum": 0, "rank_deficient": 0}
    rets = {s: {} for s in strategies}
    wts = {s: {} for s in strategies}
    turn = {s: {} for s in strategies}
    prev = {s: pd.Series(0.0, index=assets) for s in strategies}

    mom_cache = {}          # (label source, regime) -> (mu, Sigma) for the current month only

    def regime_moments(source, hist, current):
        """Expanding regime moments for one month, computed once per (label source, regime).

        Four strategies read the same slice of the same history in the same month
        (max-Sharpe, min-variance, long-only and risk parity on the point-in-time
        labels, and the oracle pair on the smoothed ones). The covariance is the
        expensive part, so the slice is taken and its moments computed once per
        month and shared. Below `min_regime_obs` no moments are computed at all:
        the caller falls back to 60/40.
        """
        key = (source, current)
        if key not in mom_cache:
            sub = hist[hist["label"] == current].drop(columns=["label", "label_date"])
            mom_cache[key] = ((sub.mean() * 12, sub.cov() * 12) if len(sub) >= min_regime_obs
                              else (None, None))
        return mom_cache[key]

    def regime_weights(source, hist, current, objective, key):
        mu, Sig = regime_moments(source, hist, current)
        if mu is None:
            counters[key] += 1
            return None
        w, flags = mv_weights(mu, Sig, objective, leverage_cap=leverage_cap)
        counters["negsum"] += int(flags["negsum"]); counters["rank_deficient"] += int(flags["rank_deficient"])
        return w

    def constrained(w, flags, nonconverged_key):
        """Count a constrained optimiser's flags; None (-> 60/40) if it did not converge."""
        counters["rank_deficient"] += int(flags["rank_deficient"])
        if not flags["converged"]:
            counters[nonconverged_key] += 1
            return None
        return w

    def regime_constrained(source, hist, current, kind, key):
        mu, Sig = regime_moments(source, hist, current)
        if mu is None:
            counters[key] += 1
            return None
        if kind == "longonly":
            return constrained(*longonly_weights(mu, Sig), "longonly_nonconverged")
        return constrained(*risk_parity_weights(Sig), "riskparity_nonconverged")

    for r in months:
        mom_cache.clear()                               # moments expand with the decision date
        d = r - pd.DateOffset(months=1)                 # decision at the end of d
        hist_pit, hist_orc = pit.loc[:d], orc.loc[:d]   # returns <= d, each with its strictly-available label
        cur_pit, cur_orc = pit.loc[r, "label"], orc.loc[r, "label"]
        label_date = pit.loc[r, "label_date"]
        row = returns.loc[r].reindex(assets)
        for s in strategies:
            w = None
            if s == "PIT_MaxSharpe":
                w = regime_weights("pit", hist_pit, cur_pit, "max_sharpe", "pit_maxsharpe_fallback")
            elif s == "PIT_MinVar":
                w = regime_weights("pit", hist_pit, cur_pit, "min_var", "pit_minvar_fallback")
            elif s == "Oracle_MaxSharpe":
                w = regime_weights("orc", hist_orc, cur_orc, "max_sharpe", "oracle_fallback")
            elif s == "ProbWeighted_MaxSharpe":
                mu, cov = moments_by_label(hist_pit, min_regime_obs)
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
            elif s == "PIT_LongOnly_MaxSharpe":
                w = regime_constrained("pit", hist_pit, cur_pit, "longonly", "pit_longonly_fallback")
            elif s == "PIT_RiskParity":
                w = regime_constrained("pit", hist_pit, cur_pit, "riskparity", "pit_riskparity_fallback")
            elif s == "Oracle_LongOnly_MaxSharpe":
                w = regime_constrained("orc", hist_orc, cur_orc, "longonly", "oracle_longonly_fallback")
            elif s == "InSample_LongOnly_expost":
                if cur_orc in full_mu:
                    # Flags discarded, exactly as InSample_MaxSharpe_expost discards mv_weights',
                    # so the published counters treat the two ex-post comparators alike.
                    w, _ = longonly_weights(full_mu[cur_orc], full_cov[cur_orc])
                else:
                    counters["insample_longonly_fallback"] += 1
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


LOOKAHEAD_ROWS = {
    "unconstrained": ("InSample_MaxSharpe_expost", "Oracle_MaxSharpe", "PIT_MaxSharpe"),
    "longonly": ("InSample_LongOnly_expost", "Oracle_LongOnly_MaxSharpe", "PIT_LongOnly_MaxSharpe"),
}


def lookahead_decomposition(perf: pd.DataFrame, family: str = "unconstrained") -> dict:
    """Sharpe by information set: in-sample (full moments, smoothed labels) -> oracle -> PIT.

    `family` picks the optimiser whose three rows are read; the two families
    share a window and a label timing, so their decompositions are comparable
    term by term.
    """
    if family not in LOOKAHEAD_ROWS:
        raise ValueError(f"unknown family {family!r}")
    ins, orc, pit = (float(perf.loc[k, "sharpe"]) for k in LOOKAHEAD_ROWS[family])
    return {"insample_sharpe": ins, "oracle_sharpe": orc, "pit_sharpe": pit,
            "moment_lookahead": ins - orc, "label_lookahead": orc - pit, "total": ins - pit}


def backtest_placebo(returns: pd.DataFrame, labels_frame: pd.DataFrame, probs_rt: pd.DataFrame,
                     n: int = 200, seed: int = 0, **kw) -> dict:
    """PIT max-Sharpe Sharpe of the real labels vs. run-preserving shuffles of the walk-forward label."""
    rng = np.random.default_rng(seed)
    base = labels_frame.dropna(subset=["hmm_walkforward"])

    def sharpe_for(lab_series: pd.Series) -> float:
        lf = base.copy()
        lf["hmm_walkforward"] = lab_series.reindex(lf.index)
        bt = backtest(returns, lf, probs_rt, strategies=["PIT_MaxSharpe"], include_expost=False, **kw)
        return float(bt.perf.loc["PIT_MaxSharpe", "sharpe"])

    real = sharpe_for(base["hmm_walkforward"])
    null = np.array([sharpe_for(block_shuffle(base["hmm_walkforward"], rng)) for _ in range(n)])
    return {"real": real, "null": null, "percentile": float((null <= real).mean() * 100.0), "n": n}
