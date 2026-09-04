import numpy as np
import pandas as pd
import pytest

from regime_v2 import assets as A, portfolio as P
from regime_v2.regimes import REGIMES


def _frame(idx, seq):
    lab = pd.DataFrame(index=pd.DatetimeIndex(idx, name="date"))
    lab["available_at"] = lab.index + pd.DateOffset(months=1)
    lab["hmm_walkforward"] = seq
    lab["hmm_smoothed_expost"] = seq
    return lab


def _planted(n=240, seed=0, premium=0.04):
    """Asset 0 pays +premium in Goldilocks and -premium otherwise, seen through the strict lag."""
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    seq = (["Goldilocks"] * 24 + ["Contraction"] * 24) * (n // 48)
    lab = _frame(idx, seq)
    r = pd.DataFrame(rng.normal(0.003, 0.02, (n, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    known = lab["hmm_walkforward"].shift(2).reindex(idx)          # label r-2 is what a trader knows at r
    r["Equity_US"] += np.where(known == "Goldilocks", premium, -premium)
    r["US_Long_Treasury"] -= np.where(known == "Goldilocks", premium, -premium)
    probs = pd.DataFrame(0.0, index=idx, columns=REGIMES)
    for reg in ("Goldilocks", "Contraction"):
        probs.loc[lab["hmm_walkforward"] == reg, reg] = 1.0
    return r, lab, probs


def test_mv_weights_diagonal_cases():
    mu = pd.Series([0.10, 0.05], index=["a", "b"])
    S = pd.DataFrame(np.diag([0.04, 0.01]), index=mu.index, columns=mu.index)
    w, flags = P.mv_weights(mu, S, "max_sharpe")
    assert np.isclose(w.sum(), 1.0) and np.isclose(w["a"] / w["b"], (0.10 / 0.04) / (0.05 / 0.01))
    assert not flags["negsum"] and not flags["rank_deficient"]
    w2, _ = P.mv_weights(mu, S, "min_var")
    assert np.isclose(w2["a"] / w2["b"], 0.25)
    w3, f3 = P.mv_weights(-mu, S, "max_sharpe")
    assert f3["negsum"] and np.isclose(w3.sum(), -1.0)
    big = pd.Series([0.5, -0.5], index=["a", "b"]); S2 = pd.DataFrame(np.diag([0.001, 0.001]), index=big.index, columns=big.index)
    w4, _ = P.mv_weights(big, S2, "max_sharpe", leverage_cap=3.0)
    assert np.isclose(w4.abs().sum(), 3.0)
    with pytest.raises(ValueError):
        P.mv_weights(mu, S, "sortino")


def test_backtest_shapes_and_weight_sums():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=0.0)
    assert list(bt.returns.columns) == P.STRATEGIES + P.EXPOST
    assert bt.returns.index[0] >= pd.Timestamp("2005-01-01") and len(bt.returns) > 150
    assert list(bt.perf.columns) == ["ann_ret", "ann_vol", "sharpe", "maxdd", "turnover"]
    for s in ("PIT_MaxSharpe", "PIT_MinVar", "Oracle_MaxSharpe", "ProbWeighted_MaxSharpe"):
        sums = bt.weights[s].sum(axis=1)
        assert np.allclose(sums.abs(), 1.0)
    assert np.allclose(bt.weights["Static_6040"].sum(axis=1), 1.0) and np.allclose(bt.weights["EqualWeight"].sum(axis=1), 1.0)
    assert bt.turnover.loc[bt.turnover.index[1:], "Static_6040"].eq(0).all()


def test_backtest_uses_only_strictly_available_labels():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01")
    # the planted premium is keyed to label r-2, so PIT beats 60/40 clearly
    assert bt.perf.loc["PIT_MaxSharpe", "sharpe"] > bt.perf.loc["Static_6040", "sharpe"] + 0.5
    # oracle uses the same (synthetic) labels with the same timing -> identical path
    assert np.allclose(bt.returns["Oracle_MaxSharpe"], bt.returns["PIT_MaxSharpe"])
    # a label that only becomes available three months late must not help
    late = lab.copy(); late["available_at"] = late.index + pd.DateOffset(months=3)
    bt2 = P.backtest(r, late, probs, start="2005-01-01")
    assert bt2.perf.loc["PIT_MaxSharpe", "sharpe"] < bt.perf.loc["PIT_MaxSharpe", "sharpe"] - 0.5


def test_backtest_fallback_and_costs():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01", min_regime_obs=10_000)
    assert bt.counters["pit_fallback"] > 0
    assert np.allclose(bt.returns["PIT_MaxSharpe"].iloc[1:], bt.returns["Static_6040"].iloc[1:])
    b0 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=0.0)
    b10 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=10.0)
    diff = b0.returns - b10.returns
    assert np.allclose(diff, 0.001 * b0.turnover)


def test_probweighted_reduces_to_pit_under_one_hot():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01")
    assert np.allclose(bt.returns["ProbWeighted_MaxSharpe"], bt.returns["PIT_MaxSharpe"])
