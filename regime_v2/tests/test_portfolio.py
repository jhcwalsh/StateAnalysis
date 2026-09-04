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
    assert bt.counters["pit_maxsharpe_fallback"] > 0 and bt.counters["pit_minvar_fallback"] > 0
    assert "pit_fallback" not in bt.counters          # split per strategy, never shared
    assert np.allclose(bt.returns["PIT_MaxSharpe"].iloc[1:], bt.returns["Static_6040"].iloc[1:])
    b0 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=0.0)
    b10 = P.backtest(r, lab, probs, start="2005-01-01", cost_bp=10.0)
    diff = b0.returns - b10.returns
    assert np.allclose(diff, 0.001 * b0.turnover)


def test_probweighted_reduces_to_pit_under_one_hot():
    r, lab, probs = _planted()
    bt = P.backtest(r, lab, probs, start="2005-01-01")
    assert np.allclose(bt.returns["ProbWeighted_MaxSharpe"], bt.returns["PIT_MaxSharpe"])


def _planted_diverging(n=240, seed=0, premium=0.04, corrupt_frac=0.3):
    """Like _planted, but hmm_walkforward and hmm_smoothed_expost genuinely
    differ: the planted premium and hmm_smoothed_expost both track the true
    regime; hmm_walkforward is the true regime with `corrupt_frac` of months
    seeded-randomly relabelled to the other regime. Oracle (reads
    hmm_smoothed_expost) and PIT (reads hmm_walkforward) then see different
    histories and different current regimes, so a bug that made Oracle read
    the walk-forward column would be caught by the Sharpe/weight asserts below.
    """
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    true_seq = pd.Series((["Goldilocks"] * 24 + ["Contraction"] * 24) * (n // 48), index=idx)
    other = {"Goldilocks": "Contraction", "Contraction": "Goldilocks"}
    flip = rng.random(n) < corrupt_frac
    corrupted_seq = pd.Series([other[v] if f else v for v, f in zip(true_seq, flip)], index=idx)
    lab = _frame(idx, true_seq.tolist())
    lab["hmm_walkforward"] = corrupted_seq.to_numpy()      # PIT sees this, corrupted
    lab["hmm_smoothed_expost"] = true_seq.to_numpy()       # Oracle sees this, true

    r = pd.DataFrame(rng.normal(0.003, 0.02, (n, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    known = true_seq.shift(2).reindex(idx)                 # premium keyed to the TRUE label, r-2 via strict lag
    r["Equity_US"] += np.where(known == "Goldilocks", premium, -premium)
    r["US_Long_Treasury"] -= np.where(known == "Goldilocks", premium, -premium)
    probs = pd.DataFrame(0.0, index=idx, columns=REGIMES)
    for reg in ("Goldilocks", "Contraction"):
        probs.loc[corrupted_seq == reg, reg] = 1.0
    return r, lab, probs, true_seq, corrupted_seq


def test_oracle_reads_smoothed_labels_not_walkforward():
    r, lab, probs, true_seq, corrupted_seq = _planted_diverging()
    assert (true_seq != corrupted_seq).sum() > 0          # corruption actually did something

    bt = P.backtest(r, lab, probs, start="2005-01-01")
    # (a) oracle sees the true regime and should beat PIT (which sees the corrupted one) clearly
    assert bt.perf.loc["Oracle_MaxSharpe", "sharpe"] > bt.perf.loc["PIT_MaxSharpe", "sharpe"] + 0.5
    # (b) the two strategies must actually diverge, not just coincidentally match
    wdiff = (bt.weights["Oracle_MaxSharpe"] - bt.weights["PIT_MaxSharpe"]).abs().sum(axis=1)
    assert (wdiff > 1e-8).any()

    # (c) counterfactual: this test is load-bearing only if collapsing the two label
    # columns back together collapses Oracle back onto PIT (same column -> same path)
    lab_same = lab.copy()
    lab_same["hmm_smoothed_expost"] = corrupted_seq.to_numpy()
    bt2 = P.backtest(r, lab_same, probs, start="2005-01-01")
    assert np.allclose(bt2.returns["Oracle_MaxSharpe"], bt2.returns["PIT_MaxSharpe"])
    assert np.isclose(bt2.perf.loc["Oracle_MaxSharpe", "sharpe"], bt2.perf.loc["PIT_MaxSharpe", "sharpe"])


def test_lookahead_decomposition_sums():
    perf = pd.DataFrame({"sharpe": {"InSample_MaxSharpe_expost": 2.0, "Oracle_MaxSharpe": 1.2, "PIT_MaxSharpe": 0.7}})
    d = P.lookahead_decomposition(perf)
    assert d["moment_lookahead"] == pytest.approx(0.8) and d["label_lookahead"] == pytest.approx(0.5)
    assert d["total"] == pytest.approx(d["moment_lookahead"] + d["label_lookahead"]) == pytest.approx(1.3)
    assert d["insample_sharpe"] == 2.0 and d["pit_sharpe"] == 0.7


def test_backtest_placebo_ranks_real_labels_high():
    r, lab, probs = _planted()
    out = P.backtest_placebo(r, lab, probs, n=30, seed=0, start="2005-01-01")
    assert set(out) >= {"real", "percentile", "null"} and len(out["null"]) == 30
    assert out["percentile"] >= 90.0
    again = P.backtest_placebo(r, lab, probs, n=30, seed=0, start="2005-01-01")
    assert np.allclose(out["null"], again["null"])
