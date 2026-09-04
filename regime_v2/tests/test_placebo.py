import numpy as np
import pandas as pd

from regime_v2 import placebo as P


def _labels():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    lab = ["Goldilocks"] * 40 + ["Overheating"] * 20 + ["Contraction"] * 10 + ["Goldilocks"] * 30 + ["Stagflation"] * 20
    return pd.Series(lab, index=idx)


def _run_multiset(s):
    runs = (s != s.shift()).cumsum()
    return sorted(s.groupby(runs).agg(["first", "size"]).itertuples(index=False, name=None))


def test_block_shuffle_preserves_labels_and_index():
    lab = _labels()
    out = P.block_shuffle(lab, np.random.default_rng(3))
    assert out.index.equals(lab.index)
    assert out.value_counts().sort_index().equals(lab.value_counts().sort_index())
    # pieces are moved whole: adjacent same-label pieces may merge, so the run
    # count can only fall and no run can be shorter than the shortest original
    assert len(_run_multiset(out)) <= len(_run_multiset(lab))
    assert min(n for _, n in _run_multiset(out)) >= min(n for _, n in _run_multiset(lab))
    assert not out.equals(lab)


def test_placebo_percentile_is_extreme_for_real_signal():
    lab = _labels()
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=120), index=lab.index)
    x[lab == "Contraction"] -= 3.0                      # real regime effect
    stat = lambda l: float(x[l == "Contraction"].mean())
    out = P.placebo(lab, stat, n=300, seed=1)
    assert out["null"].shape == (300,)
    # shuffles that leave the Contraction piece at its original offset reproduce
    # the real statistic exactly, so the null has a point mass there: use <=
    assert out["real"] <= np.percentile(out["null"], 10)
    assert out["percentile"] <= 10.0
    again = P.placebo(lab, stat, n=300, seed=1)
    assert np.allclose(out["null"], again["null"])


def test_block_bootstrap_shape_and_seed():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    df = pd.DataFrame({"a": np.random.default_rng(0).normal(size=120)}, index=idx)
    stat = lambda d: pd.Series({"mean": d["a"].mean(), "sd": d["a"].std()})
    out = P.block_bootstrap(df, stat, block=12, n=50, seed=2)
    assert out.shape == (50, 2) and list(out.columns) == ["mean", "sd"]
    assert out.equals(P.block_bootstrap(df, stat, block=12, n=50, seed=2))
    assert out["mean"].std() < df["a"].std()


def test_block_bootstrap_can_sample_every_row():
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    df = pd.DataFrame({"pos": np.arange(30)}, index=idx)
    seen = set()
    for draw in P.block_bootstrap(df, lambda d: pd.Series({"rows": tuple(d["pos"])}), block=12, n=200, seed=0)["rows"]:
        seen.update(draw)
    assert seen == set(range(30))
    single = P.block_bootstrap(df.iloc[:5], lambda d: pd.Series({"n": len(d)}), block=12, n=3, seed=0)
    assert (single["n"] == 5).all()
