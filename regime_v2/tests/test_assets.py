import numpy as np
import pandas as pd
import pytest

from regime_v2 import assets as A


def test_universe_and_6040():
    assert list(A.UNIVERSE) == ["SPY", "VEA", "EEM", "AGG", "TLT", "LQD", "HYG", "VNQ", "GLD", "DBC", "TIP"]
    assert A.UNIVERSE["DBC"] == "Commodities" and A.UNIVERSE["VEA"] == "Equity_DevelopedExUS"
    assert A.W6040 == {"Equity_US": 0.6, "US_Aggregate_Bonds": 0.4}


def test_fixture_shape_and_scale(returns_path):
    r = pd.read_parquet(returns_path)
    assert list(r.columns) == list(A.UNIVERSE.values())
    assert r.index.name == "date" and r.index.freqstr in ("MS", None) and r.index[0].day == 1
    assert r.index[0] <= pd.Timestamp("2007-08-01") and len(r) >= 200
    assert r.notna().all().all() and r.abs().max().max() < 0.6


def test_load_returns_uses_cache_without_network(returns_path, tmp_path):
    called = []
    r = A.load_returns(cache=returns_path, fetch=lambda tickers, start: called.append(1))
    assert not called and list(r.columns) == list(A.UNIVERSE.values())


def test_load_returns_converts_prices_and_writes_cache(tmp_path):
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for t in A.UNIVERSE}, index=idx)
    cache = tmp_path / "r.parquet"
    r = A.load_returns(cache=cache, fetch=lambda tickers, start: px[list(tickers)])
    assert list(r.columns) == list(A.UNIVERSE.values())
    assert r.index[0] == pd.Timestamp("2020-02-01") and len(r) == 11    # first month is the base
    jan_end, feb_end = px.loc["2020-01"].iloc[-1]["SPY"], px.loc["2020-02"].iloc[-1]["SPY"]
    assert np.isclose(r.loc["2020-02-01", "Equity_US"], feb_end / jan_end - 1)
    assert cache.exists() and pd.read_parquet(cache).equals(r)


def test_load_returns_reports_missing_ticker(tmp_path):
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    px = pd.DataFrame({t: 100.0 for t in list(A.UNIVERSE)[:-1]}, index=idx)
    with pytest.raises(ValueError, match="TIP"):
        A.load_returns(fetch=lambda tickers, start: px)


def _labels(idx, seq):
    lab = pd.DataFrame(index=pd.DatetimeIndex(idx, name="date"))
    lab["available_at"] = lab.index + pd.DateOffset(months=1)
    lab["hmm_walkforward"] = seq
    lab["hmm_smoothed_expost"] = seq
    return lab


def _panel(n=60, seed=0):
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.normal(0.005, 0.03, (n, 11)), index=idx, columns=list(A.UNIVERSE.values()))
    r.index.name = "date"
    lab = _labels(idx, (["Goldilocks"] * 30 + ["Contraction"] * 30))
    return r, lab


def test_align_uses_available_at_not_date():
    r, lab = _panel()
    al = A.align_to_available(r, lab, "hmm_walkforward")
    assert list(al.columns[:2]) == ["label", "label_date"]
    # return of month r meets the label of r-1 (available on the first day of r)
    assert (al["label_date"] == al.index - pd.DateOffset(months=1)).all()
    assert al.index[0] == pd.Timestamp("2015-02-01")               # January has no available label yet
    strict = A.align_to_available(r, lab, "hmm_walkforward", strict=True)
    assert (strict["label_date"] == strict.index - pd.DateOffset(months=2)).all()
    assert strict.index[0] == pd.Timestamp("2015-03-01")


def test_align_never_sees_the_future():
    r, lab = _panel()
    lab2 = lab.copy(); lab2["available_at"] = lab2.index + pd.DateOffset(months=3)
    al = A.align_to_available(r, lab2, "hmm_walkforward")
    assert ((al.index - al["label_date"]) >= pd.Timedelta(days=85)).all()


def test_conditional_table_counts_and_ses():
    r, lab = _panel()
    r["Equity_US"] += np.where(lab["hmm_walkforward"].shift(1).reindex(r.index) == "Goldilocks", 0.02, -0.02)
    t = A.regime_conditional_table(r, lab, n_boot=50)
    assert t.index.names == ["asset", "regime"]
    assert list(t.columns) == ["n", "ann_ret", "ann_vol", "sharpe", "maxdd", "hit", "se_ann_ret", "se_sharpe"]
    n = t.xs("Equity_US", level="asset")["n"]
    assert n.sum() == 59 and set(n.index) <= {"Goldilocks", "Contraction"}
    assert t.loc[("Equity_US", "Goldilocks"), "ann_ret"] > t.loc[("Equity_US", "Contraction"), "ann_ret"]
    assert (t["se_ann_ret"] > 0).all() and (t["se_sharpe"] > 0).all() and (t["maxdd"] <= 0).all()
    assert t.equals(A.regime_conditional_table(r, lab, n_boot=50))     # seed-stable


def test_conditional_corr_and_moments():
    r, lab = _panel()
    corr = A.conditional_corr(r, lab)
    assert set(corr) == {"Goldilocks", "Contraction"}
    assert np.allclose(np.diag(corr["Goldilocks"]), 1.0) and corr["Goldilocks"].shape == (11, 11)
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    assert set(mu) == {"Goldilocks", "Contraction"} and cov["Contraction"].shape == (11, 11)
    al = A.align_to_available(r, lab, "hmm_walkforward")
    sub = al[al["label"] == "Goldilocks"].drop(columns=["label", "label_date"])
    assert np.allclose(mu["Goldilocks"], sub.mean() * 12) and np.allclose(cov["Goldilocks"], sub.cov() * 12)
