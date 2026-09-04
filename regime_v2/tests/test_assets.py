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
    # first month is the base; December is dropped (no observation in a later month proves it complete)
    assert r.index[0] == pd.Timestamp("2020-02-01") and r.index[-1] == pd.Timestamp("2020-11-01") and len(r) == 10
    jan_end, feb_end = px.loc["2020-01"].iloc[-1]["SPY"], px.loc["2020-02"].iloc[-1]["SPY"]
    assert np.isclose(r.loc["2020-02-01", "Equity_US"], feb_end / jan_end - 1)
    assert cache.exists() and pd.read_parquet(cache).equals(r)


def test_returns_to_monthly_drops_final_calendar_month():
    # The series ends mid-September, so September is a 10-day stub: August is the last
    # month the download can prove is complete.
    idx = pd.date_range("2025-07-01", "2025-09-10", freq="D")
    px = pd.DataFrame({"SPY": 100 + np.arange(len(idx), dtype=float)}, index=idx)
    rets = A.returns_to_monthly(px)
    assert rets.index[-1] == pd.Timestamp("2025-08-01")
    assert pd.Timestamp("2025-09-01") not in rets.index


def test_returns_to_monthly_keeps_month_with_a_later_observation():
    # 2025-05-30 (Friday) is the real last trading day of May 2025: 2025-05-31 is a
    # Saturday and Memorial Day closed the 26th. A last-business-day rule would call
    # May partial (BMonthEnd is 2025-06-02 for a Saturday date); the "is there an
    # observation in a later month?" rule keeps it, because June has observations.
    idx = pd.DatetimeIndex(list(pd.date_range("2025-03-03", "2025-05-30", freq="B"))
                           + list(pd.date_range("2025-06-02", "2025-06-06", freq="B")))
    px = pd.DataFrame({"SPY": 100 + np.arange(len(idx), dtype=float)}, index=idx)
    rets = A.returns_to_monthly(px)
    assert pd.Timestamp("2025-05-01") in rets.index
    assert rets.index[-1] == pd.Timestamp("2025-05-01")       # June is itself incomplete
    may_end, apr_end = px.loc["2025-05"].iloc[-1]["SPY"], px.loc["2025-04"].iloc[-1]["SPY"]
    assert np.isclose(rets.loc["2025-05-01", "SPY"], may_end / apr_end - 1)


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


def test_mixture_moments_one_hot_and_total_variance():
    r, lab = _panel()
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    one_hot = pd.Series({"Goldilocks": 1.0, "Contraction": 0.0})
    m, S = A.mixture_moments(mu, cov, one_hot)
    assert np.allclose(m, mu["Goldilocks"]) and np.allclose(S, cov["Goldilocks"])
    half = pd.Series({"Goldilocks": 0.5, "Contraction": 0.5})
    m2, S2 = A.mixture_moments(mu, cov, half)
    assert np.allclose(m2, 0.5 * (mu["Goldilocks"] + mu["Contraction"]))
    within = 0.5 * (cov["Goldilocks"] + cov["Contraction"])
    assert (np.diag(S2) >= np.diag(within) - 1e-12).all()      # between-regime term is PSD
    # regimes absent from the moments are dropped and probabilities renormalised
    m3, _ = A.mixture_moments(mu, cov, pd.Series({"Goldilocks": 0.25, "Stagflation": 0.75}))
    assert np.allclose(m3, mu["Goldilocks"])


def test_mixture_path_and_portfolio_returns():
    r, lab = _panel()
    mu, cov = A.regime_moments(r, lab, "hmm_walkforward")
    probs = pd.DataFrame({"Goldilocks": [1.0, 0.0], "Contraction": [0.0, 1.0]},
                         index=pd.DatetimeIndex(["2016-01-01", "2016-02-01"], name="date"))
    w = pd.Series(A.W6040).reindex(r.columns).fillna(0.0)
    path = A.mixture_path(mu, cov, probs, w)
    assert list(path.columns) == ["mu", "sigma"] and len(path) == 2
    assert np.isclose(path.iloc[0]["mu"], w @ mu["Goldilocks"])
    assert np.isclose(path.iloc[0]["sigma"], np.sqrt(w @ cov["Goldilocks"] @ w))
    pr = A.portfolio_returns(r, A.W6040)
    assert np.isclose(pr.iloc[0], 0.6 * r.iloc[0]["Equity_US"] + 0.4 * r.iloc[0]["US_Aggregate_Bonds"])


def test_growth_share_bounds_and_planted_signal():
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    rng = np.random.default_rng(2)
    g, p = pd.Series(rng.normal(size=300), index=idx), pd.Series(rng.normal(size=300), index=idx)
    r = 0.02 * g + 0.002 * p + rng.normal(0, 0.01, 300)
    out = A.growth_share_6040(pd.DataFrame({"r6040": r, "growth_gap": g, "inflation_gap": p}))
    assert set(out) == {"r2", "growth_share", "inflation_share", "n"}
    assert 0.0 <= out["growth_share"] <= 1.0 and out["growth_share"] > 0.9
    assert np.isclose(out["growth_share"] + out["inflation_share"], 1.0) and out["n"] == 300


def test_sharpe_spread_placebo_detects_planted_premium():
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    rng = np.random.default_rng(3)
    lab = pd.Series((["Goldilocks"] * 24 + ["Contraction"] * 24) * 5, index=idx)
    r = pd.Series(rng.normal(0.004, 0.02, 240), index=idx) + np.where(lab == "Contraction", -0.02, 0.01)
    out = A.sharpe_spread_placebo(pd.DataFrame({"label": lab, "r6040": r}), n=200, seed=0)
    assert set(out) >= {"real", "percentile", "null"} and out["percentile"] >= 95.0
