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
