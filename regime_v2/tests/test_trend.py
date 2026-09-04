import numpy as np
import pandas as pd
import pytest

from regime_v2 import trend


@pytest.fixture
def series():
    idx = pd.date_range("1960-01-01", periods=480, freq="MS")
    rng = np.random.default_rng(1)
    y = pd.Series(np.cumsum(rng.normal(0, 0.3, 480)) * 0.05 + rng.normal(0, 1, 480), index=idx)
    return y, pd.Series(True, index=idx)


@pytest.mark.parametrize("method", ["smoothed_trailing", "trailing_mean", "trailing_median", "hamilton", "onesided_hp"])
def test_trend_step_is_exactly_real_time(series, method):
    y, m = series
    full = trend.make_gap(y, method, m)
    cut = trend.make_gap(y.iloc[:360], method, m.iloc[:360])
    ov = cut.index
    d = (full.loc[ov, "gap"] - cut["gap"]).abs().dropna()
    assert len(d) > 100
    assert d.max() < 1e-10


def test_columns_and_defaults(series):
    y, m = series
    out = trend.make_gap(y, "smoothed_trailing", m)
    assert list(out.columns) == ["level", "trend", "gap_raw", "gap"]
    assert out["gap"].first_valid_index() == y.index[118]  # trend valid from i=59, 60 more for the expanding std


def test_expanding_std_ignores_masked_rows(series):
    y, m = series
    y2 = y.copy(); y2.iloc[300] = 50.0            # spike
    m2 = m.copy(); m2.iloc[300] = False
    sd_masked = trend.standardise_expanding(y2, m2, 60)
    sd_all = trend.standardise_expanding(y2, m, 60)
    # gap/sd: with the spike excluded from sd, later values keep their scale
    assert abs(sd_masked.iloc[-1]) > abs(sd_all.iloc[-1]) * 2
    assert not np.isnan(sd_masked.iloc[300])       # masked month still scored


def test_revision_stats_perfect_and_noisy():
    idx = pd.date_range("2000-01-01", periods=100, freq="MS")
    a = pd.Series(np.sin(np.arange(100) / 5 + 0.3), index=idx)   # phase shift: no exact zeros
    r = trend.revision_stats(a, a)
    assert r["corr_first_final"] == pytest.approx(1.0)
    assert r["noise_to_signal_rmse"] == pytest.approx(0.0)
    assert r["sign_agreement"] == 1.0 and r["n"] == 100
    r2 = trend.revision_stats(a, -a)
    assert r2["sign_agreement"] == 0.0


def test_expost_comparator_is_named_expost():
    assert trend.centred_trend_expost.__name__.endswith("_expost")
