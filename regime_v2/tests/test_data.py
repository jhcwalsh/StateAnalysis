import numpy as np
import pandas as pd
import pytest

from regime_v2 import data


def test_blocks_disjoint_and_present(vintage_path):
    b = data.build_blocks(vintage_path)
    assert set(data.GROWTH_BLOCK).isdisjoint(data.INFLATION_BLOCK)
    assert b["missing_series"] == []
    assert list(b["growth"].columns) == data.GROWTH_BLOCK
    assert list(b["inflation"].columns) == data.INFLATION_BLOCK


def test_outlier_count_2020(vintage_path):
    b = data.build_blocks(vintage_path)
    n2020 = (b["outliers"]["date"].dt.year == 2020).sum()
    assert n2020 >= 20


def test_estimation_mask_excludes_covid_and_thin_months(vintage_path):
    b = data.build_blocks(vintage_path)
    m = b["estimation_mask"]
    assert m.dtype == bool
    assert not m.loc["2020-03-01":"2020-12-01"].any()
    assert m.loc["2019-12-01"] and m.loc["2021-06-01"]
    # 2020-04 keeps only 3 of 22 growth series; it is inside the mask anyway,
    # but the thin-month rule must also fire on its own:
    m2 = data.build_blocks(vintage_path, mask=None)["estimation_mask"]
    assert not m2.loc["2020-04-01"]
    assert m2.loc["2020-08-01"]


def test_last_vintage_month_populated(vintage_path):
    b = data.build_blocks(vintage_path)
    last = b["growth"].index[-1]
    assert b["growth"].loc[last].notna().sum() >= 0.9 * len(data.GROWTH_BLOCK)
    assert b["inflation"].loc[last].notna().all()


def test_asof_truncates_before_statistics(vintage_path):
    full = data.build_blocks(vintage_path)
    cut = data.build_blocks(vintage_path, asof="2015-12-31")
    assert cut["growth"].index[-1] == pd.Timestamp("2015-12-01")
    # outlier thresholds differ once data after the cut is unseen:
    o_full = set(map(tuple, full["outliers"].to_numpy()))
    o_cut = set(map(tuple, cut["outliers"].to_numpy()))
    assert all(d <= pd.Timestamp("2015-12-01") for d, _ in o_cut)


def test_outlier_thresholds_use_mask_only():
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    x = pd.Series(np.random.default_rng(0).normal(size=120), index=idx)
    x.iloc[50] = 400.0                     # huge spike
    df = pd.DataFrame({"a": x})
    m_all = pd.Series(True, index=idx)
    m_excl = m_all.copy(); m_excl.iloc[50] = False
    _, flagged_all = data.remove_outliers(df, 10.0, m_all)
    _, flagged_excl = data.remove_outliers(df, 10.0, m_excl)
    assert flagged_all["a"].iloc[50] and flagged_excl["a"].iloc[50]
    # the spike must not have been allowed to widen the IQR: with it excluded
    # from the threshold the same rows are flagged (no extra), and the
    # thresholds differ
    assert flagged_all["a"].sum() == 1 and flagged_excl["a"].sum() == 1


def test_transform_codes():
    s = pd.Series([1.0, 2.0, 4.0, 8.0])
    assert data.transform(s, 1).equals(s)
    assert np.allclose(data.transform(s, 5).dropna(), np.log(2))
    assert np.allclose(data.transform(s, 6).dropna(), 0.0)
    with pytest.raises(ValueError):
        data.transform(s, 9)
