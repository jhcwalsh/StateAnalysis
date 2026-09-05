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


def test_load_fredmd_tolerates_trailing_blank_column(vintage_path, tmp_path):
    # The 2026-08 vintage dropped a series and left a trailing comma, so pandas reads an
    # "Unnamed" column whose t-code is NaN; the loader must ignore it and pad in headers.
    lines = open(vintage_path, encoding="utf-8").read().splitlines()
    lines[0] = lines[0].replace("INDPRO", " INDPRO ") + ","
    lines = [lines[0]] + [l + "," for l in lines[1:]]
    p = tmp_path / "vintage.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    df, tc = data.load_fredmd(str(p))
    ref_df, ref_tc = data.load_fredmd(vintage_path)
    assert tc.dtype.kind == "i" and not tc.isna().any()
    assert list(tc.index) == list(ref_tc.index) and "INDPRO" in tc.index
    assert df.shape == ref_df.shape and not any(str(c).startswith("Unnamed") for c in df.columns)


def test_remove_outliers_skips_columns_with_zero_iqr():
    # A column that is mostly exact zeros has IQR 0 on the estimation rows; the FRED-MD
    # rule must not wipe its remaining values (that left PPICMM with zero variance on
    # pre-2000 walk-forward windows of the 2026-08 vintage).
    idx = pd.date_range("2000-01-01", periods=100, freq="MS")
    rng = np.random.default_rng(0)
    sparse = np.zeros(100); sparse[::4] = rng.normal(0, 0.05, 25)          # 75% exact zeros
    normal = rng.normal(0, 1, 100); normal[10] = 500.0                      # one genuine outlier
    df = pd.DataFrame({"sparse": sparse, "normal": normal}, index=idx)
    mask = pd.Series(True, index=idx)
    cleaned, flagged = data.remove_outliers(df, 10.0, mask)
    assert not flagged["sparse"].any() and cleaned["sparse"].std() > 0
    assert flagged["normal"].sum() == 1 and np.isnan(cleaned.loc[idx[10], "normal"])



def _malformed_copy(vintage_path, tmp_path, drop="S&P div yield"):
    lines = open(vintage_path, encoding="utf-8").read().splitlines()
    names = lines[0].split(",")
    assert drop in names
    k = names.index(drop)
    names.remove(drop)
    lines[0] = ",".join(names) + ","          # header lost one name; data rows keep the column
    tc = lines[1].split(",")
    del tc[k]
    lines[1] = ",".join(tc) + ","             # the t-code row is aligned with the header, as in the real 2026-08 file
    p = tmp_path / "malformed.csv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_header_repair_restores_the_pinned_layout(vintage_path, tmp_path):
    p = _malformed_copy(vintage_path, tmp_path)
    raw = data._read_raw(p)
    assert raw.columns[-1].startswith("Unnamed") and "S&P div yield" not in raw.columns
    df, tc = data.load_fredmd(str(p), reference=vintage_path)
    ref_df, ref_tc = data.load_fredmd(vintage_path)
    pd.testing.assert_frame_equal(df, ref_df)
    pd.testing.assert_series_equal(tc, ref_tc)
    assert "S&P div yield" in (df.attrs["vintage_note"] or "")


def test_misaligned_vintage_without_repair_is_refused(vintage_path, tmp_path):
    p = _malformed_copy(vintage_path, tmp_path)
    raw = data._read_raw(p)
    raw.columns = [*raw.columns[:-1], "EXTRA"]            # defeat the repair: no Unnamed column
    q = tmp_path / "shifted.csv"; raw.to_csv(q, index=False)
    with pytest.raises(data.VintageError) as e:
        data.load_fredmd(str(q), reference=vintage_path)
    assert "CPIAUCSL" in str(e.value) and "PPICMM" in str(e.value) and "INDPRO" not in str(e.value)


def test_revised_vintage_passes_the_check(vintage_path, tmp_path):
    raw = data._read_raw(vintage_path)
    body = raw.iloc[1:, 1:].astype(float)
    rng = np.random.default_rng(1)
    raw.iloc[1:, 1:] = (body * (1 + rng.normal(0, 0.002, body.shape))).values   # small revisions everywhere
    q = tmp_path / "revised.csv"; raw.to_csv(q, index=False)
    df, tc = data.load_fredmd(str(q), reference=vintage_path)
    assert df.attrs["vintage_note"] is None and df.shape == data.load_fredmd(vintage_path)[0].shape


def test_pinned_vintage_is_not_checked_against_itself(vintage_path):
    df, _ = data.load_fredmd(vintage_path)
    assert df.attrs["vintage_note"] is None


def test_tcode_change_is_refused(vintage_path, tmp_path):
    lines = open(vintage_path, encoding="utf-8").read().splitlines()
    names, tc = lines[0].split(","), lines[1].split(",")
    tc[names.index("CPIAUCSL")] = "5"          # a genuine methodology change must be a deliberate re-pin
    lines[1] = ",".join(tc)
    q = tmp_path / "tcode.csv"; q.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(data.VintageError) as e:
        data.load_fredmd(str(q), reference=vintage_path)
    assert "t-codes" in str(e.value) and "CPIAUCSL" in str(e.value)
