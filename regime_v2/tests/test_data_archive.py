"""data.load_archive_vintage: old FRED-MD vintages with renamed, missing or redefined series."""
import numpy as np
import pandas as pd
import pytest

from regime_v2 import data as D
from regime_v2.pipeline import run_pipeline


def _write(raw: pd.DataFrame, path) -> str:
    raw.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def raw(vintage_path):
    return D._read_raw(vintage_path)       # row 0 = t-codes, rows 1.. = data


def test_loads_the_pinned_vintage_unchanged_without_a_neighbour(vintage_path):
    levels, tcodes, report = D.load_archive_vintage(vintage_path)
    ref, ref_tc = D.load_fredmd(vintage_path)
    pd.testing.assert_frame_equal(levels, ref)
    pd.testing.assert_series_equal(tcodes, ref_tc)
    assert report == {"renamed": {}, "dropped_check": [], "dropped_tcode": [], "missing": [],
                      "n_growth": len(D.GROWTH_BLOCK), "n_inflation": len(D.INFLATION_BLOCK)}


def test_renames_the_old_ppi_names(raw, tmp_path):
    r = raw.rename(columns={"WPSFD49207": "PPIFGS", "WPSFD49502": "PPIFCG"})
    levels, tcodes, report = D.load_archive_vintage(_write(r, tmp_path / "v.csv"))
    assert "WPSFD49207" in levels.columns and "PPIFGS" not in levels.columns
    assert "WPSFD49502" in tcodes.index
    assert report["renamed"] == {"PPIFGS": "WPSFD49207", "PPIFCG": "WPSFD49502"}


def test_drops_the_nsa_cpi_stand_in_instead_of_substituting_it(raw, tmp_path):
    r = raw.drop(columns=["CUSR0000SA0L2"])
    r["CUUR0000SA0L2"] = raw["CUSR0000SA0L2"]          # same numbers under the NSA name
    levels, _, report = D.load_archive_vintage(_write(r, tmp_path / "v.csv"))
    assert "CUUR0000SA0L2" not in levels.columns and "CUSR0000SA0L2" not in levels.columns
    assert "CUSR0000SA0L2" in report["missing"]


def test_missing_block_series_are_reported_and_the_vintage_kept(raw, tmp_path):
    r = raw.drop(columns=["HWI", "USTPU"])
    levels, _, report = D.load_archive_vintage(_write(r, tmp_path / "v.csv"))
    assert sorted(report["missing"]) == ["HWI", "USTPU"]
    assert report["n_growth"] == len(D.GROWTH_BLOCK) - 2
    assert "INDPRO" in levels.columns


def test_a_redefined_series_is_dropped_against_the_neighbour_and_recorded(raw, vintage_path, tmp_path):
    r = raw.copy()
    rng = np.random.default_rng(0)
    vals = pd.to_numeric(r.loc[1:, "CUMFNS"], errors="coerce")
    r.loc[1:, "CUMFNS"] = (vals * (1 + rng.normal(0, 0.5, len(vals)))).to_numpy()   # corr with neighbour << 0.98
    levels, _, report = D.load_archive_vintage(_write(r, tmp_path / "v.csv"), neighbour=vintage_path)
    assert report["dropped_check"] == ["CUMFNS"] and "CUMFNS" not in levels.columns
    assert "INDPRO" in levels.columns                    # the vintage itself is kept


def test_a_changed_tcode_drops_the_series_not_the_vintage(raw, vintage_path, tmp_path):
    r = raw.copy()
    r.loc[0, "CUMFNS"] = "5"                             # neighbour has 1
    levels, _, report = D.load_archive_vintage(_write(r, tmp_path / "v.csv"), neighbour=vintage_path)
    assert report["dropped_tcode"] == ["CUMFNS"] and "CUMFNS" not in levels.columns


def test_a_renamed_series_is_still_checked_against_a_renamed_neighbour(raw, tmp_path):
    """Both sides of a pre-2015 pair carry the old PPI names: the rename must be applied to the
    neighbour too, or the renamed series would silently skip the check on both sides."""
    base = raw.rename(columns={"WPSFD49207": "PPIFGS", "WPSFD49502": "PPIFCG"})
    neighbour = _write(base.copy(), tmp_path / "n.csv")

    levels, _, report = D.load_archive_vintage(_write(base.copy(), tmp_path / "v.csv"), neighbour=neighbour)
    assert report["dropped_check"] == []
    assert report["renamed"] == {"PPIFGS": "WPSFD49207", "PPIFCG": "WPSFD49502"}

    corrupted = base.copy()
    rng = np.random.default_rng(0)
    vals = pd.to_numeric(corrupted.loc[1:, "PPIFGS"], errors="coerce")
    corrupted.loc[1:, "PPIFGS"] = (vals * (1 + rng.normal(0, 0.5, len(vals)))).to_numpy()   # corr << 0.98
    levels, _, report = D.load_archive_vintage(_write(corrupted, tmp_path / "v2.csv"), neighbour=neighbour)
    assert report["dropped_check"] == ["WPSFD49207"] and "WPSFD49207" not in levels.columns


def test_anchor_failure_refuses_the_vintage(raw, vintage_path, tmp_path):
    r = raw.copy()
    vals = pd.to_numeric(r.loc[1:, "INDPRO"], errors="coerce")
    r.loc[1:, "INDPRO"] = (vals * (1 + np.random.default_rng(1).normal(0, 0.5, len(vals)))).to_numpy()
    with pytest.raises(D.VintageError, match="INDPRO"):
        D.load_archive_vintage(_write(r, tmp_path / "v.csv"), neighbour=vintage_path)


def test_missing_anchor_refuses_the_vintage(raw, tmp_path):
    with pytest.raises(D.VintageError, match="CPIAUCSL"):
        D.load_archive_vintage(_write(raw.drop(columns=["CPIAUCSL"]), tmp_path / "v.csv"))


def test_half_empty_block_refuses_the_vintage(raw, tmp_path):
    drop = [c for c in D.GROWTH_BLOCK if c != "INDPRO"][:12]          # 22 -> 10 kept, below half
    with pytest.raises(D.VintageError, match="growth"):
        D.load_archive_vintage(_write(raw.drop(columns=drop), tmp_path / "v.csv"))


def test_archive_loader_refuses_a_shifted_header(raw, vintage_path, tmp_path):
    """The plain 2026-08 file lost one header name so every later column carried its neighbour's
    name. Against a neighbour, the misaligned series fail the check en masse; an anchor among
    them refuses the vintage rather than labelling from shifted columns."""
    r = raw.copy()
    cols = list(r.columns)
    i = cols.index("S&P div yield")
    shifted = cols[:i] + cols[i + 1:] + ["Unnamed: 999"]  # names slide left by one from i on
    r.columns = shifted
    with pytest.raises(D.VintageError):
        D.load_archive_vintage(_write(r, tmp_path / "v.csv"), neighbour=vintage_path)


def test_build_blocks_and_run_pipeline_accept_a_loader(vintage_path):
    calls = []

    def spy(path):
        calls.append(path)
        return D.load_fredmd(path)

    blocks = D.build_blocks(vintage_path, loader=spy)
    assert calls == [vintage_path] and set(blocks) >= {"growth", "inflation"}
    res = run_pipeline(vintage_path, asof="2010-12-31", loader=spy)
    assert calls[-1] == vintage_path and res.hmm.labels_filtered.index[-1] == pd.Timestamp("2010-12-01")
    ref = run_pipeline(vintage_path, asof="2010-12-31")
    pd.testing.assert_frame_equal(res.hmm.probs_filtered, ref.hmm.probs_filtered)
