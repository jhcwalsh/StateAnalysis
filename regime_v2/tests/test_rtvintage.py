"""fit_hmm4_rt_vintage: month t labelled from vintage t+1 only."""
import numpy as np
import pandas as pd
import pytest

from regime_v2 import rtvintage as RT
from regime_v2.pipeline import run_pipeline
from regime_v2.regimes import REGIMES


def _bump_recent_indpro(raw, months=6, factor=1.03):
    raw = raw.copy()
    vals = pd.to_numeric(raw.loc[1:, "INDPRO"], errors="coerce").to_numpy()
    vals[-months:] = vals[-months:] * factor
    raw.loc[1:, "INDPRO"] = vals
    return raw


def test_helpers():
    assert RT.vintage_for_month("2026-06") == "2026-07"
    assert RT.vintage_for_month("2025-12") == "2026-01"
    assert RT.provenance("2014-12") == "reconstructed" and RT.provenance("2015-01") == "published"


def test_list_vintages_sorted_and_named(make_archive, tmp_path):
    root = make_archive(tmp_path / "a", ["2026-07", "2026-05", "2026-06"])
    vs = RT.list_vintages(root)
    assert list(vs) == ["2026-05", "2026-06", "2026-07"]
    assert vs["2026-06"].name == "fredmd_2026-06.csv"


def test_last_month_equals_the_pipeline_on_the_pinned_file(make_archive, tmp_path, vintage_path):
    root = make_archive(tmp_path / "a", ["2026-06", "2026-07"])      # 2026-07 == the pinned file
    out = RT.fit_hmm4_rt_vintage(root, start="2026-05")
    ref = run_pipeline(vintage_path, asof="2026-06-30")
    t = pd.Timestamp("2026-06-01")
    assert list(out.labels.index) == [pd.Timestamp("2026-05-01"), t]
    np.testing.assert_allclose(out.probs.loc[t].to_numpy(), ref.hmm.probs_filtered.loc[t].reindex(REGIMES).to_numpy())
    assert out.labels.loc[t] == ref.hmm.labels_filtered.loc[t]
    assert out.growth_gap.loc[t] == pytest.approx(float(ref.G.loc[t]))
    assert out.provenance.loc[t] == "published" and out.gaps == {}


def test_month_t_reads_only_vintage_t_plus_1(make_archive, tmp_path):
    clean = make_archive(tmp_path / "clean", ["2026-05", "2026-06", "2026-07"])
    later = make_archive(tmp_path / "later", ["2026-05", "2026-06", "2026-07"],
                         plant={"2026-07": _bump_recent_indpro})          # revision only in t+2 for t=2026-05
    own = make_archive(tmp_path / "own", ["2026-05", "2026-06", "2026-07"],
                       plant={"2026-06": _bump_recent_indpro})            # revision in t+1 for t=2026-05
    a = RT.fit_hmm4_rt_vintage(clean, start="2026-04")
    b = RT.fit_hmm4_rt_vintage(later, start="2026-04")
    c = RT.fit_hmm4_rt_vintage(own, start="2026-04")
    t = pd.Timestamp("2026-05-01")
    np.testing.assert_array_equal(a.probs.loc[t].to_numpy(), b.probs.loc[t].to_numpy())
    assert not np.allclose(a.probs.loc[t].to_numpy(), c.probs.loc[t].to_numpy(), atol=1e-9)
    assert a.growth_gap.loc[t] != c.growth_gap.loc[t]


def test_start_is_clamped_to_the_first_vintage(make_archive, tmp_path):
    root = make_archive(tmp_path / "a", ["2026-06", "2026-07"])
    out = RT.fit_hmm4_rt_vintage(root, start="1999-07")
    assert out.start == "2026-05" and list(out.labels.index.strftime("%Y-%m")) == ["2026-05", "2026-06"]


def test_missing_vintage_file_is_a_gap_with_reason(make_archive, tmp_path):
    root = make_archive(tmp_path / "a", ["2026-05", "2026-07"])         # 2026-06 absent
    out = RT.fit_hmm4_rt_vintage(root, start="2026-04")
    assert list(out.labels.index.strftime("%Y-%m")) == ["2026-04", "2026-06"]
    assert out.gaps == {"2026-05": "vintage 2026-06 not in archive"}


def test_gap_when_the_target_month_row_is_empty(make_archive, tmp_path):
    def blank_last_row(raw):
        raw = raw.copy()
        raw.iloc[-1, 1:] = ""                                            # the shutdown-file shape
        return raw
    root = make_archive(tmp_path / "a", ["2026-06", "2026-07"], plant={"2026-07": blank_last_row})
    out = RT.fit_hmm4_rt_vintage(root, start="2026-05")
    assert "2026-06" in out.gaps and "2026-06" in out.gaps["2026-06"]
    assert list(out.labels.index.strftime("%Y-%m")) == ["2026-05"]


def test_refused_vintage_is_a_gap_and_reported(make_archive, tmp_path):
    def kill_cpi(raw):
        return raw.drop(columns=["CPIAUCSL"])
    root = make_archive(tmp_path / "a", ["2026-06", "2026-07"], plant={"2026-07": kill_cpi})
    out = RT.fit_hmm4_rt_vintage(root, start="2026-05")
    assert "CPIAUCSL" in out.gaps["2026-06"] and "refused" in out.reports["2026-07"]
    assert "renamed" in out.reports["2026-06"]


def test_a_pipeline_error_for_one_month_is_a_gap_not_an_abort(make_archive, tmp_path, monkeypatch):
    root = make_archive(tmp_path / "a", ["2026-05", "2026-06", "2026-07"])
    middle = "fredmd_2026-06.csv"                                        # vintage for month 2026-05

    def flaky_run_pipeline(path, *a, **kw):
        if str(path).endswith(middle):
            raise ValueError("only 12 months")
        return run_pipeline(path, *a, **kw)

    monkeypatch.setattr(RT, "run_pipeline", flaky_run_pipeline)
    out = RT.fit_hmm4_rt_vintage(root, start="2026-04")
    assert list(out.labels.index.strftime("%Y-%m")) == ["2026-04", "2026-06"]
    assert "failed: ValueError" in out.gaps["2026-05"]
    assert out.reports["2026-06"] == {"failed": "ValueError: only 12 months"}
