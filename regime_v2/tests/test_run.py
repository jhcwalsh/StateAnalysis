import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run as runmod
from regime_v2 import regimes as R


def test_download_vintage_builds_url_and_writes(tmp_path):
    seen = {}
    def fake_fetch(url, timeout=60):
        seen["url"] = url
        return io.BytesIO(b"sasdate,INDPRO\nTransform:,5\n1/1/1959,1.0\n")
    p = runmod.download_vintage("2026-08", tmp_path, fetch=fake_fetch)
    assert seen["url"] == runmod.FREDMD_URL.format(vintage="2026-08")
    assert p == tmp_path / "fredmd_2026-08.csv" and p.read_bytes().startswith(b"sasdate")


def test_main_writes_contract(vintage_path, tmp_path):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding",
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    labels = pd.read_csv(out / "regime_labels.csv", index_col="date", parse_dates=["date", "available_at"])
    for c in ["available_at", "growth_gap", "inflation_gap", "quadrant", "quadrant_theta", "hmm_walkforward",
              "hmm_filtered", "hmm_smoothed_expost", "hmm_free", "gmm"] + [f"p_{r}" for r in R.REGIMES]:
        assert c in labels.columns, c
    assert labels["hmm_walkforward"].isna().all()          # disabled in this run
    s = json.loads((out / "summary.json").read_text())
    tm = pd.DataFrame(s["transition_matrix"]).T            # written orient="index": outer key = from-state
    assert list(tm.index) == R.REGIMES and np.allclose(tm.sum(axis=1), 1.0, atol=1e-3)   # rounded to 4 dp
    for k in ["n_months", "sample", "regime_counts", "agreement_with_quadrants", "emission_only_agreement",
              "filtered_vs_smoothed_agreement", "share_max_prob_gt_095", "expected_duration_months",
              "min_transition_prob", "acceptance_tests", "label_source", "stagflation_1973_75", "stagflation_1980_82",
              "growth_gap_revision", "loadings"]:
        assert k in s, k
    assert s["label_source"] == "full-sample filtered (walk-forward disabled)"
    assert (out / "acceptance.csv").exists() and (out / "outliers_removed.csv").exists()
    assert (tmp_path / "README.md").read_text().count("| INDPRO |") == 1
    assert all((figs / f"fig{i}_{n}.png").exists() for i, n in
               [(1, "factors_gaps"), (2, "regime_timeline"), (3, "state_space"), (4, "hmm_probabilities"),
                (5, "revisions"), (6, "classifier_comparison")])


def test_publish_refuses_when_acceptance_fails(vintage_path, tmp_path, monkeypatch):
    out, figs = tmp_path / "out", tmp_path / "figs"
    out.mkdir(); (out / "regime_labels.csv").write_text("old")
    monkeypatch.setattr(runmod.acceptance, "all_passed", lambda table: False)
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding",
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 1
    assert (out / "regime_labels.csv").read_text() == "old"
    assert (tmp_path / "out.staging" / "output" / "summary.json").exists()   # staged results kept for inspection


def test_publish_swaps_by_rename_and_cleans_up(tmp_path):
    staging = tmp_path / "out.staging"
    (staging / "output").mkdir(parents=True); (staging / "figs").mkdir()
    (staging / "output" / "regime_labels.csv").write_text("new")
    (staging / "figs" / "fig1.png").write_text("png")
    out, figs = tmp_path / "out", tmp_path / "figs"
    out.mkdir(); (out / "regime_labels.csv").write_text("old"); (out / "stale.txt").write_text("x")
    runmod.publish(staging, out, figs)
    assert (out / "regime_labels.csv").read_text() == "new"
    assert not (out / "stale.txt").exists()
    assert (figs / "fig1.png").read_text() == "png"
    assert not staging.exists()
    assert not any(p.name.endswith((".new", ".old")) for p in tmp_path.iterdir())
