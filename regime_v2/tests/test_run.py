import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run as runmod
from regime_v2 import regimes as R

# The asset stage refuses to run on substituted full-sample labels, so every driver test that
# exercises it must run a genuine walk-forward. --wf-step 24 keeps that honest and cheap
# (24 re-estimations instead of 455).
ASSET_WF = ["--wf-step", "24", "--skip-robustness", "--skip-expanding", "--skip-placebo"]


@pytest.fixture
def coarse_walkforward_publishes(monkeypatch):
    """Let a --wf-step 24 run publish.

    Subsampling the walk-forward makes §8's calendar-window shares unrepresentative: with a
    24-month step no sampled month falls inside covid_contraction_hmm's 4-month window at all,
    so it evaluates to NaN and blocks the publish. The thresholds themselves are owned by
    tests/test_acceptance.py (fast path, plus the RUN_SLOW full walk-forward); these tests are
    about the asset stage, so the gate is stubbed rather than the labels being faked.
    """
    monkeypatch.setattr(runmod.acceptance, "all_passed", lambda table: True)


def test_download_vintage_builds_url_and_writes(tmp_path):
    seen = {}
    def fake_fetch(url, timeout=60):
        seen["url"] = url
        return io.BytesIO(b"sasdate,INDPRO\nTransform:,5\n1/1/1959,1.0\n")
    p = runmod.download_vintage("2026-08", tmp_path, fetch=fake_fetch)
    assert seen["url"] == runmod.FREDMD_URL.format(vintage="2026-08")
    assert p == tmp_path / "fredmd_2026-08.csv" and p.read_bytes().startswith(b"sasdate")


def test_download_vintage_reports_every_url_on_failure(tmp_path):
    tried = []
    def failing(url, timeout=60):
        tried.append(url); raise OSError("403 Forbidden")
    with pytest.raises(SystemExit) as e:
        runmod.download_vintage("2026-08", tmp_path, fetch=failing)
    assert len(tried) == len(runmod.FREDMD_URLS) and all("2026-08" in u for u in tried)
    assert "403" in str(e.value) and not (tmp_path / "fredmd_2026-08.csv").exists()


def test_download_vintage_rejects_a_non_fredmd_response_and_falls_through(tmp_path):
    """The Fed site answers a missing vintage with an HTML page and status 200; that must not be
    saved as the vintage, and the next URL must still be tried."""
    tried = []
    def fetch(url, timeout=60):
        tried.append(url)
        if len(tried) == 1:
            return io.BytesIO(b"<!DOCTYPE html><html><body>Page not found</body></html>")
        return io.BytesIO(b"sasdate,INDPRO\nTransform:,5\n1/1/1959,1.0\n")
    p = runmod.download_vintage("2099-01", tmp_path, fetch=fetch)
    assert len(tried) == 2 and p.read_bytes().startswith(b"sasdate")


def test_download_vintage_fails_clearly_when_every_url_returns_html(tmp_path):
    def html(url, timeout=60):
        return io.BytesIO(b"<html>not yet</html>")
    with pytest.raises(SystemExit) as e:
        runmod.download_vintage("2099-01", tmp_path, fetch=html)
    assert "not a FRED-MD file" in str(e.value) and "not published yet" in str(e.value)
    assert not (tmp_path / "fredmd_2099-01.csv").exists()


def test_main_writes_contract(vintage_path, tmp_path):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding", "--no-assets",
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    labels = pd.read_csv(out / "regime_labels.csv", index_col="date", parse_dates=["date", "available_at"])
    for c in ["available_at", "growth_gap", "inflation_gap", "quadrant", "quadrant_theta", "hmm_walkforward",
              "quadrant_walkforward", "hmm_filtered", "hmm_smoothed_expost", "hmm_free", "gmm"] + [f"p_{r}" for r in R.REGIMES]:
        assert c in labels.columns, c
    assert labels["hmm_walkforward"].isna().all()          # disabled in this run
    assert labels["quadrant_walkforward"].isna().all()     # disabled in this run
    s = json.loads((out / "summary.json").read_text())
    tm = pd.DataFrame(s["transition_matrix"]).T            # written orient="index": outer key = from-state
    assert list(tm.index) == R.REGIMES and np.allclose(tm.sum(axis=1), 1.0, atol=1e-3)   # rounded to 4 dp
    for k in ["n_months", "sample", "regime_counts", "agreement_with_quadrants", "emission_only_agreement",
              "filtered_vs_smoothed_agreement", "share_max_prob_gt_095", "expected_duration_months",
              "min_transition_prob", "acceptance_tests", "label_source", "quadrant_source", "stagflation_1973_75",
              "stagflation_1980_82", "growth_gap_revision", "loadings"]:
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


def test_assets_stage_offline(vintage_path, returns_path, tmp_path, coarse_walkforward_publishes):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, *ASSET_WF, "--returns-cache", returns_path, "--out-dir", str(out),
                      "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    a = s["assets"]
    assert a["skipped"] is None and a["window"]["n_months"] >= 200
    assert "label_column" not in a                            # no substitution: the labels are the real thing
    assert set(a["n_per_regime"]) <= set(R.REGIMES) and sum(a["n_per_regime"].values()) == a["window"]["n_months"]
    assert 0.0 <= a["growth_share_6040"]["growth_share"] <= 1.0
    assert set(a["backtest"]) == {"cost_bp_0", "cost_bp_10"} and "PIT_MaxSharpe" in a["backtest"]["cost_bp_0"]["perf"]
    assert set(a["lookahead"]) >= {"moment_lookahead", "label_lookahead", "total"}
    assert set(a["lookahead_longonly"]) == {"insample_sharpe", "oracle_sharpe", "pit_sharpe",
                                            "moment_lookahead", "label_lookahead", "total"}
    perf0 = a["backtest"]["cost_bp_0"]["perf"]
    for s in ["PIT_LongOnly_MaxSharpe", "PIT_RiskParity", "Oracle_LongOnly_MaxSharpe", "InSample_LongOnly_expost"]:
        assert s in perf0, s
    assert a["backtest_placebo"] is None                      # skipped in this run
    for f in ["regime_returns.csv", "regime_corr_Goldilocks.csv", "backtest_returns.csv", "portfolio_weights.csv"]:
        assert (out / f).exists(), f
    for n in ["fig8_regime_returns", "fig9_mixture_6040", "fig10_backtest_wealth", "fig11_pit_weights"]:
        assert (figs / f"{n}.png").exists(), n
    assert not (out / runmod.ASSETS_STAGING).exists() and not (figs / runmod.ASSETS_STAGING).exists()
    pw = pd.read_csv(out / "portfolio_weights.csv", index_col=0, header=[0, 1], parse_dates=True)
    assert list(dict.fromkeys(pw.columns.get_level_values(0))) == ["PIT_MaxSharpe", "ProbWeighted_MaxSharpe",
                                                                  "PIT_LongOnly_MaxSharpe", "PIT_RiskParity"]
    for s in ("PIT_LongOnly_MaxSharpe", "PIT_RiskParity"):
        assert (pw[s].to_numpy() >= -1e-9).all() and np.allclose(pw[s].sum(axis=1), 1.0), s
    acc = pd.read_csv(out / "acceptance.csv", index_col=0)
    for n in ["pit_sharpe", "oracle_sharpe", "insample_sharpe", "label_lookahead", "moment_lookahead",
              "growth_share_6040", "static_6040_sharpe", "pit_sharpe_10bp", "growth_share_6040_r2",
              "pit_longonly_sharpe", "pit_riskparity_sharpe", "longonly_moment_lookahead", "longonly_label_lookahead"]:
        assert n in acc.index and acc.loc[n, "op"] == "report", n
        assert np.isfinite(acc.loc[n, "value"]), n
    # each benchmark row must equal the number it is the benchmark for
    assert acc.loc["static_6040_sharpe", "value"] == pytest.approx(a["backtest"]["cost_bp_0"]["perf"]["Static_6040"]["sharpe"])
    assert acc.loc["pit_sharpe_10bp", "value"] == pytest.approx(a["backtest"]["cost_bp_10"]["perf"]["PIT_MaxSharpe"]["sharpe"])
    assert acc.loc["growth_share_6040_r2", "value"] == pytest.approx(a["growth_share_6040"]["r2"])
    assert acc.loc["pit_longonly_sharpe", "value"] == pytest.approx(perf0["PIT_LongOnly_MaxSharpe"]["sharpe"])
    assert acc.loc["pit_riskparity_sharpe", "value"] == pytest.approx(perf0["PIT_RiskParity"]["sharpe"])
    assert acc.loc["longonly_moment_lookahead", "value"] == pytest.approx(a["lookahead_longonly"]["moment_lookahead"])
    assert acc.loc["longonly_label_lookahead", "value"] == pytest.approx(a["lookahead_longonly"]["label_lookahead"])
    assert "growth_share_6040_r2" in acc.loc["growth_share_6040", "rationale"]
    assert "static_6040_sharpe" in acc.loc["pit_sharpe", "rationale"]
    assert "static_6040_sharpe" in acc.loc["pit_riskparity_sharpe", "rationale"]


def test_no_walkforward_skips_asset_stage(vintage_path, returns_path, tmp_path):
    """Without a walk-forward there are no real-time labels, so the stage must skip and say so
    rather than quietly publishing full-sample labels under PIT names."""
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--no-walkforward", "--skip-robustness", "--skip-expanding", "--skip-placebo",
                      "--returns-cache", returns_path, "--out-dir", str(out), "--figs-dir", str(figs),
                      "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    a = json.loads((out / "summary.json").read_text())["assets"]
    assert a["skipped"] == "walk-forward disabled: the asset stage needs real-time labels"
    assert "label_column" not in a
    assert not (out / "backtest_returns.csv").exists() and not (figs / "fig8_regime_returns.png").exists()


def test_assets_stage_skips_cleanly_on_failure(vintage_path, tmp_path, monkeypatch, coarse_walkforward_publishes):
    out, figs = tmp_path / "out", tmp_path / "figs"
    def boom(**kw): raise OSError("no network")
    monkeypatch.setattr(runmod.assets, "load_returns", boom)
    rc = runmod.main([vintage_path, *ASSET_WF, "--returns-cache", str(tmp_path / "missing.parquet"),
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert "no network" in s["assets"]["skipped"]
    assert not (out / "backtest_returns.csv").exists()


def test_assets_stage_skips_cleanly_on_failure_after_download(vintage_path, returns_path, tmp_path, monkeypatch,
                                                              coarse_walkforward_publishes):
    """A failure anywhere past load_returns (e.g. inside portfolio.backtest, after publish() has already
    swapped output/ into place and after regime_returns.csv etc. would have been written) must still leave
    rc == 0 and acceptance_all_passed True -- the asset stage never changes the engine's exit code -- and
    must publish nothing: no half-written asset artefacts, and no survivors from an earlier run."""
    out, figs = tmp_path / "out", tmp_path / "figs"
    out.mkdir(); (out / "backtest_returns.csv").write_text("stale from the previous run")
    def boom(*a, **kw): raise RuntimeError("backtest exploded")
    monkeypatch.setattr(runmod.portfolio, "backtest", boom)
    rc = runmod.main([vintage_path, *ASSET_WF, "--returns-cache", returns_path, "--out-dir", str(out),
                      "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert "backtest exploded" in s["assets"]["skipped"]
    assert s["acceptance_all_passed"] is True
    for f in ["regime_returns.csv", "backtest_returns.csv", "portfolio_weights.csv"]:
        assert not (out / f).exists(), f
    assert list(out.glob("regime_corr_*.csv")) == []
    for n in ["fig8_regime_returns", "fig9_mixture_6040", "fig10_backtest_wealth", "fig11_pit_weights"]:
        assert not (figs / f"{n}.png").exists(), n
    assert not (out / runmod.ASSETS_STAGING).exists() and not (figs / runmod.ASSETS_STAGING).exists()


def test_summary_has_current_and_run_blocks(vintage_path, tmp_path):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--no-walkforward", "--no-assets", "--skip-robustness", "--skip-expanding",
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    lab = pd.read_csv(out / "regime_labels.csv", index_col=0, parse_dates=True)
    for c in ["growth_factor", "inflation_factor"]:
        assert c in lab.columns and lab[c].notna().sum() > 500
    run = s["run"]
    assert run["engine"] == "regime_v2" and run["vintage"] == os.path.basename(vintage_path)
    assert run["asof"] == str(lab.index[-1].date()) and "T" in run["timestamp"]
    cur = s["current"]
    assert cur["month"] == lab.index[-1].strftime("%Y-%m")
    assert cur["regime"] in R.REGIMES and cur["quadrant"] in R.REGIMES
    assert cur["regime"] == lab["hmm_filtered"].iloc[-1]          # walk-forward disabled -> filtered column
    assert set(cur["probs"]) == set(R.REGIMES) and abs(sum(cur["probs"].values()) - 1) < 1e-6
    assert cur["growth_gap"] == pytest.approx(float(lab["growth_gap"].iloc[-1]))


def test_summary_current_uses_last_labelled_month_under_coarse_walkforward(vintage_path, tmp_path,
                                                                           coarse_walkforward_publishes):
    """A coarse --wf-step leaves the sample's most recent months without an hmm_walkforward
    label (labels_frame left-joins the sparse walk-forward grid with no forward-fill), so
    "current" and run["asof"] must be read at the last month that column actually has a
    label for, not at the sample's last month, or they come back NaN."""
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, "--wf-step", "24", "--no-assets", "--skip-robustness", "--skip-expanding",
                      "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    lab = pd.read_csv(out / "regime_labels.csv", index_col=0, parse_dates=True)
    last_labelled = lab["hmm_walkforward"].dropna().index[-1]
    assert last_labelled < lab.index[-1]           # the coarse grid really does miss the sample end
    assert s["run"]["asof"] == str(last_labelled.date())
    assert s["current"]["month"] == last_labelled.strftime("%Y-%m")
    assert s["current"]["regime"] in R.REGIMES and s["current"]["quadrant"] in R.REGIMES
