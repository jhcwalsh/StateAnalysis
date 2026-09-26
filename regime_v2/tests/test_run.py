import io
import json
import os
from datetime import date
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


FREDMD_CSV = b"sasdate,INDPRO\nTransform:,5\n1/1/1959,1.0\n"
FREDMD_HTML = b"<!DOCTYPE html><html><body>Page not found</body></html>"


def test_placebo_block_publishes_each_strategy_and_none_when_skipped():
    plcs = {s: {"real": 1.0 + i, "percentile": 40.0 + i, "n": 3, "null": np.array([0.1, 0.2, 0.3]) + i}
            for i, s in enumerate(runmod.PLACEBO_STRATEGIES)}
    assert runmod.PLACEBO_STRATEGIES == ("PIT_MaxSharpe", "PIT_LongOnly_MaxSharpe")
    lo = runmod._placebo_block(plcs, "PIT_LongOnly_MaxSharpe")
    assert lo == {"real": 2.0, "percentile": 41.0, "n": 3, "null": [1.1, 1.2, 1.3]}
    json.dumps(lo)                                             # summary.json must serialise it
    assert runmod._placebo_block(None, "PIT_LongOnly_MaxSharpe") is None


def test_download_vintage_prefers_the_revised_file_and_writes(tmp_path):
    """The Fed's own current.csv link points at YYYY-rev-MM-md.csv, not YYYY-MM-md.csv. The plain
    file has shipped a malformed header (the dropped `S&P div yield` name that load_fredmd repairs
    against the pinned vintage), so the revised file is tried first."""
    seen = {}
    def fake_fetch(url, timeout=60):
        seen["url"] = url
        return io.BytesIO(FREDMD_CSV)
    p = runmod.download_vintage("2026-08", tmp_path, fetch=fake_fetch)
    assert seen["url"].endswith("/monthly/2026-rev-08-md.csv")
    assert seen["url"] == runmod.fredmd_urls("2026-08")[0]
    assert p == tmp_path / "fredmd_2026-08.csv" and p.read_bytes().startswith(b"sasdate")


def test_download_vintage_falls_back_to_the_plain_file(tmp_path):
    """Not every vintage has a revised file; when it is missing the plain one must still be used."""
    tried = []
    def fetch(url, timeout=60):
        tried.append(url)
        return io.BytesIO(FREDMD_HTML if "-rev-" in url else FREDMD_CSV)
    p = runmod.download_vintage("2026-08", tmp_path, fetch=fetch)
    assert len(tried) == 2 and tried[1].endswith("/monthly/2026-08-md.csv")
    assert p.read_bytes().startswith(b"sasdate")


def test_download_vintage_reports_every_url_on_failure(tmp_path):
    tried = []
    def failing(url, timeout=60):
        tried.append(url); raise OSError("403 Forbidden")
    with pytest.raises(SystemExit) as e:
        runmod.download_vintage("2026-08", tmp_path, fetch=failing)
    assert tried == runmod.fredmd_urls("2026-08")
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


def _fetch_published_through(*vintages):
    """A fetch that serves a FRED-MD file for the named vintages and the site's HTML 404 for the rest."""
    def fetch(url, timeout=60):
        return io.BytesIO(FREDMD_CSV if any(v.replace("-", "-rev-") in url or v in url
                                            for v in vintages) else FREDMD_HTML)
    return fetch


def test_latest_vintage_skips_a_month_that_is_not_published_yet(tmp_path):
    """FRED-MD does not post a vintage during its own month: on 2026-09-13 the site answers
    2026-09 with an HTML page and the newest real vintage is 2026-08."""
    v, p = runmod.latest_vintage(date(2026, 9, 13), tmp_path, fetch=_fetch_published_through("2026-08"))
    assert v == "2026-08"
    assert p == tmp_path / "fredmd_2026-08.csv" and p.read_bytes().startswith(b"sasdate")


def test_latest_vintage_takes_the_current_month_once_it_appears(tmp_path):
    v, _ = runmod.latest_vintage(date(2026, 10, 20), tmp_path, fetch=_fetch_published_through("2026-10", "2026-09"))
    assert v == "2026-10"


def test_latest_vintage_returns_none_when_nothing_downloads(tmp_path):
    """A source outage, not a missing month: the caller must be able to tell this apart from a
    vintage that simply is not newer than the published one."""
    def html(url, timeout=60):
        return io.BytesIO(FREDMD_HTML)
    assert runmod.latest_vintage(date(2026, 9, 13), tmp_path, fetch=html) == (None, None)


def test_latest_vintage_does_not_probe_indefinitely(tmp_path):
    tried = []
    def html(url, timeout=60):
        tried.append(url); return io.BytesIO(FREDMD_HTML)
    runmod.latest_vintage(date(2026, 9, 13), tmp_path, fetch=html, max_back=2)
    months = {u.rsplit("/", 1)[1] for u in tried}
    assert {m for m in months if "2026-09" in m or "2026-rev-09" in m}          # the current month
    assert not any("2026-06" in m for m in months)                              # and no further back than 2026-07


def test_published_vintage_reads_the_run_block(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    (out / "summary.json").write_text(json.dumps({"run": {"vintage": "fredmd_2026-08.csv"}}))
    assert runmod.published_vintage(out) == "2026-08"


def test_published_vintage_is_none_when_nothing_is_published(tmp_path):
    assert runmod.published_vintage(tmp_path / "nothing") is None


def test_if_newer_exits_3_without_running_the_engine(tmp_path):
    """The daily job's quiet path: the newest vintage is the one already published, so the run
    must cost nothing and report "nothing to do" distinctly from a failure."""
    out = tmp_path / "out"; out.mkdir()
    (out / "summary.json").write_text(json.dumps({"run": {"vintage": "fredmd_2026-08.csv"}}))
    rc = runmod.main(["--vintage", "latest", "--if-newer", "--out-dir", str(out), "--figs-dir", str(tmp_path / "figs")],
                     today=date(2026, 9, 13), fetch=_fetch_published_through("2026-08"))
    assert rc == 3
    assert not (tmp_path / "out.staging").exists()      # the pipeline never started


def test_a_check_that_publishes_nothing_still_records_that_it_happened(tmp_path):
    """The daily job's quiet path must leave evidence, or the dashboard can only ever show the
    date of the last publish and a fresh site looks stale after a few silent days."""
    from regime_v2 import publish as P
    out = tmp_path / "out"; out.mkdir()
    (out / "summary.json").write_text(json.dumps({"run": {"vintage": "fredmd_2026-08.csv"}}))
    rc = runmod.main(["--vintage", "latest", "--if-newer", "--out-dir", str(out), "--figs-dir", str(tmp_path / "figs")],
                     today=date(2026, 9, 13), fetch=_fetch_published_through("2026-08"))
    assert rc == 3
    assert P.read_last_check(out)["latest_vintage"] == "2026-08"


def test_a_check_that_found_no_vintage_is_recorded_too(tmp_path):
    from regime_v2 import publish as P
    out = tmp_path / "out"; out.mkdir()
    def html(url, timeout=60):
        return io.BytesIO(FREDMD_HTML)
    rc = runmod.main(["--vintage", "latest", "--out-dir", str(out), "--figs-dir", str(tmp_path / "figs")],
                     today=date(2026, 9, 13), fetch=html)
    assert rc == 4
    assert P.read_last_check(out)["latest_vintage"] is None


def test_if_newer_proceeds_when_the_vintage_is_newer(tmp_path, monkeypatch):
    out = tmp_path / "out"; out.mkdir()
    (out / "summary.json").write_text(json.dumps({"run": {"vintage": "fredmd_2026-07.csv"}}))
    def reached(*a, **kw):
        raise RuntimeError("pipeline reached")
    monkeypatch.setattr(runmod, "run_pipeline", reached)
    with pytest.raises(RuntimeError, match="pipeline reached"):
        runmod.main(["--vintage", "latest", "--if-newer", "--out-dir", str(out), "--figs-dir", str(tmp_path / "figs")],
                    today=date(2026, 9, 13), fetch=_fetch_published_through("2026-08"))


def test_if_newer_proceeds_when_nothing_is_published_yet(tmp_path, monkeypatch):
    """A fresh volume has no summary.json; the first start must still publish."""
    def reached(*a, **kw):
        raise RuntimeError("pipeline reached")
    monkeypatch.setattr(runmod, "run_pipeline", reached)
    with pytest.raises(RuntimeError, match="pipeline reached"):
        runmod.main(["--vintage", "latest", "--if-newer", "--out-dir", str(tmp_path / "out"),
                     "--figs-dir", str(tmp_path / "figs")],
                    today=date(2026, 9, 13), fetch=_fetch_published_through("2026-08"))


def test_unresolvable_vintage_exits_4(tmp_path):
    """No vintage at all downloaded — the source is unreachable, not merely unchanged. The
    scheduled job debounces this code and alerts immediately on the others."""
    def html(url, timeout=60):
        return io.BytesIO(FREDMD_HTML)
    rc = runmod.main(["--vintage", "latest", "--out-dir", str(tmp_path / "out"),
                      "--figs-dir", str(tmp_path / "figs")], today=date(2026, 9, 13), fetch=html)
    assert rc == 4


def test_if_newer_requires_a_vintage(tmp_path, vintage_path):
    with pytest.raises(SystemExit):
        runmod.main([vintage_path, "--if-newer", "--out-dir", str(tmp_path / "out")])


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
    assert a["backtest_placebo_longonly"] is None             # skipped in this run
    assert acc.loc["longonly_placebo_pct", "op"] == "report" and np.isnan(acc.loc["longonly_placebo_pct", "value"])
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


RT_FLAGS = ["--wf-step", "24", "--skip-robustness", "--skip-expanding", "--skip-placebo"]


def _run(vintage_path, returns_path, tmp_path, extra):
    out, figs = tmp_path / "out", tmp_path / "figs"
    rc = runmod.main([vintage_path, *RT_FLAGS, "--returns-cache", returns_path, "--out-dir", str(out),
                      "--figs-dir", str(figs), "--data-sheet", str(tmp_path / "README.md"), *extra])
    return rc, out, figs


def test_rt_vintage_stage_publishes_columns_block_rows_and_figure(vintage_path, returns_path, tmp_path,
                                                                  make_archive, monkeypatch):
    from regime_v2 import acceptance
    monkeypatch.setattr(acceptance, "all_passed", lambda table: True)
    # 2024-12 is on the --wf-step 24 grid (1978-12 + 24k), so the overlap with hmm_walkforward is
    # non-empty; vintages 2024-12..2025-02 label 2024-11..2025-01.
    arch = make_archive(tmp_path / "arch", ["2024-12", "2025-01", "2025-02"])
    rc, out, figs = _run(vintage_path, returns_path, tmp_path, ["--vintage-archive", str(arch)])
    assert rc == 0
    lab = pd.read_csv(out / "regime_labels.csv", index_col=0, parse_dates=True)
    for c in ["hmm_rt_vintage", "growth_gap_rt_vintage", "inflation_gap_rt_vintage", *(f"p_rt_{r}" for r in R.REGIMES)]:
        assert c in lab.columns, c
    assert list(lab["hmm_rt_vintage"].dropna().index.strftime("%Y-%m")) == ["2024-11", "2024-12", "2025-01"]
    s = json.loads((out / "summary.json").read_text())
    b = s["rt_vintage"]
    assert b.get("skipped") is None and b["window"] == {"start": "2024-12", "end": "2024-12", "n_months": 1} and b["n_gaps"] == 0
    assert b["agreement"]["reconstructed"] is None and b["agreement"]["published"] in (0.0, 1.0)
    acc = pd.read_csv(out / "acceptance.csv", index_col=0)
    for n in ["rt_vintage_agreement", "rt_vintage_agreement_reconstructed", "rt_vintage_agreement_published",
              "rt_growth_gap_corr", "rt_inflation_gap_corr"]:
        assert n in acc.index and acc.loc[n, "op"] == "report", n
    assert acc.loc["rt_vintage_agreement", "value"] == pytest.approx(b["agreement"]["overall"])
    assert (figs / "fig12_rt_vintage.png").exists()
    # the published label and its numbers are the same as a run without the archive
    rc2, out2, _ = _run(vintage_path, returns_path, tmp_path / "noarch", [])
    lab2 = pd.read_csv(out2 / "regime_labels.csv", index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(lab.drop(columns=[c for c in lab if "rt_vintage" in c or c.startswith("p_rt_")]),
                                  lab2.drop(columns=[c for c in lab2 if "rt_vintage" in c or c.startswith("p_rt_")]))
    s2 = json.loads((out2 / "summary.json").read_text())
    assert s2["current"] == s["current"] and s2["rt_vintage"]["skipped"] == "no vintage archive configured"
    assert not (tmp_path / "noarch" / "figs" / "fig12_rt_vintage.png").exists()


def test_rt_stage_failure_never_changes_the_exit_code(vintage_path, returns_path, tmp_path, make_archive, monkeypatch):
    from regime_v2 import acceptance, rtvintage
    monkeypatch.setattr(acceptance, "all_passed", lambda table: True)
    monkeypatch.setattr(rtvintage, "fit_hmm4_rt_vintage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    arch = make_archive(tmp_path / "arch", ["2024-12", "2025-01"])
    rc, out, figs = _run(vintage_path, returns_path, tmp_path, ["--vintage-archive", str(arch)])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["rt_vintage"] == {"skipped": "RuntimeError: boom"} and s["current"]["regime"] in R.REGIMES
    assert not (figs / "fig12_rt_vintage.png").exists()
    lab = pd.read_csv(out / "regime_labels.csv", index_col=0, parse_dates=True)
    assert lab["hmm_rt_vintage"].isna().all()


def test_empty_archive_is_skipped_not_fatal(vintage_path, returns_path, tmp_path, monkeypatch):
    from regime_v2 import acceptance
    monkeypatch.setattr(acceptance, "all_passed", lambda table: True)
    rc, out, figs = _run(vintage_path, returns_path, tmp_path, ["--vintage-archive", str(tmp_path / "nothing")])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert "no fredmd_" in s["rt_vintage"]["skipped"]
    acc = pd.read_csv(out / "acceptance.csv", index_col=0)
    assert np.isnan(acc.loc["rt_vintage_agreement", "value"])


def test_archive_downloaded_vintage_copies_once(tmp_path):
    src = tmp_path / "fredmd_2026-08.csv"
    src.write_text("sasdate,INDPRO\nTransform:,5\n1/1/2026,1.0\n")
    arch = tmp_path / "arch"
    assert runmod.archive_downloaded_vintage(src, arch) is True
    assert (arch / "fredmd_2026-08.csv").read_text() == src.read_text()
    src.write_text("changed")
    assert runmod.archive_downloaded_vintage(src, arch) is False          # never overwrites
    assert (arch / "fredmd_2026-08.csv").read_text() != "changed"
    assert runmod.archive_downloaded_vintage(src, None) is False


def test_download_path_is_archived_when_configured(tmp_path, monkeypatch, vintage_path):
    """`--vintage YYYY-MM --vintage-archive DIR` copies the downloaded file into DIR before the engine runs."""
    calls = {}
    monkeypatch.setattr(runmod, "download_vintage", lambda v, d, fetch=None: Path(vintage_path))
    monkeypatch.setattr(runmod, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(SystemExit(99)))
    arch = tmp_path / "arch"
    with pytest.raises(SystemExit):
        runmod.main(["--vintage", "2026-07", "--vintage-archive", str(arch), "--out-dir", str(tmp_path / "o"),
                     "--figs-dir", str(tmp_path / "f")])
    assert (arch / "fredmd_2026-07.csv").exists()


def test_rt_backtest_inputs_hold_the_label_over_gaps(vintage_path):
    from regime_v2 import rtvintage as RT
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    labels_df = pd.DataFrame({"hmm_walkforward": ["Goldilocks"] * 5, "available_at": idx + pd.DateOffset(months=1)},
                             index=pd.DatetimeIndex(idx, name="date"))
    rt_idx = idx[[0, 1, 3, 4]]                                 # 2020-03 is a gap
    probs = pd.DataFrame(0.0, index=rt_idx, columns=R.REGIMES); probs["Contraction"] = 1.0
    rtv = RT.RtVintageResult(labels=pd.Series(["Contraction"] * 4, index=rt_idx, name="hmm_rt_vintage"), probs=probs,
                             growth_gap=pd.Series(0.0, index=rt_idx), inflation_gap=pd.Series(0.0, index=rt_idx),
                             gaps={"2020-03": "x"}, provenance=pd.Series("published", index=rt_idx), start="2020-01")
    lf, pr, held = runmod.rt_backtest_inputs(labels_df, rtv)
    assert held == 1 and lf.loc["2020-03-01", "hmm_walkforward"] == "Contraction"
    assert pr.loc["2020-03-01", "Contraction"] == 1.0
    assert list(lf["hmm_walkforward"]) == ["Contraction"] * 5


def test_rt_vintage_backtest_is_published_beside_the_others(vintage_path, returns_path, tmp_path, make_archive, monkeypatch):
    from regime_v2 import acceptance
    monkeypatch.setattr(acceptance, "all_passed", lambda table: True)
    arch = make_archive(tmp_path / "arch", ["2024-12", "2025-01", "2025-02"])
    rc, out, figs = _run(vintage_path, returns_path, tmp_path, ["--vintage-archive", str(arch)])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    rb = s["assets"]["rt_vintage_backtest"]
    assert set(rb["perf"]) == {"cost_bp_0", "cost_bp_10"} and set(rb["perf"]["cost_bp_0"]) == {"PIT_MaxSharpe", "PIT_LongOnly_MaxSharpe", "Static_6040"}
    assert rb["held_months"] == 0
    acc = pd.read_csv(out / "acceptance.csv", index_col=0)
    assert acc.loc["rt_pit_sharpe", "value"] == pytest.approx(rb["perf"]["cost_bp_0"]["PIT_MaxSharpe"]["sharpe"])
    assert acc.loc["rt_pit_longonly_sharpe", "value"] == pytest.approx(rb["perf"]["cost_bp_0"]["PIT_LongOnly_MaxSharpe"]["sharpe"])
    # unchanged existing asset numbers: same PIT Sharpe as without the archive
    rc2, out2, _ = _run(vintage_path, returns_path, tmp_path / "noarch", [])
    s2 = json.loads((out2 / "summary.json").read_text())
    assert s2["assets"]["lookahead"] == s["assets"]["lookahead"] and s2["assets"]["rt_vintage_backtest"] is None
