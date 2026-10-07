import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_v2 import publish as P
from regime_v2.regimes import REGIMES


def test_load_published_contract(published_dir):
    out, figs = published_dir
    pub = P.load_published(out, figs)
    assert isinstance(pub.labels.index, pd.DatetimeIndex) and pub.labels.index.name == "date"
    for c in ["growth_gap", "inflation_gap", "growth_factor", "inflation_factor", "quadrant", "hmm_filtered",
              "hmm_smoothed_expost", "hmm_walkforward", "available_at"] + [f"p_{r}" for r in REGIMES]:
        assert c in pub.labels.columns, c
    assert pub.summary["current"]["regime"] in REGIMES and "run" in pub.summary
    assert {"name", "value", "op", "threshold", "passed", "rationale", "known_failure"} <= set(pub.acceptance.columns)
    assert pub.regime_returns is not None and pub.regime_returns.index.names == ["asset", "regime"]
    assert set(pub.corr) == set(REGIMES)
    assert pub.backtest_returns is not None and "PIT_MaxSharpe" in pub.backtest_returns.columns
    assert pub.portfolio_weights is not None and pub.portfolio_weights.columns.nlevels == 2
    assert set(pub.figures) == set(P.FIGURES) | set(P.DOC_FIGURES)
    # doc_placebo is skipped by --skip-placebo; fig12_rt_vintage is a real-time-vintage
    # comparator that run.py does not yet produce (it needs an archive of dated FRED-MD
    # vintages, not the single pinned fixture this driver run uses) — load_published must
    # still tolerate its png being absent (spec: missing asset-stage files are None).
    non_placebo = {k: v for k, v in pub.figures.items() if k not in ("doc_placebo", "fig12_rt_vintage")}
    assert all(p is not None and p.exists() for p in non_placebo.values())
    assert pub.figures["doc_placebo"] is None       # --skip-placebo drops backtest_placebo's null draws


def test_load_published_without_assets(published_dir, tmp_path):
    out, figs = published_dir
    o2 = tmp_path / "out"; o2.mkdir()
    for f in ["regime_labels.csv", "summary.json", "acceptance.csv"]:
        (o2 / f).write_bytes((out / f).read_bytes())
    pub = P.load_published(o2, tmp_path / "nofigs")
    assert pub.regime_returns is None and pub.corr == {} and pub.backtest_returns is None
    assert all(v is None for v in pub.figures.values())


def test_missing_summary_raises(tmp_path):
    with pytest.raises(P.PublishedMissing):
        P.load_published(tmp_path, tmp_path)
    assert P.published_mtime(tmp_path) == 0.0


def test_default_vintage_is_previous_month():
    assert P.default_vintage(date(2026, 9, 4)) == "2026-08"
    assert P.default_vintage(date(2026, 1, 15)) == "2025-12"


def test_last_check_round_trips(tmp_path):
    out = tmp_path / "output"; out.mkdir()
    P.write_last_check(out, "2026-09-14T07:00:11Z", "2026-08")
    got = P.read_last_check(out)
    assert got == {"checked_at": "2026-09-14T07:00:11Z", "latest_vintage": "2026-08"}


def test_last_check_lives_beside_the_output_dir(tmp_path):
    """publish() swaps output/ by rename and deletes whatever the new run did not write, so a
    heartbeat inside it would vanish on every publish. It belongs on the volume, next to it."""
    out = tmp_path / "output"; out.mkdir()
    P.write_last_check(out, "2026-09-14T07:00:11Z", "2026-08")
    assert (tmp_path / P.LAST_CHECK).exists()
    assert list(out.iterdir()) == []


def test_last_check_is_none_when_never_written(tmp_path):
    assert P.read_last_check(tmp_path / "output") is None


def test_last_check_is_none_when_unreadable(tmp_path):
    """A half-written heartbeat must not take the dashboard down: it is a nicety, not a contract."""
    out = tmp_path / "output"; out.mkdir()
    (tmp_path / P.LAST_CHECK).write_text("{not json", encoding="utf-8")
    assert P.read_last_check(out) is None


def test_last_check_records_a_check_that_found_nothing(tmp_path):
    """An unreachable source is still a check; the vintage is null rather than the entry missing."""
    out = tmp_path / "output"; out.mkdir()
    P.write_last_check(out, "2026-09-14T07:00:11Z", None)
    assert P.read_last_check(out)["latest_vintage"] is None


def test_refresh_command_shape(tmp_path):
    cmd = P.refresh_command("py", "run.py", "2026-08", tmp_path / "o", tmp_path / "f", tmp_path / "r.parquet")
    assert cmd[:3] == ["py", "run.py", "--vintage"] and cmd[3] == "2026-08"
    for flag in ["--out-dir", "--figs-dir", "--returns-cache", "--refresh-returns"]:
        assert flag in cmd
    assert "--vintage-archive" not in cmd                      # no archive configured
    cmd = P.refresh_command("py", "run.py", "2026-08", tmp_path / "o", tmp_path / "f", tmp_path / "r.parquet",
                            tmp_path / "v")
    assert cmd[-2:] == ["--vintage-archive", str(tmp_path / "v")]


def test_run_refresh_lock_and_tail(tmp_path):
    lock = tmp_path / "refresh.lock"
    ok, tail = P.run_refresh([sys.executable, "-c", "print('hello'); print('world')"], str(tmp_path), lock)
    assert ok and "world" in tail and not lock.exists()
    ok, tail = P.run_refresh([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"], str(tmp_path), lock)
    assert not ok and "boom" in tail and "exit code 3" in tail and not lock.exists()
    lock.unlink(missing_ok=True)
    # U+2192 has no cp1252 encoding, so a child writing it proves the UTF-8 pinning end to end.
    ok, tail = P.run_refresh([sys.executable, "-c", "print('§ — a → b ok')"], str(tmp_path), lock)
    assert ok and "§ — a → b ok" in tail and not lock.exists()


def test_run_refresh_fresh_lock_blocks(tmp_path):
    lock = tmp_path / "refresh.lock"
    lock.write_text("pid=1")
    ok, tail = P.run_refresh([sys.executable, "-c", "print('ran')"], str(tmp_path), lock, timeout_s=60)
    assert not ok and "already running" in tail and "ran" not in tail
    assert lock.exists()          # someone else's live lock is never removed


def test_run_refresh_reclaims_a_stale_lock(tmp_path):
    lock = tmp_path / "refresh.lock"
    lock.write_text("pid=1")
    timeout_s = 60
    old = time.time() - 2 * timeout_s
    os.utime(lock, (old, old))
    ok, tail = P.run_refresh([sys.executable, "-c", "print('ran')"], str(tmp_path), lock, timeout_s=timeout_s)
    assert ok and "ran" in tail
    assert "stale lock" in tail          # the tail says the lock was reclaimed
    assert not lock.exists()


def test_run_refresh_writes_pid_and_timestamp(tmp_path):
    lock = tmp_path / "refresh.lock"
    body = tmp_path / "body.txt"
    ok, _ = P.run_refresh([sys.executable, "-c",
                           f"import pathlib; pathlib.Path({str(body)!r}).write_text(pathlib.Path({str(lock)!r}).read_text())"],
                          str(tmp_path), lock)
    assert ok and not lock.exists()
    held = body.read_text()
    assert f"pid={os.getpid()}" in held and "started=" in held


def test_redraw_doc_figures_restores_pngs_and_summary_key(published_dir):
    import json
    from regime_v2 import docfigs
    out, figs = published_dir
    # A volume whose doc PNGs are gone and whose summary predates the doc_figures key.
    for name in docfigs.DOC_FIGURES:
        (figs / f"{name}.png").unlink(missing_ok=True)
    sp = out / P.FILES["summary"]
    s = json.loads(sp.read_text(encoding="utf-8"))
    s.pop("doc_figures", None)
    sp.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")

    result = P.redraw_doc_figures(out, figs)

    assert set(result) == set(docfigs.DOC_FIGURES)
    assert result["doc_pipeline"] == "doc_pipeline.png" and (figs / "doc_pipeline.png").exists()
    assert result["doc_placebo"] is None          # published_dir runs with --skip-placebo
    s2 = json.loads(sp.read_text(encoding="utf-8"))
    assert s2["doc_figures"] == result
    assert s2["run"] == s["run"]                  # nothing else in the summary was touched


def test_redraw_doc_figures_refuses_an_empty_dir(tmp_path):
    with pytest.raises(P.PublishedMissing):
        P.redraw_doc_figures(tmp_path / "out", tmp_path / "figs")
