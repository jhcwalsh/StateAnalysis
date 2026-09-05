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
    non_placebo = {k: v for k, v in pub.figures.items() if k != "doc_placebo"}
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


def test_refresh_command_shape(tmp_path):
    cmd = P.refresh_command("py", "run.py", "2026-08", tmp_path / "o", tmp_path / "f", tmp_path / "r.parquet")
    assert cmd[:3] == ["py", "run.py", "--vintage"] and cmd[3] == "2026-08"
    for flag in ["--out-dir", "--figs-dir", "--returns-cache", "--refresh-returns"]:
        assert flag in cmd


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
