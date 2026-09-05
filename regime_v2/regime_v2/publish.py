"""The published-output contract for the dashboard, and the refresh runner.

Everything the dashboard reads comes through here; no other module opens
files under output/ or figs/. Missing asset-stage files are None, not errors,
because the asset stage is allowed to skip (spec §6 Stage 6).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .docfigs import DOC_FIGURES

FILES = {"labels": "regime_labels.csv", "summary": "summary.json", "acceptance": "acceptance.csv",
         "regime_returns": "regime_returns.csv", "backtest_returns": "backtest_returns.csv",
         "portfolio_weights": "portfolio_weights.csv"}
CORR_GLOB = "regime_corr_*.csv"
FIGURES = ["fig1_factors_gaps", "fig2_regime_timeline", "fig3_state_space", "fig4_hmm_probabilities", "fig5_revisions",
           "fig6_classifier_comparison", "fig7_walkforward", "fig8_regime_returns", "fig9_mixture_6040",
           "fig10_backtest_wealth", "fig11_pit_weights"]


class PublishedMissing(Exception):
    """No published run in out_dir (summary.json absent)."""


@dataclass
class Published:
    out_dir: Path
    figs_dir: Path
    labels: pd.DataFrame
    summary: dict
    acceptance: pd.DataFrame
    regime_returns: pd.DataFrame | None
    corr: dict[str, pd.DataFrame]
    backtest_returns: pd.DataFrame | None
    portfolio_weights: pd.DataFrame | None
    figures: dict[str, Path | None]


def _opt(path: Path, reader):
    return reader(path) if path.exists() else None


def load_published(out_dir, figs_dir) -> Published:
    out_dir, figs_dir = Path(out_dir), Path(figs_dir)
    summary_path = out_dir / FILES["summary"]
    if not summary_path.exists():
        raise PublishedMissing(f"no published run in {out_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(out_dir / FILES["labels"], index_col=0, parse_dates=[0, 1])
    labels.index.name = "date"
    acceptance = pd.read_csv(out_dir / FILES["acceptance"])
    regime_returns = _opt(out_dir / FILES["regime_returns"], lambda p: pd.read_csv(p, index_col=[0, 1]))
    corr = {Path(p).stem.replace("regime_corr_", ""): pd.read_csv(p, index_col=0)
            for p in sorted(glob.glob(str(out_dir / CORR_GLOB)))}
    backtest_returns = _opt(out_dir / FILES["backtest_returns"], lambda p: pd.read_csv(p, index_col=0, parse_dates=True))
    portfolio_weights = _opt(out_dir / FILES["portfolio_weights"], lambda p: pd.read_csv(p, index_col=0, header=[0, 1], parse_dates=True))
    figures = {n: (figs_dir / f"{n}.png" if (figs_dir / f"{n}.png").exists() else None)
              for n in FIGURES + DOC_FIGURES}
    return Published(out_dir, figs_dir, labels, summary, acceptance, regime_returns, corr, backtest_returns,
                     portfolio_weights, figures)


def published_mtime(out_dir) -> float:
    p = Path(out_dir) / FILES["summary"]
    return p.stat().st_mtime if p.exists() else 0.0


def default_vintage(today: date | None = None) -> str:
    today = today or date.today()
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def refresh_command(python, run_py, vintage, out_dir, figs_dir, returns_cache) -> list[str]:
    return [str(python), str(run_py), "--vintage", str(vintage), "--out-dir", str(out_dir), "--figs-dir", str(figs_dir),
            "--returns-cache", str(returns_cache), "--refresh-returns"]


def _acquire_lock(lock_path: Path, timeout_s: int) -> tuple[bool, str]:
    """Take the refresh lock atomically. Returns (acquired, note).

    `os.open(O_CREAT | O_EXCL)` is the whole point: an `exists()` test followed
    by a write lets two sessions that check at the same moment both conclude the
    lock is free and both start the engine. A lock older than `timeout_s` cannot
    belong to a live run — the subprocess itself is killed at `timeout_s` — so it
    is treated as abandoned (a container restart, a killed session), removed, and
    reported in the returned tail rather than blocking refreshes forever.
    """
    note = ""
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue                       # the holder released it between our open and stat
            if age <= timeout_s:
                return False, (f"A refresh is already running (lock {lock_path.name} present, "
                               f"{age / 60:.0f} min old).")
            lock_path.unlink(missing_ok=True)
            note = (f"[stale lock] {lock_path.name} was {age / 60:.0f} min old (limit {timeout_s} s) and cannot "
                    "belong to a live run; removed it and continued.\n")
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()}\nstarted={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
        return True, note
    return False, f"Could not take the refresh lock {lock_path.name}; another session holds it."


def run_refresh(cmd: list[str], cwd: str, lock_path, timeout_s: int = 1800) -> tuple[bool, str]:
    """Run the engine once under a lock file. Returns (ok, log tail). Never raises."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired, note = _acquire_lock(lock_path, timeout_s)
    if not acquired:
        return False, note
    # Pin the subprocess's text encoding: on Windows the default locale encoding (cp1252)
    # would mojibake the engine's UTF-8 output (em-dashes, section signs) when decoding
    # stdout/stderr, and PYTHONIOENCODING makes the child itself write UTF-8 on every
    # platform regardless of its own locale.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env=child_env, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, note + f"Refresh timed out after {timeout_s} s."
        log = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            log += f"\n[exit code {proc.returncode}]"
        return proc.returncode == 0, note + log[-4000:]
    finally:
        lock_path.unlink(missing_ok=True)
