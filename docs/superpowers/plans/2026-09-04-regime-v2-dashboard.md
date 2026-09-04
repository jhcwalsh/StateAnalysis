# regime_v2 Stage 7: dashboard on the engine, deployed to states.lazyeconomist.com

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-point the Streamlit dashboard at `regime_v2/output/`, retire the notebook pipeline, and add the Docker deployment that serves `states.lazyeconomist.com` from `~/apps/states` on the Mac Mini.

**Architecture:** The engine (`regime_v2/run.py`) already publishes everything except a `current` block, run metadata and factor levels; Task 1 adds those. A small loader `regime_v2/regime_v2/publish.py` becomes the only place the published file names live and also owns the refresh subprocess with a lock file. `app.py` is rewritten on that loader with the spec's four regime names and the engine's colours; `regime_core.py`, `ui_io.py`, `ui_data/`, the notebook and the root artefacts are deleted. Deployment is a root `Dockerfile` + `docker-compose.yml` following the MacMiniHosting runbook (one app dir, `restart: unless-stopped`, published outputs on a named volume, Cloudflare Tunnel route already created by the user).

**Tech Stack:** Python 3.12, pandas, matplotlib, Streamlit ≥ 1.36 (`streamlit.testing.v1.AppTest` for tests), pytest, Docker (python:3.12-slim) under OrbStack on the Mini.

**Spec:** `docs/SPEC.md` §6 Stage 7, §9 Q7 (answered 2026-09-04), §12 Deployment, D10 (causal hysteresis), D11 (`available_at`).

## Global Constraints

- Regime names everywhere are exactly `Contraction`, `Goldilocks`, `Overheating`, `Stagflation` (`regime_v2.regimes.REGIMES`) with colours from `regime_v2.regimes.COLORS`. After this plan `grep -rE "Q1_|Q2_|Q3_|Q4_" --include=*.py .` returns nothing outside `.venv`.
- The dashboard reads only `regime_v2/output/` and `regime_v2/figs/` (or the directories named by `REGIME_OUTPUT_DIR` / `REGIME_FIGS_DIR`), through `regime_v2.publish`. No other module may open those files.
- The Refresh button runs `python regime_v2/run.py --vintage YYYY-MM …` under a lock file; a failed run must leave the previously published outputs on screen (the driver's staged publish guarantees the files; the app must not clear its cache on failure).
- Tests never touch the network (the autouse `_no_network` fixture in `regime_v2/tests/conftest.py` stays; the root `tests/conftest.py` gets the same guard) and never require Docker.
- Port 8505 and host `states.lazyeconomist.com` appear in exactly one file each: `docker-compose.yml` (port, as `STREAMLIT_SERVER_PORT`) and `docs/SPEC.md` §12 / `README.md` (host).
- Python interpreter for every command: `C:\Users\james\PycharmProjects\StateAnalysis\.venv\Scripts\python.exe`. Engine tests run from `regime_v2/`; dashboard tests run from the repo root.
- Commit messages end with the two attribution lines the session uses.

## File Structure

- `regime_v2/run.py` (modify): `labels_frame` extra columns `growth_factor`, `inflation_factor`; `summary["run"]`, `summary["current"]`.
- `regime_v2/regime_v2/publish.py` (create): file contract, `Published` dataclass, `load_published`, `published_mtime`, `default_vintage`, `refresh_command`, `run_refresh`.
- `regime_v2/tests/test_publish.py` (create), `regime_v2/tests/conftest.py` (modify: session-scoped `published_dir` fixture), `regime_v2/tests/test_run.py` (modify: assertions for the new fields).
- `app.py` (rewrite), `tests/conftest.py` (rewrite), `tests/test_app_smoke.py` (rewrite).
- Delete: `Macro_Regime_Analysis.ipynb`, `main.py`, `regime_core.py`, `ui_io.py`, `tests/test_regime_core.py`, `tests/test_ui_io.py`, `regime_v2/tests/test_regimes_quadrants.py` gains the two hysteresis tests worth keeping.
- `requirements.txt` (rewrite: one file for engine + dashboard; `regime_v2/requirements.txt` becomes a one-line pointer), `.gitignore` (modify), `README.md` (rewrite).
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docker/entrypoint.sh` (create), `docs/SPEC.md` §12 checklist (modify).

---

### Task 1: The engine publishes `current`, `run` and the factor levels

**Files:**
- Modify: `regime_v2/run.py` (the `parts = [...]` block and the summary write in `main`)
- Modify: `regime_v2/tests/test_run.py` (append)

**Interfaces:**
- Produces in `regime_labels.csv`: new columns `growth_factor`, `inflation_factor` (the composite factor levels `res.growth_factor["factor"]`, `res.inflation_factor["factor"]`, reindexed to the labels index).
- Produces in `summary.json`: `run = {"timestamp": ISO-8601 UTC string, "vintage": basename of the input CSV, "asof": "YYYY-MM-DD" of the last labelled month, "engine": "regime_v2", "label_source": <existing label_source string>}` and `current = {"month": "YYYY-MM", "regime": <label from the history column at the last month>, "quadrant": <quadrant_walkforward or quadrant at the last month>, "growth_gap": float, "inflation_gap": float, "probs": {regime: float for the four regimes}}`.

- [ ] **Step 1: Write the failing test**

Append to `regime_v2/tests/test_run.py`:
```python
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
```
(`os`, `json`, `pd`, `pytest`, `R` and `runmod` are already imported at the top of that file; add `import os` if it is missing.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_run.py -q -k current_and_run`
Expected: FAIL with `KeyError: 'run'` (or the factor-column assertion).

- [ ] **Step 3: Implement**

In `regime_v2/run.py`, add `from datetime import datetime, timezone` to the imports. In `main`, where `parts = [...]` is built, add the factor levels:
```python
    parts = [res.growth_factor["factor"].rename("growth_factor"),
             res.inflation_factor["factor"].rename("inflation_factor"),
             pd.Series(pd.NA, index=res.hmm.labels_filtered.index, name="hmm_walkforward") if wf is None else wf.labels_rt,
             ...  # the existing entries follow unchanged
```
Then, immediately after `labels_df = labels_frame(res, extra=parts)` (the variable Task 7 of the previous plan introduced) and before the summary is written to staging, add:
```python
    last = labels_df.index[-1]
    hist_col = "hmm_walkforward" if wf is not None else "hmm_filtered"
    quad_col = "quadrant_walkforward" if wf is not None else "quadrant"
    summary["run"] = {"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "vintage": os.path.basename(str(path)), "asof": str(last.date()),
                      "engine": "regime_v2", "label_source": label_source}
    summary["current"] = {"month": last.strftime("%Y-%m"),
                          "regime": str(labels_df.loc[last, hist_col]),
                          "quadrant": str(labels_df.loc[last, quad_col]),
                          "growth_gap": float(labels_df.loc[last, "growth_gap"]),
                          "inflation_gap": float(labels_df.loc[last, "inflation_gap"]),
                          "probs": {r: float(labels_df.loc[last, f"p_{r}"]) for r in R.REGIMES}}
```
`path` is the resolved vintage CSV path used by `main` (check its name in the file; use whatever variable holds the CSV path). `label_source` already exists in `main`. Make sure this block runs before *both* summary writes (the staged write and the post-asset-stage rewrite reuse the same `summary` dict, so placing it before the staged write is sufficient).

- [ ] **Step 4: Run the tests**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_run.py -q`
Expected: all pass (the asset-stage tests still pass because they read `summary["assets"]`, untouched).

- [ ] **Step 5: Commit**

```bash
git add regime_v2/run.py regime_v2/tests/test_run.py
git commit -m "stage7: publish current state, run metadata and factor levels"
```

---

### Task 2: `publish.py` — the loader and the refresh runner

**Files:**
- Create: `regime_v2/regime_v2/publish.py`
- Modify: `regime_v2/tests/conftest.py` (append a session-scoped fixture)
- Create: `regime_v2/tests/test_publish.py`

**Interfaces:**
- Produces:
```python
FILES = {"labels": "regime_labels.csv", "summary": "summary.json", "acceptance": "acceptance.csv",
         "regime_returns": "regime_returns.csv", "backtest_returns": "backtest_returns.csv",
         "portfolio_weights": "portfolio_weights.csv"}
CORR_GLOB = "regime_corr_*.csv"
FIGURES = ["fig1_factors_gaps", "fig2_regime_timeline", "fig3_state_space", "fig4_hmm_probabilities", "fig5_revisions",
           "fig6_classifier_comparison", "fig7_walkforward", "fig8_regime_returns", "fig9_mixture_6040",
           "fig10_backtest_wealth", "fig11_pit_weights"]
class PublishedMissing(Exception)
@dataclass class Published: out_dir: Path; figs_dir: Path; labels: pd.DataFrame; summary: dict; acceptance: pd.DataFrame;
    regime_returns: pd.DataFrame | None; corr: dict[str, pd.DataFrame]; backtest_returns: pd.DataFrame | None;
    portfolio_weights: pd.DataFrame | None; figures: dict[str, Path | None]
def load_published(out_dir, figs_dir) -> Published          # PublishedMissing if summary.json absent
def published_mtime(out_dir) -> float                        # mtime of summary.json, 0 if absent
def default_vintage(today: date | None = None) -> str        # previous calendar month, "YYYY-MM"
def refresh_command(python, run_py, vintage, out_dir, figs_dir, returns_cache) -> list[str]
def run_refresh(cmd, cwd, lock_path, timeout_s=1800) -> tuple[bool, str]   # (ok, log tail ≤ 4000 chars); (False, "A refresh is already running…") if the lock exists
```
- Consumes: the Task 1 fields.

- [ ] **Step 1: Session fixture and failing tests**

Append to `regime_v2/tests/conftest.py`:
```python
@pytest.fixture(scope="session")
def published_dir(tmp_path_factory):
    """One real driver run per session, with the asset stage on the pinned fixture cache.

    --wf-step 24 gives genuine (coarse) walk-forward labels cheaply; because a 24-month
    step cannot sample both the GFC and COVID acceptance windows, the gate is stubbed
    exactly as test_run.py does. Thresholds are tested for real in test_acceptance.py.
    """
    import run as runmod
    from regime_v2 import acceptance
    root = tmp_path_factory.mktemp("published")
    out, figs = root / "output", root / "figs"
    mp = pytest.MonkeyPatch()
    mp.setattr(acceptance, "all_passed", lambda table: True)
    try:
        rc = runmod.main([str(HERE / "data" / "fredmd_2026-07.csv"), "--wf-step", "24", "--skip-robustness",
                          "--skip-expanding", "--skip-placebo", "--returns-cache", str(HERE / "data" / "returns_fixture.parquet"),
                          "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(root / "README.md")])
    finally:
        mp.undo()
    assert rc == 0
    return out, figs
```
(`HERE` must be the `regime_v2/` directory; check how the existing conftest computes paths and reuse its variable, and make sure `run` is importable — the existing tests already import it as `runmod`, so mirror that import.)

Create `regime_v2/tests/test_publish.py`:
```python
import json
import os
import subprocess
import sys
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
    assert set(pub.figures) == set(P.FIGURES) and all(p is not None and p.exists() for p in pub.figures.values())


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
    for flag in ["--out-dir", "--figs-dir", "--returns-cache"]:
        assert flag in cmd


def test_run_refresh_lock_and_tail(tmp_path):
    lock = tmp_path / "refresh.lock"
    ok, tail = P.run_refresh([sys.executable, "-c", "print('hello'); print('world')"], str(tmp_path), lock)
    assert ok and "world" in tail and not lock.exists()
    ok, tail = P.run_refresh([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"], str(tmp_path), lock)
    assert not ok and "boom" in tail and "exit code 3" in tail and not lock.exists()
    lock.write_text("pid")
    ok, tail = P.run_refresh([sys.executable, "-c", "pass"], str(tmp_path), lock)
    assert not ok and "already running" in tail and lock.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_publish.py -q`
Expected: FAIL on `ImportError: cannot import name 'publish'`.

- [ ] **Step 3: Implement `publish.py`**

```python
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
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

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
    figures = {n: (figs_dir / f"{n}.png" if (figs_dir / f"{n}.png").exists() else None) for n in FIGURES}
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
            "--returns-cache", str(returns_cache)]


def run_refresh(cmd: list[str], cwd: str, lock_path, timeout_s: int = 1800) -> tuple[bool, str]:
    """Run the engine once under a lock file. Returns (ok, log tail). Never raises."""
    lock_path = Path(lock_path)
    if lock_path.exists():
        return False, f"A refresh is already running (lock {lock_path.name} present)."
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"Refresh timed out after {timeout_s} s."
        log = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            log += f"\n[exit code {proc.returncode}]"
        return proc.returncode == 0, log[-4000:]
    finally:
        lock_path.unlink(missing_ok=True)
```
Note on `parse_dates=[0, 1]`: column 0 is `date` (the index), column 1 is `available_at`; if pandas complains about parsing the index column this way, read with `parse_dates=["available_at"]` and `index_col="date"` and convert the index with `pd.to_datetime` explicitly.

- [ ] **Step 4: Run the tests**

Run: `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_publish.py -q`
Expected: 6 passed (the session fixture takes ~60–90 s the first time).

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/publish.py regime_v2/tests/test_publish.py regime_v2/tests/conftest.py
git commit -m "stage7: publish.py loader for the published-output contract and the refresh runner"
```

---

### Task 3: Rewrite `app.py` on the engine's outputs

**Files:**
- Rewrite: `app.py`
- Rewrite: `tests/conftest.py`, `tests/test_app_smoke.py`
- Delete: `regime_core.py`, `ui_io.py`, `tests/test_regime_core.py`, `tests/test_ui_io.py`
- Modify: `regime_v2/tests/test_regimes_quadrants.py` (append two hysteresis tests)

**Interfaces:**
- Consumes: `regime_v2.publish` (Task 2), `regime_v2.regimes` (`REGIMES`, `COLORS`, `quadrant_labels`). Run lengths are computed inline (`regimes.run_lengths` returns a per-regime mean, not the runs).
- Produces: environment variables `REGIME_OUTPUT_DIR`, `REGIME_FIGS_DIR`, `REGIME_RETURNS_CACHE` (defaults `regime_v2/output`, `regime_v2/figs`, `regime_v2/data/returns_yfinance.parquet`, resolved against the repo root), and `REGIME_PYTHON` (default `sys.executable`) — Task 5's compose file sets the first three.

- [ ] **Step 1: Port the hysteresis tests, delete the legacy modules, write the failing app tests**

Append to `regime_v2/tests/test_regimes_quadrants.py` (these two come from the deleted `tests/test_regime_core.py`; skip any that already exist there under another name):
```python
def test_hysteresis_reduces_switches_on_noisy_series():
    v = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, -0.30, -0.10, 0.10, -0.10, 0.30, 0.05]
    s = pd.Series(v, index=pd.date_range("2020-01-01", periods=len(v), freq="MS"))
    switches = lambda x: int((x.diff() != 0).sum()) - 1
    assert switches(R.hysteretic_sign(s, 0.0)) == 8
    assert switches(R.hysteretic_sign(s, 0.25)) == 2


def test_hysteresis_is_causal_prefix_property():
    v = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, -0.30, -0.10, 0.10, -0.10, 0.30, 0.05]
    s = pd.Series(v, index=pd.date_range("2020-01-01", periods=len(v), freq="MS"))
    full, prefix = R.hysteretic_sign(s, 0.25), R.hysteretic_sign(s.iloc[:7], 0.25)
    assert list(full.iloc[:7]) == list(prefix)
```
(`R` is how that file imports `regime_v2.regimes`; match its import name. If `hysteretic_sign`'s initial-state convention differs from the legacy one and the switch counts come out different, report the counts you measured and keep the assertion that θ = 0.25 gives strictly fewer switches than θ = 0.)

Delete with git: `git rm regime_core.py ui_io.py tests/test_regime_core.py tests/test_ui_io.py`.

Rewrite `tests/conftest.py`:
```python
"""Dashboard tests: one real engine run per session, no network, no Docker."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "regime_v2"
sys.path.insert(0, str(ENGINE))   # makes `regime_v2` (package) and `run` importable, as app.py does


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    from regime_v2 import assets
    def boom(*a, **k): raise RuntimeError("network disabled in tests")
    monkeypatch.setattr(assets, "_download_yfinance", boom)


@pytest.fixture(scope="session")
def published_dir(tmp_path_factory):
    import run as runmod
    from regime_v2 import acceptance
    root = tmp_path_factory.mktemp("published")
    out, figs = root / "output", root / "figs"
    mp = pytest.MonkeyPatch()
    mp.setattr(acceptance, "all_passed", lambda table: True)   # see regime_v2/tests/conftest.py for why
    try:
        rc = runmod.main([str(ENGINE / "data" / "fredmd_2026-07.csv"), "--wf-step", "24", "--skip-robustness",
                          "--skip-expanding", "--skip-placebo", "--returns-cache", str(ENGINE / "data" / "returns_fixture.parquet"),
                          "--out-dir", str(out), "--figs-dir", str(figs), "--data-sheet", str(root / "README.md")])
    finally:
        mp.undo()
    assert rc == 0
    return out, figs
```

Rewrite `tests/test_app_smoke.py`:
```python
from streamlit.testing.v1 import AppTest

TABS = ["Factor gaps", "Factor levels", "Probabilities", "State space", "Regime returns", "Correlations",
        "Portfolios", "Backtest", "Acceptance", "Figures"]


def _run(monkeypatch, out, figs):
    monkeypatch.setenv("REGIME_OUTPUT_DIR", str(out))
    monkeypatch.setenv("REGIME_FIGS_DIR", str(figs))
    return AppTest.from_file("app.py", default_timeout=120).run()


def test_app_renders_status(published_dir, monkeypatch):
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    import json
    s = json.loads((out / "summary.json").read_text())
    page = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
    assert s["current"]["regime"] in page
    assert s["current"]["month"] in page
    assert "walk-forward" in page.lower()


def test_app_tabs_and_asset_content(published_dir, monkeypatch):
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    labels = [t.label for t in at.tabs]
    for t in TABS:
        assert t in labels, t
    page = " ".join(md.value for md in at.markdown)
    assert "Static_6040" in page or "60/40" in page      # benchmark shown beside the PIT Sharpe
    assert "look-ahead" in page.lower()


def test_app_without_asset_stage(published_dir, monkeypatch, tmp_path):
    out, figs = published_dir
    o2 = tmp_path / "out"; o2.mkdir()
    for f in ["regime_labels.csv", "summary.json", "acceptance.csv"]:
        (o2 / f).write_bytes((out / f).read_bytes())
    at = _run(monkeypatch, o2, tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown) + " ".join(i.value for i in at.info)
    assert "asset stage" in page.lower()                 # the tabs explain what is missing


def test_app_empty_state(tmp_path, monkeypatch):
    at = _run(monkeypatch, tmp_path / "nowhere", tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "No published run" in page
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `../.venv/Scripts/python.exe -m pytest tests -q` from the repo root.
Expected: the four app tests FAIL (old `app.py` imports `ui_io`, which is gone).

- [ ] **Step 3: Rewrite `app.py`**

```python
"""Macro regime dashboard — a thin Streamlit viewer over regime_v2/output/ (spec §12).

Reads only through regime_v2.publish. Regime names and colours come from
regime_v2.regimes. The Refresh button runs the engine; a failed run leaves the
last published outputs on screen because the engine publishes atomically.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "regime_v2"
sys.path.insert(0, str(ENGINE))
from regime_v2 import publish, regimes as R   # noqa: E402  (path set above)


def _dir(env, default):
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


OUT_DIR = _dir("REGIME_OUTPUT_DIR", "regime_v2/output")
FIGS_DIR = _dir("REGIME_FIGS_DIR", "regime_v2/figs")
RETURNS_CACHE = _dir("REGIME_RETURNS_CACHE", "regime_v2/data/returns_yfinance.parquet")
PYTHON = os.environ.get("REGIME_PYTHON", sys.executable)
LOCK = OUT_DIR.parent / ".refresh.lock"

st.set_page_config(page_title="Macro Regime Dashboard", layout="wide")

# ---- chart style: theme-matched ink, transparent surfaces --------------------
FIG_W = 12
try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
INK = "#FAFAFA" if DARK else "#31333F"
INK_MUTED = "#A3A8B4" if DARK else "#808495"
SURFACE = "#0E1117" if DARK else "#FFFFFF"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": INK, "axes.titlesize": 10, "axes.edgecolor": INK_MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": INK_MUTED, "grid.alpha": 0.15 if DARK else 0.25, "grid.linewidth": 0.5,
})
GROWTH_C, INFL_C = "#2C7FB8", "#D95F0E"
CLIP_NOTE = " Extreme COVID-2020 observations lie beyond the visible range."


def _fig(height, width=FIG_W):
    return plt.subplots(figsize=(width, height))


@st.cache_data
def _load(mtime, out_dir, figs_dir):
    return publish.load_published(out_dir, figs_dir)


def _refresh_ui(button_label):
    vintage = st.text_input("FRED-MD vintage (YYYY-MM)", publish.default_vintage(),
                            help="The vintage file to download from the St. Louis Fed; previous month by default.")
    if st.button(button_label):
        cmd = publish.refresh_command(PYTHON, ENGINE / "run.py", vintage, OUT_DIR, FIGS_DIR, RETURNS_CACHE)
        with st.spinner("Running the engine (≈3 min: download, walk-forward, asset stage)…"):
            ok, tail = publish.run_refresh(cmd, str(ENGINE), LOCK)
        if ok:
            st.success("Refresh complete."); st.cache_data.clear(); st.rerun()
        st.error("Refresh failed — the previously published run is still shown. Log tail:")
        st.code(tail)


try:
    pub = _load(publish.published_mtime(OUT_DIR), str(OUT_DIR), str(FIGS_DIR))
except publish.PublishedMissing:
    st.title("Macro Regime Dashboard")
    st.markdown("**No published run found.** Run the engine once to publish `output/` and `figs/`.")
    _refresh_ui("Run the engine now")
    st.stop()

S, cur, run = pub.summary, pub.summary["current"], pub.summary["run"]
lab = pub.labels
theta_run = float(S["params"]["theta"])
hist_col = "hmm_walkforward" if lab["hmm_walkforward"].notna().any() else "hmm_filtered"

# ---------------- Zone 1: status header ----------------
st.title("Macro Regime Dashboard")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"<div style='background:{R.COLORS[cur['regime']]};color:white;padding:1.2em;border-radius:8px;"
        f"font-size:1.5em;font-weight:bold'>{cur['regime']} · {cur['month']}</div>", unsafe_allow_html=True)
    st.markdown(f"Growth gap **{cur['growth_gap']:+.2f}** · Inflation gap **{cur['inflation_gap']:+.2f}** "
                f"· quadrant rule says **{cur['quadrant']}** (θ = {theta_run:.2f})")
with c2:
    probs = pd.Series(cur["probs"]).reindex(R.REGIMES).fillna(0.0)
    fig, ax = _fig(1.6, width=7)
    lefts = probs.cumsum().shift(fill_value=0).values
    ax.barh([0] * len(R.REGIMES), probs.values, left=lefts, color=[R.COLORS[k] for k in R.REGIMES], height=0.5,
            edgecolor=SURFACE, linewidth=1.5)
    for k, p, l in zip(R.REGIMES, probs.values, lefts):
        if p > 0.08:
            ax.text(l + p / 2, 0, f"{k} {p:.0%}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1); ax.axis("off")
    st.pyplot(fig, clear_figure=True)

gate = "all acceptance tests passed" if S["acceptance_all_passed"] else "acceptance FAILED"
known = S.get("acceptance_known_failures") or {}
st.caption(f"Labels: {run['label_source']} · vintage {run['vintage']} · data through {run['asof'][:7]} · run {run['timestamp'][:16]} "
           f"· {gate}" + (f" · known failures: {', '.join(known)}" if known else ""))

# ---------------- Zone 2: explore (theta) ----------------
st.header("Explore: hysteresis θ")
theta = st.slider("θ (regime persistence band)", 0.0, 1.0, theta_run, 0.05,
                  help=f"Run value: {theta_run}. Live causal relabelling of the published gaps (spec D10).")
g, p = lab["growth_gap"].dropna(), lab["inflation_gap"].dropna()
live = R.quadrant_labels(g, p, theta)
lengths = live.groupby((live != live.shift()).cumsum()).size()   # one entry per run
m1, m2, m3 = st.columns(3)
m1.metric("Avg regime duration", f"{lengths.mean():.1f} mo")
m2.metric("Regime switches", int(len(lengths) - 1))
m3.metric("Months classified", len(live))
fig, ax = _fig(1.4)
for reg in [r for r in R.REGIMES if (live == r).any()]:
    mask = (live == reg).values
    ax.bar(live.index[mask], 1, width=32, color=R.COLORS[reg], label=reg)
ax.set_yticks([]); ax.margins(x=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4)
st.pyplot(fig, clear_figure=True)
occ = live.value_counts().reindex(R.REGIMES).fillna(0).astype(int).rename_axis("Regime").to_frame("Months")
occ["% Time"] = (occ["Months"] / len(live)).map("{:.1%}".format)
st.dataframe(occ, width="stretch")
st.info(f"The HMM labels, tables and backtest below are pinned to the run's θ = {theta_run}; the slider only relabels the quadrant rule.")

# ---------------- Zone 3: results tabs ----------------
st.header("Results")
(t_gap, t_lev, t_prob, t_ss, t_ret, t_corr, t_port, t_bt, t_acc, t_fig) = st.tabs(
    ["Factor gaps", "Factor levels", "Probabilities", "State space", "Regime returns", "Correlations",
     "Portfolios", "Backtest", "Acceptance", "Figures"])
NO_ASSETS = "The asset stage did not publish (skipped: {}). Re-run the engine with network access to fill this tab."
skipped = (S.get("assets") or {}).get("skipped", "not run")


def _robust_lim(*series, floor=2.0):
    v = pd.concat([s.abs() for s in series])
    lim = 1.2 * max(floor, float(v.quantile(0.98)))
    return lim, bool(v.max() > lim)


def _lines(ax, a, b, la, lb):
    ax.plot(lab.index, lab[a], color=GROWTH_C, linewidth=1.8, label=la)
    ax.plot(lab.index, lab[b], color=INFL_C, linewidth=1.8, label=lb)
    lim, clipped = _robust_lim(lab[a].dropna(), lab[b].dropna())
    ax.set_ylim(-lim, lim); ax.margins(x=0); ax.legend(loc="upper left"); ax.grid(True)
    return clipped


with t_gap:
    fig, ax = _fig(4)
    ax.axhspan(-theta, theta, color=INK, alpha=0.07, zorder=0); ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    clipped = _lines(ax, "growth_gap", "inflation_gap", "Growth gap", "Inflation gap")
    ax.set_title(f"Classification inputs (shaded: ±θ = {theta:.2f} dead band)", loc="left")
    st.pyplot(fig, clear_figure=True)
    st.caption("One-sided trend gaps (spec D4): each point uses data up to that month only." + (CLIP_NOTE if clipped else ""))

with t_lev:
    fig, ax = _fig(4)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    clipped = _lines(ax, "growth_factor", "inflation_factor", "Growth factor", "Inflation factor")
    ax.set_title("Underlying factor levels (cumulated diffusion indices)", loc="left")
    st.pyplot(fig, clear_figure=True)
    st.caption("Composite activity and inflation factors before the gap transformation." + (CLIP_NOTE if clipped else ""))

with t_prob:
    pr = lab[[f"p_{r}" for r in R.REGIMES]].dropna().clip(lower=0)
    pr.columns = R.REGIMES
    fig, ax = _fig(4)
    ax.stackplot(pr.index, [pr[k].values for k in R.REGIMES], colors=[R.COLORS[k] for k in R.REGIMES],
                 labels=R.REGIMES, edgecolor=SURFACE, linewidth=0.8)
    ax.set_ylim(0, 1); ax.margins(x=0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4)
    st.pyplot(fig, clear_figure=True)
    st.caption(f"Constrained 4-state HMM probabilities, {run['label_source']}: causal at every month, never revised by later data.")
    with st.expander("View data"):
        st.dataframe(pr.style.format("{:.1%}"), width="stretch")

with t_ss:
    ss = pd.concat([g, p], axis=1).dropna(); ss.columns = ["growth_gap", "inflation_gap"]
    fig, ax = _fig(6, width=6.5)
    ax.axvspan(-theta, theta, color=INK, alpha=0.06, zorder=0); ax.axhspan(-theta, theta, color=INK, alpha=0.06, zorder=0)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8); ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    for reg in [r for r in R.REGIMES if (live == r).any()]:
        m = (live.reindex(ss.index) == reg).values
        ax.scatter(ss["growth_gap"][m], ss["inflation_gap"][m], s=16, color=R.COLORS[reg], edgecolors=SURFACE,
                   linewidths=0.5, alpha=0.9, label=reg)
    last = ss.iloc[-1]
    ax.scatter([last["growth_gap"]], [last["inflation_gap"]], s=130, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=5)
    ax.annotate(ss.index[-1].strftime("%Y-%m"), (last["growth_gap"], last["inflation_gap"]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Growth gap"); ax.set_ylabel("Inflation gap")
    lim, clipped = _robust_lim(ss["growth_gap"], ss["inflation_gap"])
    lim = max(lim, 1.15 * abs(last["growth_gap"]), 1.15 * abs(last["inflation_gap"]))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.11), ncol=2)
    st.pyplot(fig, clear_figure=True, width="content")
    st.caption("Each point is one month, coloured by the live θ quadrant labels from the slider. Points inside the band keep "
               "their previous regime (hysteresis)." + (CLIP_NOTE if clipped else ""))
    if pub.figures["fig3_state_space"]:
        with st.expander("Published run version (fig 3, HMM labels)"):
            st.image(str(pub.figures["fig3_state_space"]))

with t_ret:
    if pub.regime_returns is None:
        st.info(NO_ASSETS.format(skipped))
    else:
        tbl = pub.regime_returns.copy()
        st.dataframe(tbl.style.format({"ann_ret": "{:+.1%}", "ann_vol": "{:.1%}", "sharpe": "{:+.2f}", "maxdd": "{:.1%}",
                                       "hit": "{:.0%}", "se_ann_ret": "{:.1%}", "se_sharpe": "{:.2f}", "n": "{:.0f}"}), width="stretch")
        st.caption("Monthly returns of month r paired with the regime label available before r (spec D11). Standard errors "
                   "from a 12-month block bootstrap of the whole aligned panel. maxdd chains within-regime months.")
        if pub.figures["fig8_regime_returns"]:
            st.image(str(pub.figures["fig8_regime_returns"]))

with t_corr:
    if not pub.corr:
        st.info(NO_ASSETS.format(skipped))
    for reg in [r for r in R.REGIMES if r in pub.corr]:
        st.subheader(reg)
        st.dataframe(pub.corr[reg].style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")

with t_port:
    a = S.get("assets") or {}
    if a.get("skipped") is not None or "backtest" not in a:
        st.info(NO_ASSETS.format(skipped))
    else:
        for key, title in [("cost_bp_0", "No transaction costs"), ("cost_bp_10", "10 bp per unit turnover")]:
            st.subheader(title)
            perf = pd.DataFrame(a["backtest"][key]["perf"]).T
            st.dataframe(perf.style.format({"ann_ret": "{:+.1%}", "ann_vol": "{:.1%}", "sharpe": "{:+.2f}",
                                            "maxdd": "{:.1%}", "turnover": "{:.2f}"}), width="stretch")
        st.caption("PIT_* use only labels available at the decision date; Oracle uses the ex-post smoothed labels; "
                   "InSample_MaxSharpe_expost also uses full-sample moments and is not achievable. Static_6040 is the benchmark.")
        st.caption(f"Fallbacks and guards: {a['backtest']['cost_bp_0']['counters']}")
        if pub.portfolio_weights is not None and pub.figures["fig11_pit_weights"]:
            st.image(str(pub.figures["fig11_pit_weights"]))

with t_bt:
    a = S.get("assets") or {}
    if pub.backtest_returns is None or "lookahead" not in a:
        st.info(NO_ASSETS.format(skipped))
    else:
        wealth = (1 + pub.backtest_returns).cumprod()
        fig, ax = _fig(5)
        for col in wealth.columns:
            style = dict(linewidth=1.0, linestyle="--", alpha=0.7) if col.endswith("_expost") else dict(linewidth=2.0 if col.startswith("PIT") else 1.2)
            ax.plot(wealth.index, wealth[col], label=col, **style)
        ax.set_yscale("log"); ax.margins(x=0); ax.legend(); ax.grid(True)
        st.pyplot(fig, clear_figure=True)
        L, bp = a["lookahead"], a["backtest_placebo"]
        perf0 = pd.DataFrame(a["backtest"]["cost_bp_0"]["perf"]).T
        st.markdown(
            f"**Look-ahead decomposition** — in-sample Sharpe {L['insample_sharpe']:+.2f} → oracle {L['oracle_sharpe']:+.2f} "
            f"→ achievable (PIT) {L['pit_sharpe']:+.2f}; moment look-ahead {L['moment_lookahead']:+.2f}, label look-ahead "
            f"{L['label_lookahead']:+.2f}. Static_6040 over the same window: {perf0.loc['Static_6040', 'sharpe']:+.2f}.")
        if bp:
            st.markdown(f"**Backtest placebo** — the real PIT Sharpe sits at the {bp['percentile']:.0f}th percentile of {bp['n']} "
                        "run-preserving label shuffles; below 50 means the real labels underperform the median shuffle.")
        sp = a.get("sharpe_spread_placebo")
        if sp:
            st.markdown(f"**Sharpe-spread placebo** — max-minus-min regime Sharpe of 60/40 at the {sp['percentile']:.0f}th percentile.")
        gs = a.get("growth_share_6040")
        if gs:
            st.caption(f"Growth share of the 60/40 regime regression: {gs['growth_share']:.0%} of an R² of {gs['r2']:.3f} "
                       f"(n = {gs['n']}); with an R² this small the split is not informative.")

with t_acc:
    acc = pub.acceptance.copy()
    acc["status"] = acc.apply(lambda r: "report" if r["op"] == "report" else ("known failure" if r["known_failure"] else ("pass" if r["passed"] else "FAIL")), axis=1)
    st.dataframe(acc[["name", "value", "op", "threshold", "status", "rationale"]], width="stretch", hide_index=True)
    st.caption("Spec §8. Known failures are declared in the code with their mechanism and never block publishing; "
               "report rows have no threshold.")

with t_fig:
    for name in publish.FIGURES:
        path = pub.figures[name]
        if path is None:
            st.info(f"{name}.png not published.")
            continue
        st.subheader(name)
        st.image(str(path))
        st.download_button(f"Download {name}.png", data=path.read_bytes(), file_name=f"{name}.png", mime="image/png", key=f"dl_{name}")
    st.subheader("Data downloads")
    for key in ["labels", "acceptance", "regime_returns", "backtest_returns", "portfolio_weights"]:
        f = pub.out_dir / publish.FILES[key]
        if f.exists():
            st.download_button(f"Download {f.name}", data=f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{key}")

# ---------------- Zone 4: refresh ----------------
st.header("Refresh")
st.caption("Downloads the FRED-MD vintage and the ETF returns, reruns the engine and the walk-forward, re-evaluates the "
           "acceptance tests and publishes. If any blocking test fails the previous outputs stay live.")
_refresh_ui("Refresh data (≈3 min)")
```
If `st.dataframe(..., width="stretch")` or `st.pyplot(..., width="content")` raise on the installed Streamlit, replace with `use_container_width=True` and drop `width="content"`; say so in the report.

- [ ] **Step 4: Run the tests**

Run from the repo root: `../.venv/Scripts/python.exe -m pytest tests -q` (i.e. `.venv/Scripts/python.exe -m pytest tests -q`), then `cd regime_v2 && ../.venv/Scripts/python.exe -m pytest tests/test_regimes_quadrants.py -q`.
Expected: 4 app tests pass (first run ≈ 90 s for the session fixture); the quadrant tests pass.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/conftest.py tests/test_app_smoke.py regime_v2/tests/test_regimes_quadrants.py
git commit -m "stage7: dashboard reads regime_v2 outputs; retire regime_core and ui_io"
```
(The `git rm` from Step 1 is staged already; confirm with `git status` that the four deletions are in this commit.)

---

### Task 4: Retire the notebook pipeline; one requirements file; README

**Files:**
- Delete: `Macro_Regime_Analysis.ipynb`, `main.py`
- Rewrite: `requirements.txt`; replace `regime_v2/requirements.txt` with a pointer
- Modify: `.gitignore`, `README.md`, `regime_v2/README.md` (one paragraph), `docs/SPEC.md` §6 Stage 7 checkboxes

- [ ] **Step 1: Delete and rewrite**

```bash
git rm Macro_Regime_Analysis.ipynb main.py
```
Also remove the untracked root artefacts from the working tree if present (`macro_regime_results.xlsx`, `state_space_regimes.png`, `prob_weighted_6040_dynamics.png`, `portfolio_weights_comparison.png`, `ui_data/`) — they are gitignored, so this is a plain delete, not a git operation.

`requirements.txt` (root, the single file the Dockerfile installs):
```
# Python 3.12. Install: .venv/Scripts/python.exe -m pip install -r requirements.txt
numpy>=1.26,<3
pandas>=2.2,<3
scipy>=1.11
scikit-learn>=1.4
statsmodels>=0.14
hmmlearn>=0.3.3
matplotlib>=3.8
yfinance>=0.2.40
pyarrow>=16
streamlit>=1.36
pytest>=8
```
`regime_v2/requirements.txt`:
```
# The engine's dependencies are the repo-root requirements.txt (one file for engine + dashboard).
-r ../requirements.txt
```
`.gitignore`: delete the "Generated outputs (rebuilt by notebook Cell H)" block (the xlsx, the three PNGs, `ui_data/`) and add:
```
# dashboard refresh lock and the container's data volume mount point
regime_v2/.refresh.lock
var/
```

`README.md` (root) — replace the whole file with, in this order: one paragraph on what the site shows (four regimes, walk-forward labels, the honest backtest numbers with the 60/40 benchmark); **Run locally** (`pip install -r requirements.txt`, `cd regime_v2 && python run.py data/fredmd_2026-07.csv`, `streamlit run app.py`); **Refresh** (the button, the vintage input, what happens on failure); **Layout** (`regime_v2/` engine and spec pointer, `app.py`, `tests/`); **Deployment** (one paragraph pointing at spec §12 and the `deploy states` command); **Tests** (`pytest tests` at root for the dashboard, `pytest` inside `regime_v2/` for the engine). No mention of the notebook, `ui_data/` or Q-codes anywhere.

`regime_v2/README.md`: replace any "notebook" or "ui_data" mention with the app; keep the CLI flag documentation.

`docs/SPEC.md` §6 Stage 7: tick the first three checkboxes (`- [x]`), leaving the grep/pytest and deployment-checklist items for Task 5.

- [ ] **Step 2: Verify**

```bash
grep -rnE "Q1_|Q2_|Q3_|Q4_|ui_data|regime_core|ui_io|Macro_Regime_Analysis" --include=*.py --include=*.md --include=*.txt --include=*.toml . | grep -v "\.venv\|docs/superpowers\|docs/regime_v2_prototype\|\.claude/"
.venv/Scripts/python.exe -m pip install -r requirements.txt --dry-run 2>&1 | tail -2
.venv/Scripts/python.exe -m pytest tests -q
cd regime_v2 && ../.venv/Scripts/python.exe -m pytest -q
```
Expected: the grep prints nothing except lines inside `docs/SPEC.md` §10 history that name the retired files (acceptable: the decision log records what was deleted); pip resolves; both suites green.

- [ ] **Step 3: Commit**

```bash
git add -A requirements.txt regime_v2/requirements.txt .gitignore README.md regime_v2/README.md docs/SPEC.md
git commit -m "stage7: retire the notebook pipeline; one requirements file; README for the site"
```

---

### Task 5: Docker deployment for `~/apps/states` on the Mini

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docker/entrypoint.sh`
- Modify: `docs/SPEC.md` §12 (deployment checklist), `README.md` (Deployment section already points here)

**Interfaces:**
- Consumes: app env vars from Task 3 (`REGIME_OUTPUT_DIR`, `REGIME_FIGS_DIR`, `REGIME_RETURNS_CACHE`), `run.py` flags.
- Produces: container serving Streamlit on `$STREAMLIT_SERVER_PORT` (8505), published outputs on the named volume `states_var` mounted at `/app/var`.

- [ ] **Step 1: Write the files**

`Dockerfile`:
```dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x docker/entrypoint.sh
# Published outputs and caches live on a volume so a rebuild keeps the last good run.
ENV REGIME_OUTPUT_DIR=/app/var/output REGIME_FIGS_DIR=/app/var/figs REGIME_RETURNS_CACHE=/app/var/returns_yfinance.parquet
VOLUME ["/app/var"]
ENTRYPOINT ["docker/entrypoint.sh"]
```

`docker/entrypoint.sh`:
```sh
#!/bin/sh
# First start: publish the pinned vintage (offline-capable) so the site is never empty.
set -e
mkdir -p /app/var
if [ ! -f /app/var/output/summary.json ]; then
  echo "no published run on the volume; publishing the pinned vintage"
  (cd /app/regime_v2 && python run.py data/fredmd_2026-07.csv --out-dir /app/var/output --figs-dir /app/var/figs \
      --returns-cache /app/var/returns_yfinance.parquet) || echo "pinned run failed; the app will show the empty state"
fi
exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.port="${STREAMLIT_SERVER_PORT:-8505}"
```

`docker-compose.yml`:
```yaml
services:
  states:
    build: .
    container_name: states
    restart: unless-stopped
    environment:
      STREAMLIT_SERVER_PORT: "8505"        # the only place the port lives; the tunnel route points at localhost:8505
    ports:
      - "127.0.0.1:8505:8505"
    volumes:
      - states_var:/app/var
volumes:
  states_var:
```

`.dockerignore`:
```
.venv
.git
.idea
.claude
.superpowers
__pycache__
*.pyc
external_docs
docs/regime_v2_prototype.zip
regime_v2/output*
regime_v2/figs*
regime_v2/data/returns_yfinance.parquet
var
tests
```
(`tests` is excluded from the image; `regime_v2/tests` stays so `pytest` can run inside the container if ever needed. `regime_v2/data/fredmd_2026-07.csv` and `returns_fixture.parquet` are tracked and therefore copied.)

- [ ] **Step 2: Verify what can be verified here**

Docker is not available on the dev machine. Check instead:
```bash
sh -n docker/entrypoint.sh
.venv/Scripts/python.exe -c "import yaml" 2>/dev/null || echo "no yaml; skip parse"
grep -c 8505 docker-compose.yml       # expect 3 (env, two sides of the port mapping)
grep -rn 8505 --include=*.py --include=*.toml --include=Dockerfile . | grep -v .venv   # expect nothing: the port lives in compose only
```
Then run the app once locally the way the container will, to prove the env vars work:
```bash
REGIME_OUTPUT_DIR=/tmp/states/output REGIME_FIGS_DIR=/tmp/states/figs .venv/Scripts/python.exe -c "import app" 2>&1 | tail -2
```
(Importing `app` outside Streamlit prints warnings and stops at `st.stop()`; the point is that the env-var path resolution and the `PublishedMissing` branch execute without error.)

- [ ] **Step 3: Spec §12 checklist and the Mini steps**

Append to `docs/SPEC.md` §12, after the serving-mechanics paragraph:
```
Deployment checklist (Stage 7, to run on the Mini once per new app; later deploys are `deploy states`):
1. `mkdir -p ~/apps && cd ~/apps && git clone https://github.com/jhcwalsh/StateAnalysis.git states && cd states`
2. `docker compose up -d --build` — the entrypoint publishes the pinned vintage on the first start (≈3 min), then serves on 8505.
3. In Cloudflare Zero Trust, confirm the published application route `states.lazyeconomist.com` → HTTP `localhost:8505`.
4. Open the site; the status header must show the current regime and "all acceptance tests passed" (or the declared known failure).
5. Press Refresh with the current vintage; confirm the run completes and the caption's vintage changes.
6. Add `8505 = states` to the port table in MacMiniHosting's runbook.
Logs: `docker logs -f states`. Rebuild after a push: `deploy states`. The volume `states_var` holds the published outputs; `docker volume rm states_var` resets the site to the pinned vintage on next start.
```
Tick the remaining §6 Stage 7 checkboxes, leaving a note "deployment checklist executed on the Mini: pending" until the user confirms.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore docker/entrypoint.sh docs/SPEC.md README.md
git commit -m "stage7: Docker deployment for ~/apps/states on the Mini (port 8505, volume for published outputs)"
```

---

## Self-review against the spec

- **§6 Stage 7 bullets:** `hysteretic_sign` already lives in `regime_v2/regimes.py`; the tests worth keeping move in Task 3 and `regime_core.py` is deleted (T3). `app.py`/`ui_io.py` re-pointed at `regime_labels.csv` and `summary.json` (T2, T3), notebook Refresh path removed (T3), `ui_data/` contract removed (T3, T4), `tests/test_ui_io.py` replaced by `regime_v2/tests/test_publish.py` (T2). Notebook and root artefacts retired, README and `.gitignore` updated (T4). Grep for Q-codes empty, both suites green, `run.py` rebuilds everything the dashboard needs (T4 verify, T1 adds the missing fields). Deployment checklist in §12 (T5).
- **§12:** app reads `output/` (T3), the eleven figures and the CSVs are downloadable (T3 Figures tab), refresh via `run.py --vintage` (T2/T3), pinned vintage ships in the image so the first start publishes offline (T5 entrypoint), no secrets. Port and host each in one file (T5, Global Constraints).
- **D10/D11:** the θ explorer uses `regimes.quadrant_labels` (causal); the dashboard never joins returns to labels itself (all such tables come from the engine).
- **Type consistency:** `Published.figures` maps name → `Path | None` and the app tests `pub.figures[name]` truthiness; `publish.FILES` keys used by the downloads section exist; `summary["current"]["probs"]` keys are the four regime names (T1) as the status bar expects; `S["params"]["theta"]` exists in the current summary; `a["backtest"][key]["perf"]` is `to_dict(orient="index")` so `pd.DataFrame(...).T` yields strategies as rows with `sharpe` etc. as columns; run lengths are computed inline from the live labels (the engine's `run_lengths` returns per-regime means).
- **Known risks:** Streamlit `width=` kwargs vary by version (T3 has the fallback); `parse_dates` on the index column (T2 note); the session fixture stubs the acceptance gate for the same reason `regime_v2/tests/test_run.py` does; Docker is verified only on the Mini.
