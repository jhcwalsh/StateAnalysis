# Regime Dashboard UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Streamlit dashboard over `Macro_Regime_Analysis.ipynb`: glanceable current-regime status, a live hysteresis-theta slider, the paper's tables/figures/backtest, and a refresh button that re-executes the notebook.

**Architecture:** The notebook stays the single source of truth for all economics and exports its results to `ui_data/` (parquet + json + the existing xlsx/PNGs). A shared module `regime_core.py` holds the hysteresis trigger and paper-convention constants, imported by BOTH the notebook and the app so there is exactly one implementation. `ui_io.py` handles loading/schema-guard/refresh-subprocess so it is unit-testable without Streamlit; `app.py` is a thin viewer.

**Tech Stack:** Python 3.12 (project `.venv`), pandas 2.3, Streamlit, pyarrow, pytest, matplotlib, openpyxl, jupyter/nbconvert (already installed).

## Global Constraints

- All commands use the project venv python: `./.venv/Scripts/python.exe` (Windows; Git Bash paths).
- The notebook is 700KB JSON — NEVER edit it by hand or via full-file read. All notebook edits go through small Python patch scripts using `json.load`/`json.dump` with `assert`-guarded exact-string replacement (`indent=1, ensure_ascii=False`, trailing newline), the established pattern in this repo.
- Notebook cell indices (0-based) as of HEAD (`cc8815c`): cell 6 = config, cell 20 = quadrant classifier, cell 51 = paper-convention remap, cell 60 = Cell J (PIT). 63 cells total. Verify with the assert in each patch script before replacing.
- Executing the notebook takes ~6 min (FRED + Yahoo downloads). Execute copies OUTSIDE the repo (scratchpad) with `PYTHONPATH` pointing at the project dir so `regime_core` imports; never `--inplace` on the working copy during this plan.
- Economics must not change: Task 2's bit-identity gate compares pre/post-switch runs executed the same day (FRED revisions make cross-day comparisons invalid).
- Commit after every task; commit messages end with the project's Co-Authored-By trailer.
- `SCRATCH` below means any temp dir outside the repo (e.g. the session scratchpad). Replace with an absolute path when running.

---

### Task 1: `regime_core.py` + tests + dependencies

**Files:**
- Create: `regime_core.py`
- Create: `tests/test_regime_core.py`
- Modify: `requirements.txt` (append `streamlit`, `pyarrow`, `pytest`)

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `regime_core.hysteretic_sign(series: pd.Series, theta: float) -> pd.Series[int]`
  - `regime_core.assign_quadrants(g_gap: pd.Series, pi_gap: pd.Series, theta: float) -> pd.Series[str]` (name `"Quadrant"`, values `Q1_LowG_LowPi | Q2_HighG_LowPi | Q3_HighG_HighPi | Q4_LowG_HighPi`)
  - `regime_core.duration_stats(labels: pd.Series) -> tuple[float, int]` (avg run length, switch count)
  - Constants: `QUAD_BY_SIGN`, `CODE_TO_PAPER`, `PAPER_DISPLAY`, `PAPER_COLORS`, `PAPER_ORDER`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regime_core.py`:

```python
import pandas as pd
import pytest

from regime_core import (
    hysteretic_sign, assign_quadrants, duration_stats,
    QUAD_BY_SIGN, CODE_TO_PAPER, PAPER_ORDER, PAPER_COLORS, PAPER_DISPLAY,
)


def _noisy_series():
    # Verified fixture: hovers around zero, one decisive down-move, one up-move.
    v = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05,
         -0.30, -0.10, 0.10, -0.10, 0.30, 0.05]
    idx = pd.date_range("2020-01-01", periods=len(v), freq="ME")
    return pd.Series(v, index=idx)


def _switches(s):
    return int((s.diff() != 0).sum()) - 1


def test_theta_zero_is_memoryless_sign():
    s = _noisy_series()
    out = hysteretic_sign(s, 0.0)
    expected = [1 if x >= 0 else -1 for x in s]
    # theta=0: flips whenever the value crosses 0 (v<0 -> -1, v>0 -> +1);
    # exact zeros hold the previous state, none in this fixture.
    assert list(out) == expected


def test_hysteresis_reduces_switches():
    s = _noisy_series()
    assert _switches(hysteretic_sign(s, 0.0)) == 8
    assert _switches(hysteretic_sign(s, 0.25)) == 2


def test_trigger_is_causal_prefix_property():
    s = _noisy_series()
    full = hysteretic_sign(s, 0.25)
    prefix = hysteretic_sign(s.iloc[:7], 0.25)
    assert list(full.iloc[:7]) == list(prefix)


def test_assign_quadrants_maps_sign_pairs():
    idx = pd.date_range("2020-01-01", periods=4, freq="ME")
    g = pd.Series([-1.0, 1.0, 1.0, -1.0], index=idx)
    p = pd.Series([-1.0, -1.0, 1.0, 1.0], index=idx)
    out = assign_quadrants(g, p, theta=0.5)
    assert list(out) == ["Q1_LowG_LowPi", "Q2_HighG_LowPi",
                         "Q3_HighG_HighPi", "Q4_LowG_HighPi"]
    assert out.name == "Quadrant"


def test_duration_stats():
    idx = pd.date_range("2020-01-01", periods=5, freq="ME")
    labels = pd.Series(["A", "A", "B", "B", "B"], index=idx)
    avg, switches = duration_stats(labels)
    assert avg == pytest.approx(2.5)
    assert switches == 1


def test_paper_constants_consistent():
    assert set(QUAD_BY_SIGN.values()) == set(CODE_TO_PAPER) - {"Neutral"}
    assert set(PAPER_ORDER) <= set(CODE_TO_PAPER.values())
    for k in PAPER_ORDER:
        assert k in PAPER_COLORS and k in PAPER_DISPLAY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_regime_core.py -v` (pytest not installed yet — first install deps below, then expect `ModuleNotFoundError: No module named 'regime_core'`)

Append to `requirements.txt`:

```
# UI (see docs/superpowers/specs/2026-07-19-regime-ui-design.md)
streamlit>=1.36
pyarrow>=16
pytest>=8
```

Run: `./.venv/Scripts/python.exe -m pip install -r requirements.txt`
Then the pytest command above. Expected: collection error, `No module named 'regime_core'`.

- [ ] **Step 3: Write `regime_core.py`**

The function bodies are moved VERBATIM from notebook cell 20 (trigger, mapping, durations) and cell 51 (constants). Only signatures change: public names (no underscore) and explicit `theta`.

```python
"""Shared regime-classification logic.

Imported by BOTH Macro_Regime_Analysis.ipynb (cells 20/51/60) and the
Streamlit app (app.py), so the hysteresis trigger and the paper-convention
constants have exactly one implementation. Function bodies were moved
verbatim from the notebook; do not edit here without re-running the
notebook's bit-identity check (see the 2026-07-19 UI design spec).
"""
import numpy as np
import pandas as pd

QUAD_BY_SIGN = {
    (-1, -1): "Q1_LowG_LowPi",
    ( 1, -1): "Q2_HighG_LowPi",
    ( 1,  1): "Q3_HighG_HighPi",
    (-1,  1): "Q4_LowG_HighPi",
}

# Paper convention: Q1=Goldilocks, Q2=Overheating, Q3=Stagflation, Q4=Recession
CODE_TO_PAPER = {
    "Q1_LowG_LowPi":   "Q4_Recession",
    "Q2_HighG_LowPi":  "Q1_Goldilocks",
    "Q3_HighG_HighPi": "Q2_Overheating",
    "Q4_LowG_HighPi":  "Q3_Stagflation",
    "Neutral":          "Neutral",
}

PAPER_DISPLAY = {
    "Q1_Goldilocks":  "Q1: Goldilocks (High G, Low Infl)",
    "Q2_Overheating": "Q2: Overheating (High G, High Infl)",
    "Q3_Stagflation": "Q3: Stagflation (Low G, High Infl)",
    "Q4_Recession":   "Q4: Recession (Low G, Low Infl)",
}

PAPER_COLORS = {
    "Q1_Goldilocks":  "#2166AC",
    "Q2_Overheating": "#4DAC26",
    "Q3_Stagflation": "#F97B06",
    "Q4_Recession":   "#D7191C",
    "Neutral":        "#888888",
}

PAPER_ORDER = ["Q1_Goldilocks", "Q2_Overheating", "Q3_Stagflation", "Q4_Recession"]


def hysteretic_sign(series, theta):
    """
    Schmitt-trigger sign of a series.

    The state flips from + to - only once the value drops BELOW -theta, and
    from - to + only once it rises ABOVE +theta. Between -theta and +theta the
    previous state persists, so observations hovering near zero stop
    oscillating.

    CAUSAL BY CONSTRUCTION: state at t is a function of observations up to t
    only. No future information enters, unlike a run-length / minimum-duration
    filter applied after the fact.
    """
    vals  = series.values
    state = 1 if vals[0] >= 0 else -1
    out   = np.empty(len(vals), dtype=int)
    for i, v in enumerate(vals):
        if   state > 0 and v < -theta: state = -1
        elif state < 0 and v >  theta: state = 1
        out[i] = state
    return pd.Series(out, index=series.index)


def assign_quadrants(g_gap, pi_gap, theta):
    """Assign a quadrant regime to each observation, with hysteresis."""
    df = pd.concat([g_gap, pi_gap], axis=1).dropna()
    sg = hysteretic_sign(df.iloc[:, 0], theta)
    sp = hysteretic_sign(df.iloc[:, 1], theta)
    lab = [QUAD_BY_SIGN[(int(a), int(b))] for a, b in zip(sg, sp)]
    return pd.Series(lab, index=df.index, name="Quadrant")


def duration_stats(labels):
    """(avg run length in periods, number of regime switches)."""
    runs = (labels != labels.shift()).cumsum()
    lengths = labels.groupby(runs).size()
    return float(lengths.mean()), int(len(lengths) - 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_regime_core.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add regime_core.py tests/test_regime_core.py requirements.txt
git commit -m "feat: extract shared regime logic into regime_core module"
```

---

### Task 2: Notebook import switch + bit-identity gate

**Files:**
- Modify: `Macro_Regime_Analysis.ipynb` cells 20, 51, 60 (via patch script)
- Create (throwaway, in SCRATCH): `patch_import_switch.py`, `extract_baseline.py`

**Interfaces:**
- Consumes: `regime_core` names from Task 1 (aliased in-notebook to the old private names so cells 25/48/51-58/60 need no edits: `_hysteretic_sign`, `_duration_stats`).
- Produces: a notebook whose cell 20/51 definitions come from `regime_core`, verified numerically identical.

- [ ] **Step 1: Baseline run (BEFORE any patch)**

```bash
cp Macro_Regime_Analysis.ipynb "$SCRATCH/nb_base.ipynb"
cd "$SCRATCH" && PYTHONPATH="C:/Users/james/PycharmProjects/StateAnalysis" \
  "C:/Users/james/PycharmProjects/StateAnalysis/.venv/Scripts/python.exe" -m jupyter nbconvert \
  --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=1800 \
  --output nb_base_exec.ipynb nb_base.ipynb
```

Expected: completes in ~6 min. Then extract the baseline stats with `extract_baseline.py` (write to SCRATCH, run with the venv python):

```python
import json, io, re, sys
src_nb, out_json = sys.argv[1], sys.argv[2]
nb = json.load(io.open(src_nb, encoding="utf-8"))
def txt(i):
    return "".join("".join(o.get("text", [])) for o in nb["cells"][i].get("outputs", []))
t = txt(20)
assert "Using HYSTERESIS_THETA" in t, "cell 20 output missing"
counts = dict(re.findall(r"(Q\d_\w+)\s+(\d+)", t.split("Quadrant Distribution:")[1]))
m = re.search(r"Average regime duration: ([\d.]+) months\s+\((\d+) switches\)", t)
errors = [o.get("ename") for c in nb["cells"] if c["cell_type"] == "code"
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
json.dump({"counts": counts, "avg": m.group(1), "switches": m.group(2),
           "errors": errors}, io.open(out_json, "w"), indent=1)
print(json.load(io.open(out_json)))
```

Run: `python extract_baseline.py nb_base_exec.ipynb baseline.json` — record the printed values; `errors` must be `[]`.

- [ ] **Step 2: Write and apply the patch script**

`patch_import_switch.py` (SCRATCH; run from the project dir with the venv python). Uses the repo's assert-guarded replacement pattern:

```python
import json, io, re

NB = "Macro_Regime_Analysis.ipynb"
nb = json.load(io.open(NB, encoding="utf-8"))
cells = nb["cells"]
assert len(cells) == 63

# ---- cell 20: replace local defs with imports --------------------------------
src = "".join(cells[20]["source"])
m = re.search(r"def _hysteretic_sign\(series, theta\):.*?return float\(lengths\.mean\(\)\), int\(len\(lengths\) - 1\)\n", src, re.S)
assert m, "cell 20 def block not found"
IMPORTS = (
    "# Trigger/duration logic lives in regime_core.py (shared with the UI).\n"
    "# Aliased to the old private names so downstream cells need no edits.\n"
    "from regime_core import (\n"
    "    hysteretic_sign as _hysteretic_sign,\n"
    "    assign_quadrants,\n"
    "    duration_stats as _duration_stats,\n"
    "    QUAD_BY_SIGN as _QUAD_BY_SIGN,\n"
    ")\n"
)
src = src[:m.start()] + IMPORTS + src[m.end():]
# also remove the now-shadowed dict literal if present
src = re.sub(r"_QUAD_BY_SIGN = \{[^}]*\}\n\n?", "", src)
OLD_CALL = "quad = assign_quadrants(g_gap, p_gap)"
assert OLD_CALL in src
src = src.replace(OLD_CALL, "quad = assign_quadrants(g_gap, p_gap, HYSTERESIS_THETA)", 1)
OLD_TH = "_q = assign_quadrants(g_gap, p_gap, theta=_th)"
assert OLD_TH in src   # sensitivity loop already passes theta explicitly
cells[20]["source"] = src.splitlines(keepends=True)

# ---- cell 51: constants come from regime_core --------------------------------
src = "".join(cells[51]["source"])
m = re.search(r"CODE_TO_PAPER = \{.*?PAPER_ORDER = \[[^\]]*\]\n", src, re.S)
assert m, "cell 51 constants block not found"
src = (src[:m.start()]
       + "from regime_core import CODE_TO_PAPER, PAPER_DISPLAY, PAPER_COLORS, PAPER_ORDER\n"
       + src[m.end():])
cells[51]["source"] = src.splitlines(keepends=True)

# ---- cell 60: explicit theta at the PIT call site ----------------------------
src = "".join(cells[60]["source"])
OLD = "pit_quad = assign_quadrants(pit_g, pit_p)"
assert OLD in src
src = src.replace(OLD, "pit_quad = assign_quadrants(pit_g, pit_p, HYSTERESIS_THETA)", 1)
cells[60]["source"] = src.splitlines(keepends=True)

with io.open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("patched cells 20, 51, 60")
```

Expected output: `patched cells 20, 51, 60`. If any assert fires, STOP — the notebook differs from this plan's assumptions; re-inspect cell indices before proceeding.

- [ ] **Step 3: Post-switch run and bit-identity comparison**

Repeat Step 1's copy/execute (name the outputs `nb_post*`), run `extract_baseline.py nb_post_exec.ipynb post.json`, then compare:

```bash
python -c "
import json,io
a=json.load(io.open('baseline.json')); b=json.load(io.open('post.json'))
assert b['errors']==[], b['errors']
assert a['counts']==b['counts'] and a['avg']==b['avg'] and a['switches']==b['switches'], (a,b)
print('BIT-IDENTITY OK:', b)
"
```

Expected: `BIT-IDENTITY OK: {...}`. Any mismatch fails the task — do not rationalize small differences; find the cause.

- [ ] **Step 4: Commit**

```bash
git add Macro_Regime_Analysis.ipynb
git commit -m "refactor: notebook imports trigger and paper constants from regime_core

Bit-identity verified: same-day pre/post runs produce identical quadrant
counts, average duration, and switch count."
```

---

### Task 3: Notebook export cell + `ui_data/` contract

**Files:**
- Modify: `Macro_Regime_Analysis.ipynb` (append 1 markdown + 1 code cell, becoming cells 63-64)
- Modify: `.gitignore` (add `ui_data/`)
- Create (throwaway, SCRATCH): `patch_export_cell.py`

**Interfaces:**
- Consumes (existing notebook globals at end of run): `g_gap, p_gap, gf, pf, pit_g, pit_p, quad, pit_quad, labels, probs_paper, ret_bt, bt_returns, bt_perf, monthly_prices, HYSTERESIS_THETA, COVID_EXCLUDE, PUBLICATION_LAG_M, _agree, _common, regime_stat_df, EXCEL_PATH`
- Produces `ui_data/` (all later tasks read ONLY these):
  - `gaps.parquet` — columns `g_gap, p_gap, pit_g, pit_p, growth_factor, inflation_factor` (DatetimeIndex, outer-joined; NaN where a series lacks that month)
  - `labels.parquet` — columns `quad, pit_quad, gmm_cluster`
  - `probs.parquet` — columns = `PAPER_ORDER`
  - `returns.parquet` — monthly simple returns, asset columns
  - `backtest.parquet` — strategy return columns `PIT_MaxSharpe, PIT_MinVar, Oracle_MaxSharpe, Static_6040, EqualWeight`
  - `tables.xlsx` — copy of `macro_regime_results.xlsx`
  - `meta.json` — schema below
- `UI_DATA_DIR` env var overrides the output dir (default `<cwd>/ui_data`); the plan's verification runs use this to keep exports out of the repo.

- [ ] **Step 1: Add `ui_data/` to `.gitignore`**

Append under the "Generated outputs" section:

```
ui_data/
```

- [ ] **Step 2: Append the export cells via patch script**

`patch_export_cell.py` appends one markdown cell and one code cell (same `mk()` helper pattern as previous notebook patches: markdown has no `outputs`; code gets `outputs: []`, `execution_count: None`; `metadata: {}`). Markdown source:

```
## UI Export (Cell M)

Writes `ui_data/` for the Streamlit dashboard (see
`docs/superpowers/specs/2026-07-19-regime-ui-design.md`). Files are written
to a `.tmp` staging dir and atomically renamed; `meta.json` goes LAST so a
reader that sees it can trust the rest. `UI_DATA_DIR` env var overrides the
target (default `./ui_data`).
```

Code cell source (complete):

```python
# ============================================================
# CELL M - Export ui_data/ for the Streamlit dashboard
# ============================================================
import json as _json, os as _os, shutil as _shutil

UI_SCHEMA_VERSION = 1
_ui_dir  = _os.environ.get("UI_DATA_DIR", _os.path.join(_os.getcwd(), "ui_data"))
_tmp_dir = _ui_dir + ".tmp"
_shutil.rmtree(_tmp_dir, ignore_errors=True)
_os.makedirs(_tmp_dir, exist_ok=True)
_os.makedirs(_ui_dir, exist_ok=True)

_gaps = pd.concat(
    [g_gap.rename("g_gap"), p_gap.rename("p_gap"),
     pit_g.rename("pit_g"), pit_p.rename("pit_p"),
     gf.rename("growth_factor"), pf.rename("inflation_factor")],
    axis=1)
_labels = pd.concat(
    [quad.rename("quad"), pit_quad.rename("pit_quad"),
     labels.rename("gmm_cluster")], axis=1)

_gaps.to_parquet(_os.path.join(_tmp_dir, "gaps.parquet"))
_labels.to_parquet(_os.path.join(_tmp_dir, "labels.parquet"))
probs_paper.to_parquet(_os.path.join(_tmp_dir, "probs.parquet"))
ret_bt.to_parquet(_os.path.join(_tmp_dir, "returns.parquet"))
bt_returns.to_parquet(_os.path.join(_tmp_dir, "backtest.parquet"))
_shutil.copy2(EXCEL_PATH, _os.path.join(_tmp_dir, "tables.xlsx"))

_cur_month = quad.index[-1]
_meta = {
    "schema_version": UI_SCHEMA_VERSION,
    "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "theta": HYSTERESIS_THETA,
    "covid_exclude": list(COVID_EXCLUDE) if COVID_EXCLUDE else None,
    "publication_lag_m": PUBLICATION_LAG_M,
    "sample": {"start": str(quad.index[0].date()), "end": str(_cur_month.date())},
    "current": {
        "month": _cur_month.strftime("%Y-%m"),
        "quadrant": str(quad.iloc[-1]),
        "growth_gap": float(g_gap.iloc[-1]),
        "inflation_gap": float(p_gap.iloc[-1]),
        "probs": {k: float(probs_paper.iloc[-1][k]) for k in probs_paper.columns},
    },
    "pit": {
        "agreement": float(_agree.mean()),
        "n_common": int(len(_common)),
        "start": pit_quad.index[0].strftime("%Y-%m"),
        "end": pit_quad.index[-1].strftime("%Y-%m"),
    },
    "backtest": {
        "start": bt_returns.index[0].strftime("%Y-%m"),
        "end": bt_returns.index[-1].strftime("%Y-%m"),
        "n_months": int(len(bt_returns)),
        "perf": {s: {k: float(v) for k, v in bt_perf.loc[s].items()}
                 for s in bt_perf.index},
    },
}

# Atomic promotion: data files first, meta.json LAST.
_names = ["gaps.parquet", "labels.parquet", "probs.parquet",
          "returns.parquet", "backtest.parquet", "tables.xlsx"]
for _n in _names:
    _os.replace(_os.path.join(_tmp_dir, _n), _os.path.join(_ui_dir, _n))
with open(_os.path.join(_tmp_dir, "meta.json"), "w") as _f:
    _json.dump(_meta, _f, indent=1)
_os.replace(_os.path.join(_tmp_dir, "meta.json"),
            _os.path.join(_ui_dir, "meta.json"))
_shutil.rmtree(_tmp_dir, ignore_errors=True)

print(f"ui_data exported to {_ui_dir}")
for _n in _names + ["meta.json"]:
    print(f"  [{'OK' if _os.path.exists(_os.path.join(_ui_dir, _n)) else 'MISSING'}] {_n}")
```

Run the patch script. Expected: `appended cells 63-64 (65 total)`.

- [ ] **Step 3: Execute and verify the contract**

Execute a scratchpad copy as in Task 2, but with `UI_DATA_DIR="$SCRATCH/ui_data"` also in the environment. Expected: 0 errors, final cell prints all `[OK]`. Then verify readability:

```bash
python -c "
import pandas as pd, json, io, os
d = os.environ['UI_DATA_DIR']
meta = json.load(io.open(os.path.join(d, 'meta.json')))
assert meta['schema_version'] == 1
for f in ['gaps','labels','probs','returns','backtest']:
    df = pd.read_parquet(os.path.join(d, f + '.parquet'))
    assert len(df) > 0, f
print('contract OK; current:', meta['current']['quadrant'], meta['current']['month'])
"
```

Expected: `contract OK; current: <quadrant> <month>`.

- [ ] **Step 4: Commit**

```bash
git add Macro_Regime_Analysis.ipynb .gitignore
git commit -m "feat: notebook exports ui_data/ contract for the dashboard"
```

---

### Task 4: `ui_io.py` (load, schema guard, refresh runner) + tests

**Files:**
- Create: `ui_io.py`
- Create: `tests/conftest.py` (fixture builder, also used by Task 5)
- Create: `tests/test_ui_io.py`

**Interfaces:**
- Consumes: `ui_data/` contract from Task 3.
- Produces:
  - `ui_io.SCHEMA_VERSION = 1`
  - `ui_io.load_bundle(ui_dir: str) -> dict | None` — `None` if no `meta.json` (empty state); raises `ui_io.SchemaError` on version mismatch or missing file. Dict keys: `meta` (dict), `gaps`, `labels`, `probs`, `returns`, `backtest` (DataFrames), `tables_path` (str).
  - `ui_io.run_refresh(project_dir: str, timeout_s: int = 1800, runner=subprocess.run) -> tuple[bool, str]` — executes the notebook, returns `(ok, log_tail)`.

- [ ] **Step 1: Write the fixture builder and failing tests**

`tests/conftest.py`:

```python
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ui_data_dir(tmp_path):
    """Minimal ui_data/ satisfying the schema-1 contract."""
    d = tmp_path / "ui_data"
    d.mkdir()
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)

    gaps = pd.DataFrame({
        "g_gap": rng.normal(0, 1, 24), "p_gap": rng.normal(0, 1, 24),
        "pit_g": rng.normal(0, 1, 24), "pit_p": rng.normal(0, 1, 24),
        "growth_factor": rng.normal(0, 1, 24),
        "inflation_factor": rng.normal(0, 1, 24)}, index=idx)
    labels = pd.DataFrame({
        "quad": ["Q2_HighG_LowPi"] * 12 + ["Q3_HighG_HighPi"] * 12,
        "pit_quad": ["Q2_HighG_LowPi"] * 24,
        "gmm_cluster": [0] * 24}, index=idx)
    probs = pd.DataFrame(
        {k: [0.25] * 24 for k in
         ["Q1_Goldilocks", "Q2_Overheating", "Q3_Stagflation", "Q4_Recession"]},
        index=idx)
    rets = pd.DataFrame({"Equity_US": rng.normal(0.005, 0.04, 24),
                         "US_Aggregate_Bonds": rng.normal(0.002, 0.01, 24)}, index=idx)
    bt = pd.DataFrame({s: rng.normal(0.004, 0.02, 24) for s in
                       ["PIT_MaxSharpe", "PIT_MinVar", "Oracle_MaxSharpe",
                        "Static_6040", "EqualWeight"]}, index=idx)

    gaps.to_parquet(d / "gaps.parquet"); labels.to_parquet(d / "labels.parquet")
    probs.to_parquet(d / "probs.parquet"); rets.to_parquet(d / "returns.parquet")
    bt.to_parquet(d / "backtest.parquet")
    pd.DataFrame({"Regime": ["Prob-Weighted (2020-12)"], "Objective": ["Max Sharpe"],
                  "Port_Return": [0.05], "Port_Vol": [0.04], "Port_Sharpe": [1.25]}
                 ).to_excel(d / "tables.xlsx", sheet_name="Table3_Portfolios", index=False)

    meta = {"schema_version": 1, "run_timestamp": "2026-07-19T12:00:00",
            "theta": 0.5, "covid_exclude": ["2020-03-01", "2020-12-01"],
            "publication_lag_m": 1,
            "sample": {"start": "2020-01-01", "end": "2021-12-01"},
            "current": {"month": "2021-12", "quadrant": "Q3_HighG_HighPi",
                        "growth_gap": 0.7, "inflation_gap": 0.6,
                        "probs": {"Q1_Goldilocks": 0.28, "Q2_Overheating": 0.32,
                                  "Q3_Stagflation": 0.28, "Q4_Recession": 0.12}},
            "pit": {"agreement": 0.65, "n_common": 24,
                    "start": "2020-01", "end": "2021-12"},
            "backtest": {"start": "2020-01", "end": "2021-12", "n_months": 24,
                         "perf": {s: {"Ann_Return": 0.05, "Ann_Vol": 0.07,
                                      "Sharpe": 0.7, "Max_Drawdown": -0.1}
                                  for s in bt.columns}}}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d)
```

`tests/test_ui_io.py`:

```python
import json, os
import pytest
import ui_io


def test_load_bundle_ok(ui_data_dir):
    b = ui_io.load_bundle(ui_data_dir)
    assert b["meta"]["schema_version"] == 1
    assert set(b) >= {"meta", "gaps", "labels", "probs", "returns",
                      "backtest", "tables_path"}
    assert len(b["gaps"]) == 24
    assert os.path.exists(b["tables_path"])


def test_load_bundle_empty_state(tmp_path):
    assert ui_io.load_bundle(str(tmp_path / "nowhere")) is None


def test_load_bundle_schema_mismatch(ui_data_dir):
    meta_path = os.path.join(ui_data_dir, "meta.json")
    meta = json.load(open(meta_path)); meta["schema_version"] = 99
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ui_io.SchemaError):
        ui_io.load_bundle(ui_data_dir)


def test_load_bundle_missing_file_is_schema_error(ui_data_dir):
    os.remove(os.path.join(ui_data_dir, "backtest.parquet"))
    with pytest.raises(ui_io.SchemaError):
        ui_io.load_bundle(ui_data_dir)


def test_run_refresh_success_and_failure(tmp_path):
    class R:
        def __init__(s, code, err=b""): s.returncode, s.stderr, s.stdout = code, err, b""
    ok, tail = ui_io.run_refresh(str(tmp_path), runner=lambda *a, **k: R(0))
    assert ok
    ok, tail = ui_io.run_refresh(str(tmp_path),
                                 runner=lambda *a, **k: R(1, b"boom\nlast line"))
    assert not ok and "last line" in tail
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ui_io.py -v`
Expected: `ModuleNotFoundError: No module named 'ui_io'`.

- [ ] **Step 3: Write `ui_io.py`**

```python
"""Data access + refresh for the regime dashboard. No Streamlit imports."""
import json
import os
import subprocess
import sys

import pandas as pd

SCHEMA_VERSION = 1
NOTEBOOK = "Macro_Regime_Analysis.ipynb"
_PARQUETS = ["gaps", "labels", "probs", "returns", "backtest"]


class SchemaError(Exception):
    """ui_data/ exists but does not match the expected contract."""


def load_bundle(ui_dir):
    """Load ui_data/. None -> empty state (no run yet). SchemaError -> bad data."""
    meta_path = os.path.join(ui_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path, encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"ui_data schema {meta.get('schema_version')!r} != {SCHEMA_VERSION}; "
            "re-run the notebook to regenerate ui_data/.")
    bundle = {"meta": meta, "tables_path": os.path.join(ui_dir, "tables.xlsx")}
    for name in _PARQUETS:
        path = os.path.join(ui_dir, name + ".parquet")
        if not os.path.exists(path):
            raise SchemaError(f"missing {name}.parquet; re-run the notebook.")
    for name in _PARQUETS:
        bundle[name] = pd.read_parquet(os.path.join(ui_dir, name + ".parquet"))
    if not os.path.exists(bundle["tables_path"]):
        raise SchemaError("missing tables.xlsx; re-run the notebook.")
    return bundle


def run_refresh(project_dir, timeout_s=1800, runner=subprocess.run):
    """Execute the notebook (exports ui_data/ via Cell M). Returns (ok, log_tail)."""
    out_dir = os.path.join(project_dir, "ui_data", "last_run")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
           "--execute", "--ExecutePreprocessor.timeout=1500",
           os.path.join(project_dir, NOTEBOOK),
           "--output-dir", out_dir, "--output", "last_refresh.ipynb"]
    try:
        proc = runner(cmd, cwd=project_dir, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"refresh timed out after {timeout_s}s"
    tail = (proc.stderr or b"").decode(errors="replace")[-2000:]
    return proc.returncode == 0, tail
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_ui_io.py tests/test_regime_core.py -v`
Expected: all pass (5 new + 6 from Task 1).

- [ ] **Step 5: Commit**

```bash
git add ui_io.py tests/conftest.py tests/test_ui_io.py
git commit -m "feat: ui_io data loading, schema guard, and refresh runner"
```

---

### Task 5: `app.py` — all four zones + smoke test

**Files:**
- Create: `app.py`
- Create: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `ui_io.load_bundle/run_refresh/SchemaError`; `regime_core.assign_quadrants/duration_stats/CODE_TO_PAPER/PAPER_DISPLAY/PAPER_COLORS/PAPER_ORDER`.
- Produces: `streamlit run app.py` serving the dashboard; `UI_DATA_DIR` env var overrides the data dir (default `<app dir>/ui_data`).

**NOTE for the implementer:** load the `dataviz` skill BEFORE writing the chart code in this task (timeline strip, probability bar, wealth curves) and follow its guidance within the constraint that regime colors MUST remain `PAPER_COLORS` (they are the paper's published palette).

- [ ] **Step 1: Write the failing smoke test**

`tests/test_app_smoke.py`:

```python
import os
from streamlit.testing.v1 import AppTest


def test_app_renders_status(ui_data_dir, monkeypatch):
    monkeypatch.setenv("UI_DATA_DIR", ui_data_dir)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "Overheating" in page          # current regime rendered
    assert "2021-12" in page              # data vintage rendered


def test_app_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_DATA_DIR", str(tmp_path / "nowhere"))
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "No cached run found" in page
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_app_smoke.py -v`
Expected: FAIL (`app.py` does not exist).

- [ ] **Step 3: Write `app.py`**

```python
"""Regime dashboard — thin Streamlit viewer over ui_data/ (see design spec)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ui_io
from regime_core import (
    assign_quadrants, duration_stats,
    CODE_TO_PAPER, PAPER_DISPLAY, PAPER_COLORS, PAPER_ORDER,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.environ.get("UI_DATA_DIR", os.path.join(PROJECT_DIR, "ui_data"))

st.set_page_config(page_title="Macro Regime Dashboard", layout="wide")


@st.cache_data
def _bundle(mtime):
    return ui_io.load_bundle(UI_DIR)


def _meta_mtime():
    p = os.path.join(UI_DIR, "meta.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0


def _paper(code_label):
    return PAPER_DISPLAY.get(CODE_TO_PAPER.get(code_label, code_label), code_label)


def _paper_color(code_label):
    return PAPER_COLORS.get(CODE_TO_PAPER.get(code_label, code_label), "#888888")


try:
    bundle = _bundle(_meta_mtime())
except ui_io.SchemaError as e:
    st.error(f"ui_data/ is out of date: {e}")
    st.stop()

if bundle is None:
    st.title("Macro Regime Dashboard")
    st.markdown("**No cached run found** — press Refresh to run the notebook (≈5 min).")
    if st.button("Refresh now"):
        with st.spinner("Executing notebook…"):
            ok, tail = ui_io.run_refresh(PROJECT_DIR)
        if ok:
            st.cache_data.clear(); st.rerun()
        st.code(tail)
    st.stop()

meta, cur = bundle["meta"], bundle["meta"]["current"]

# ---------------- Zone 1: status header ----------------
st.title("Macro Regime Dashboard")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"<div style='background:{_paper_color(cur['quadrant'])};color:white;"
        f"padding:1.2em;border-radius:8px;font-size:1.5em;font-weight:bold'>"
        f"{_paper(cur['quadrant'])}</div>", unsafe_allow_html=True)
    st.markdown(
        f"Growth gap **{cur['growth_gap']:+.2f}** · Inflation gap "
        f"**{cur['inflation_gap']:+.2f}** (θ band ±{meta['theta']:.2f})")
with c2:
    probs = pd.Series(cur["probs"]).reindex(PAPER_ORDER)
    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.barh([0]*4, probs.values,
            left=probs.cumsum().shift(fill_value=0).values,
            color=[PAPER_COLORS[k] for k in PAPER_ORDER], height=0.5)
    for k, p, l in zip(PAPER_ORDER, probs.values,
                       probs.cumsum().shift(fill_value=0).values):
        if p > 0.08:
            ax.text(l + p/2, 0, f"{PAPER_DISPLAY[k].split(':')[0]} {p:.0%}",
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1); ax.axis("off")
    st.pyplot(fig, clear_figure=True)

st.caption(
    f"FRED through {meta['sample']['end'][:7]} · run {meta['run_timestamp'][:16]} "
    f"· PIT/final agreement {meta['pit']['agreement']:.1%} "
    f"({meta['pit']['n_common']} months)")

# ---------------- Zone 2: explore (theta) ----------------
st.header("Explore: hysteresis θ")
theta = st.slider("θ (regime persistence band)", 0.0, 1.0,
                  float(meta["theta"]), 0.05,
                  help=f"Run value: {meta['theta']}. Live relabeling from cached gaps.")
gaps = bundle["gaps"]
live = assign_quadrants(gaps["g_gap"].dropna(), gaps["p_gap"].dropna(), theta)
avg, switches = duration_stats(live)

m1, m2, m3 = st.columns(3)
m1.metric("Avg regime duration", f"{avg:.1f} mo")
m2.metric("Regime switches", switches)
m3.metric("Months classified", len(live))

fig, ax = plt.subplots(figsize=(14, 1.4))
for code in live.unique():
    mask = (live == code).values
    ax.bar(live.index[mask], 1, width=32, color=_paper_color(code),
           label=_paper(code))
ax.set_yticks([]); ax.margins(x=0)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4, fontsize=8,
          frameon=False)
st.pyplot(fig, clear_figure=True)

occ = live.value_counts().rename(_paper).rename_axis("Regime").to_frame("Months")
occ["% Time"] = (occ["Months"] / len(live)).map("{:.1%}".format)
st.dataframe(occ, use_container_width=True)
st.info(f"Tables and backtest below are pinned to run θ = {meta['theta']}. "
        "Change HYSTERESIS_THETA in the notebook and press Refresh to recompute them.")

# ---------------- Zone 3: results tabs ----------------
st.header("Results")
t1, t2, t3, t4, t5 = st.tabs(
    ["Regime returns", "Correlations", "Portfolios", "State space", "Backtest"])
with t1:
    st.dataframe(pd.read_excel(bundle["tables_path"],
                               sheet_name="Table1_RegimeReturns"),
                 use_container_width=True)
with t2:
    xl = pd.ExcelFile(bundle["tables_path"])
    for sheet in [s for s in xl.sheet_names if s.startswith("T2_")]:
        st.subheader(_paper(sheet[3:]) if sheet[3:] in CODE_TO_PAPER else sheet)
        st.dataframe(pd.read_excel(xl, sheet_name=sheet), use_container_width=True)
with t3:
    st.dataframe(pd.read_excel(bundle["tables_path"],
                               sheet_name="Table3_Portfolios"),
                 use_container_width=True)
with t4:
    png = os.path.join(PROJECT_DIR, "state_space_regimes.png")
    if os.path.exists(png):
        st.image(png)
    else:
        st.info("state_space_regimes.png not found — run the notebook (Cell H).")
with t5:
    bt = bundle["backtest"]
    wealth = (1 + bt).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in wealth.columns:
        ax.plot(wealth.index, wealth[col],
                linewidth=2.0 if col.startswith("PIT") else 1.2, label=col)
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    perf = pd.DataFrame(meta["backtest"]["perf"]).T
    st.dataframe(perf.style.format("{:+.3f}"), use_container_width=True)
    t3df = pd.read_excel(bundle["tables_path"], sheet_name="Table3_Portfolios")
    is_row = t3df[t3df["Regime"].str.startswith("Prob-Weighted")
                  & (t3df["Objective"] == "Max Sharpe")]
    if len(is_row):
        st.markdown(
            f"**Look-ahead decomposition** — in-sample Sharpe "
            f"{float(is_row['Port_Sharpe'].iloc[0]):+.2f} → oracle "
            f"{perf.loc['Oracle_MaxSharpe', 'Sharpe']:+.2f} → achievable (PIT) "
            f"{perf.loc['PIT_MaxSharpe', 'Sharpe']:+.2f} "
            f"vs 60/40 {perf.loc['Static_6040', 'Sharpe']:+.2f}")

# ---------------- Zone 4: refresh ----------------
st.header("Refresh")
st.caption("Re-executes the notebook (~5 min): downloads FRED/Yahoo, recomputes "
           "everything, rewrites ui_data/. Previous data stays live if it fails.")
if st.session_state.get("refreshing"):
    st.warning("A refresh is already running.")
elif st.button("Refresh data (≈5 min)"):
    st.session_state["refreshing"] = True
    try:
        with st.spinner("Executing notebook…"):
            ok, tail = ui_io.run_refresh(PROJECT_DIR)
    finally:
        st.session_state["refreshing"] = False
    if ok:
        st.success("Refresh complete."); st.cache_data.clear(); st.rerun()
    else:
        st.error("Refresh failed — previous data still shown. Log tail:")
        st.code(tail)
```

- [ ] **Step 4: Run all tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all pass (2 smoke + 5 ui_io + 6 regime_core).

- [ ] **Step 5: Manual launch check**

Run: `./.venv/Scripts/python.exe -m streamlit run app.py` (with real `ui_data/` present from a Task 3 run copied into the project, or after pressing Refresh once). Verify all four zones render and the theta slider updates duration/switches live. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "feat: Streamlit regime dashboard (status, explore, results, refresh)"
```

---

### Task 6: Launch docs + final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# StateAnalysis — Macro Regime Engine

`Macro_Regime_Analysis.ipynb` builds growth/inflation factors from FRED,
classifies four macro regimes (deterministic quadrants with causal
hysteresis; GMM as a robustness check), and constructs regime-conditional
portfolios — including a point-in-time track that quantifies look-ahead
bias. See `docs/superpowers/specs/` for design docs.

## Setup

    .venv/Scripts/python.exe -m pip install -r requirements.txt

Optional: set `FRED_API_KEY` in the environment.

## Dashboard

    .venv/Scripts/python.exe -m streamlit run app.py

Opens a local dashboard: current regime, live hysteresis-theta slider,
paper tables, and the OOS backtest. Press **Refresh** (≈5 min) on first
launch to populate `ui_data/`; after that it opens instantly from cache.
The notebook is the single source of truth — the app only re-labels
regimes when you move the theta slider (via the shared `regime_core.py`).

## Tests

    .venv/Scripts/python.exe -m pytest tests/ -v
```

- [ ] **Step 2: Full verification pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v` — all pass.
Launch the app once more; press Refresh; confirm it completes, the vintage line updates, and the page reloads with fresh data.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with dashboard launch instructions"
```

---

## Self-review notes

- **Spec coverage:** requirements (local/cache/refresh/theta-only) → Tasks 4-5; architecture files → Tasks 1, 4, 5; notebook changes 1-5 → Tasks 2-3; data contract → Task 3; four UI zones → Task 5; error handling (empty state, schema guard, atomic writes, failed refresh) → Tasks 3-5; testing section → Tasks 1 (trigger tests), 2 (bit-identity), 5 (smoke); dependencies → Task 1; build order preserved.
- **Type consistency:** `load_bundle` keys used in `app.py` match `ui_io.py`; `assign_quadrants(g, p, theta)` signature identical across Tasks 1/2/5; meta.json keys in Task 3 export match Task 4 fixture and Task 5 reads (`current.probs` keyed by PAPER_ORDER; `backtest.perf` keyed by strategy with `Ann_Return/Ann_Vol/Sharpe/Max_Drawdown` matching `bt_perf` columns from notebook Cell K).
- **Known judgment call:** cell-index asserts make every notebook patch fail loudly rather than corrupt silently; if indices drifted, the implementer must stop and re-verify rather than force.
