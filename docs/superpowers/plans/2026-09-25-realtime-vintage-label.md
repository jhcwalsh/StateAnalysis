# Real-Time-Vintage Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a comparator label `hmm_rt_vintage` (1999-07 →) computed from the FRED-MD vintage a reader actually had each month, with its comparison metrics, backtest, figure and document text, leaving every existing number unchanged.

**Architecture:** A new loader `data.load_archive_vintage` tolerates old vintages (renames, per-series drops, neighbour check); `build_blocks`/`run_pipeline` take a `loader` argument so the unchanged pipeline can run on any file. A new module `regime_v2/rtvintage.py` owns the stage: archive helpers, `fit_hmm4_rt_vintage` (one pipeline fit per month on vintage *t*+1 cut at *t*), and `summary_block` (the comparison metrics). `run.py` wires it in behind `--vintage-archive`, publishes new label columns, a `rt_vintage` summary block, report-only acceptance rows, fig 12 and an rt-label backtest in the asset stage; sitedocs/app render it behind `if:rt_vintage` guards.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib (Agg), pytest; engine in `regime_v2/`, tests in `regime_v2/tests/` (run from `regime_v2/`) and root `tests/`.

**Spec:** `docs/superpowers/specs/2026-09-25-realtime-vintage-label-design.md`

## Global Constraints

- The published label, `summary["current"]`, and every existing number in `summary.json`, `acceptance.csv`, `regime_labels.csv` and the documents are unchanged; a run without `--vintage-archive` produces byte-identical existing columns.
- A failure anywhere in the new stage (archive, loader, fit, figure, asset backtest) never changes `run.py`'s exit code; it is recorded as `{"skipped": reason}`.
- Vintage `YYYY-MM` ends month `YYYY-MM` − 1 (spec D11); month *t* is labelled from vintage *t*+1.
- Vintages ≤ `2014-12` are `reconstructed`; later ones `published`.
- Never a verdict in the documents: each backtest number sits beside `Static_6040`; direction words come from placeholders.
- Site documents: every placeholder used is listed in `docs/site/CONTRACT.md`; guarded text uses `<!-- if:rt_vintage -->…<!-- endif:rt_vintage -->` (own closing tag; the `if:assets` pattern stops at the first bare `<!-- endif -->`).
- All tests run offline (the suite's autouse fixture blocks yfinance; nothing here may hit the network).
- Commit messages end with the two attribution lines given in the session's system reminder.
- Working branch: `rt-vintage` (already exists, holds the spec).

## Review Focus

Inputs the spec implies but a task's tests might not cover unless added — each is pinned to the owning task below:

1. A vintage file that is the malformed plain 2026-08 (header shifted by one) reaching the archive loader — the neighbour check must drop the misaligned series or refuse the vintage rather than label from shifted columns (Task 1: `test_archive_loader_refuses_a_shifted_header`).
2. A vintage whose file exists but whose last data row is entirely empty (the shutdown files 2025-10..12) — the month must be a gap with reason, not a label from a blank row (Task 3: `test_gap_when_the_target_month_row_is_empty`).
3. `--vintage-archive` pointing at an empty or non-existent directory — stage skipped with a clear reason, exit code 0 (Task 5: `test_empty_archive_is_skipped_not_fatal`).
4. An archive whose first vintage is later than 1999-08 (a partial download) — the stage starts at the first available vintage and says so, never raising (Task 3: `test_start_is_clamped_to_the_first_vintage`).
5. A run published before this change (no `rt_vintage` block) rendered by the new documents and dashboard — `n/a` everywhere, guarded text dropped, no exception (Task 7: `test_rt_placeholders_read_na_without_the_block`; Task 8: `test_app_renders_without_rt_block`).

---

### Task 1: Archive vintage loader

**Files:**
- Modify: `regime_v2/regime_v2/data.py` (after `load_fredmd`, ~line 155; `build_blocks` signature at line 207)
- Modify: `regime_v2/regime_v2/pipeline.py:50-52` (`run_pipeline` signature and `build_blocks` call)
- Test: `regime_v2/tests/test_data_archive.py` (new)

**Interfaces:**
- Consumes: `data._read_raw`, `data._levels`, `data.check_vintage`, `data.GROWTH_BLOCK`, `data.INFLATION_BLOCK`, `data.VintageError`.
- Produces:
  - `data.ARCHIVE_RENAMES: dict[str, str] = {"PPIFGS": "WPSFD49207", "PPIFCG": "WPSFD49502"}`
  - `data.ARCHIVE_DROP: set[str] = {"CUUR0000SA0L2"}`
  - `data.ANCHORS: tuple[str, str] = ("INDPRO", "CPIAUCSL")`
  - `data.load_archive_vintage(path, neighbour=None) -> tuple[pd.DataFrame, pd.Series, dict]` where the dict is `{"renamed": {old: new}, "dropped_check": [series], "dropped_tcode": [series], "missing": [block series absent], "n_growth": int, "n_inflation": int}`; raises `VintageError` when an anchor is missing/failing or a block keeps fewer than half its series.
  - `data.build_blocks(path, k_outlier=10.0, asof=None, mask=COVID_MASK, loader=load_fredmd)`; `loader(path)` must return `(levels, tcodes)` (extra tuple items ignored).
  - `pipeline.run_pipeline(path, asof=None, loader=None, **overrides)`; `loader=None` means `load_fredmd`.

- [ ] **Step 1: Write the failing tests**

```python
# regime_v2/tests/test_data_archive.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (from `regime_v2/`): `python -m pytest tests/test_data_archive.py -q`
Expected: FAIL — `AttributeError: module 'regime_v2.data' has no attribute 'load_archive_vintage'` (and `TypeError` for the `loader` keyword).

- [ ] **Step 3: Implement the loader and the `loader` argument**

Add to `regime_v2/regime_v2/data.py` after `load_fredmd`:

```python
# --- archive vintages (spec 2026-09-25 §2) --------------------------------------------------
ARCHIVE_RENAMES = {"PPIFGS": "WPSFD49207", "PPIFCG": "WPSFD49502"}   # same series, older names
ARCHIVE_DROP = {"CUUR0000SA0L2"}     # the NSA CPI stand-in of 2015-01..2017-03: never substituted
ANCHORS = ("INDPRO", "CPIAUCSL")


def load_archive_vintage(path: str, neighbour: str | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (levels, tcodes, report) for a historical FRED-MD vintage.

    Unlike `load_fredmd`, which checks a live vintage against the pinned one and refuses the
    whole file, an archive vintage is checked *per series* against its neighbouring vintage:
    a series that fails the level-correlation or t-code check is dropped and recorded, and the
    vintage is kept. The vintage is refused only when an anchor series is missing or failing,
    or when a block keeps fewer than half its series.
    """
    raw = _read_raw(path)
    renamed = {old: new for old, new in ARCHIVE_RENAMES.items() if old in raw.columns and new not in raw.columns}
    raw = raw.rename(columns=renamed)
    raw = raw.drop(columns=[c for c in ARCHIVE_DROP if c in raw.columns])
    levels, tcodes = _levels(raw)
    block = GROWTH_BLOCK + INFLATION_BLOCK
    dropped_check: list[str] = []
    dropped_tcode: list[str] = []
    if neighbour is not None:
        n_levels, n_tcodes = _levels(_read_raw(neighbour))
        dropped_check = check_vintage(levels, n_levels, block)
        dropped_tcode = [c for c in block if c in tcodes.index and c in n_tcodes.index
                         and int(tcodes[c]) != int(n_tcodes[c]) and c not in dropped_check]
    for a in ANCHORS:
        if a not in levels.columns or a in dropped_check or a in dropped_tcode:
            raise VintageError(f"{os.path.basename(str(path))}: anchor series {a} is missing or fails the "
                               "neighbour check; refusing the vintage")
    drop = dropped_check + dropped_tcode
    levels = levels.drop(columns=drop)
    tcodes = tcodes.drop(index=drop)
    missing = sorted(c for c in block if c not in levels.columns)
    n_growth = sum(c in levels.columns for c in GROWTH_BLOCK)
    n_inflation = sum(c in levels.columns for c in INFLATION_BLOCK)
    for name, n, total in (("growth", n_growth, len(GROWTH_BLOCK)), ("inflation", n_inflation, len(INFLATION_BLOCK))):
        if n * 2 < total:
            raise VintageError(f"{os.path.basename(str(path))}: only {n} of {total} {name} series usable; "
                               "refusing the vintage")
    report = {"renamed": renamed, "dropped_check": dropped_check, "dropped_tcode": dropped_tcode,
              "missing": missing, "n_growth": n_growth, "n_inflation": n_inflation}
    levels.attrs["vintage_note"] = None
    return levels, tcodes, report
```

Change `build_blocks`'s signature and first line:

```python
def build_blocks(path: str, k_outlier: float = 10.0, asof: str | None = None,
                 mask: tuple[str, str] | None = COVID_MASK, loader=None) -> dict:
    levels, tcodes_all = (loader or load_fredmd)(path)[:2]
```

In `regime_v2/regime_v2/pipeline.py`, change `run_pipeline`:

```python
def run_pipeline(path: str, asof: str | None = None, loader=None, **overrides) -> PipelineResult:
    params = {**DEFAULTS, **overrides}
    blocks = build_blocks(path, k_outlier=params["k_outlier"], asof=asof, mask=params["mask"], loader=loader)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_data_archive.py tests/test_data.py tests/test_pipeline.py -q`
Expected: all PASS. If `test_archive_loader_refuses_a_shifted_header` fails because the shifted `INDPRO` column still correlates ≥ 0.98 with the neighbour, change the test to assert `len(report["dropped_check"]) >= 20` inside a `try/except VintageError: return` — the requirement is that no shifted series survives unrecorded.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/data.py regime_v2/regime_v2/pipeline.py regime_v2/tests/test_data_archive.py
git commit -m "feat(data): archive-vintage loader with per-series neighbour checks; pipeline takes a loader"
```

---

### Task 2: Synthetic archive fixture

**Files:**
- Modify: `regime_v2/tests/conftest.py` (append)
- Test: `regime_v2/tests/test_conftest_archive.py` (new; pins the fixture's own contract)

**Interfaces:**
- Produces: fixture `make_archive` — a callable `make_archive(root: Path, vintages: list[str], plant: dict[str, callable] | None = None) -> Path`. For each vintage `YYYY-MM` it writes `root/fredmd_YYYY-MM.csv` holding the pinned file's t-code row plus data rows with date ≤ the last day of month `YYYY-MM` − 1. `plant[v]` is a function `raw -> raw` applied to that vintage's raw frame before writing (t-code row 0, data rows 1..). Returns `root`.

- [ ] **Step 1: Write the failing test**

```python
# regime_v2/tests/test_conftest_archive.py
import pandas as pd

from regime_v2 import data as D


def test_make_archive_truncates_each_vintage_to_the_previous_month(make_archive, tmp_path):
    root = make_archive(tmp_path / "arch", ["2026-05", "2026-06", "2026-07"])
    for v, last in (("2026-05", "2026-04-01"), ("2026-06", "2026-05-01"), ("2026-07", "2026-06-01")):
        levels, _ = D.load_fredmd(str(root / f"fredmd_{v}.csv"), reference=None)
        assert levels.index[-1] == pd.Timestamp(last), v
        assert levels.index[0] == pd.Timestamp("1959-01-01")


def test_make_archive_applies_a_plant_to_one_vintage_only(make_archive, tmp_path):
    def double_indpro(raw):
        raw = raw.copy()
        raw.loc[1:, "INDPRO"] = (pd.to_numeric(raw.loc[1:, "INDPRO"]) * 2).to_numpy()
        return raw
    root = make_archive(tmp_path / "arch", ["2026-06", "2026-07"], plant={"2026-07": double_indpro})
    a, _ = D.load_fredmd(str(root / "fredmd_2026-06.csv"), reference=None)
    b, _ = D.load_fredmd(str(root / "fredmd_2026-07.csv"), reference=None)
    assert b.loc["2020-01-01", "INDPRO"] == 2 * a.loc["2020-01-01", "INDPRO"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_conftest_archive.py -q`
Expected: FAIL — `fixture 'make_archive' not found`.

- [ ] **Step 3: Add the fixture**

Append to `regime_v2/tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def make_archive():
    """Build a synthetic vintage archive from the pinned file: vintage YYYY-MM holds data rows
    through the end of the previous month (spec D11). `plant[v]` edits one vintage's raw frame."""
    import pandas as pd

    raw_all = pd.read_csv(VINTAGE, dtype=str, keep_default_na=False)
    dates = pd.to_datetime(raw_all["sasdate"].iloc[1:], errors="coerce")

    def build(root, vintages, plant=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for v in vintages:
            end = pd.Period(v, "M") - 1
            keep = (dates <= end.to_timestamp(how="end")).to_numpy()
            raw = pd.concat([raw_all.iloc[:1], raw_all.iloc[1:][keep]])
            if plant and v in plant:
                raw = plant[v](raw)
            raw.to_csv(root / f"fredmd_{v}.csv", index=False)
        return root

    return build
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_conftest_archive.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/tests/conftest.py regime_v2/tests/test_conftest_archive.py
git commit -m "test: synthetic FRED-MD vintage archive fixture"
```

---

### Task 3: The real-time-vintage walk-forward (`rtvintage.py`)

**Files:**
- Create: `regime_v2/regime_v2/rtvintage.py`
- Test: `regime_v2/tests/test_rtvintage.py` (new)

**Interfaces:**
- Consumes: `data.load_archive_vintage`, `pipeline.run_pipeline(path, asof, loader=...)`, `regimes.REGIMES`, fixture `make_archive`.
- Produces:
  ```python
  RECONSTRUCTED_THROUGH = "2014-12"
  def list_vintages(archive_dir) -> dict[str, Path]          # {"1999-08": Path(.../fredmd_1999-08.csv), ...} sorted
  def vintage_for_month(month: str) -> str                    # "2026-06" -> "2026-07"
  def provenance(vintage: str) -> str                         # "reconstructed" | "published"
  @dataclass
  class RtVintageResult:
      labels: pd.Series           # name "hmm_rt_vintage"; index = labelled months only
      probs: pd.DataFrame         # columns REGIMES, same index
      growth_gap: pd.Series       # name "growth_gap_rt_vintage"
      inflation_gap: pd.Series    # name "inflation_gap_rt_vintage"
      gaps: dict[str, str]        # "YYYY-MM" -> reason, for every month in [start, end] without a label
      reports: dict[str, dict]    # vintage -> loader report, or {"refused": reason}
      provenance: pd.Series       # index = labelled months, values "reconstructed"/"published"
      start: str                  # first month attempted (after clamping), "YYYY-MM"
  def fit_hmm4_rt_vintage(archive_dir, start="1999-07", end=None, progress=None, **kw) -> RtVintageResult
  ```
  `end` defaults to the month before the last archive vintage. `kw` are pipeline overrides (window, theta).

- [ ] **Step 1: Write the failing tests**

```python
# regime_v2/tests/test_rtvintage.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rtvintage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'regime_v2.rtvintage'`.

- [ ] **Step 3: Write the module**

```python
# regime_v2/regime_v2/rtvintage.py
"""Real-time-vintage label (spec 2026-09-25): month t labelled from the FRED-MD vintage a
reader had at t+1, by the unchanged pipeline. A comparator beside the published walk-forward
label (D8 amended 2026-09-25); never the primary label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import VintageError, load_archive_vintage
from .pipeline import run_pipeline
from .regimes import REGIMES

RECONSTRUCTED_THROUGH = "2014-12"     # vintages up to here were rebuilt by the Fed from archived Haver data
_NAME = re.compile(r"^fredmd_(\d{4}-\d{2})\.csv$")


def list_vintages(archive_dir) -> dict[str, Path]:
    d = Path(archive_dir)
    if not d.is_dir():
        return {}
    out = {m.group(1): p for p in d.iterdir() if (m := _NAME.match(p.name))}
    return dict(sorted(out.items()))


def vintage_for_month(month: str) -> str:
    return (pd.Period(month, "M") + 1).strftime("%Y-%m")


def provenance(vintage: str) -> str:
    return "reconstructed" if vintage <= RECONSTRUCTED_THROUGH else "published"


@dataclass
class RtVintageResult:
    labels: pd.Series
    probs: pd.DataFrame
    growth_gap: pd.Series
    inflation_gap: pd.Series
    gaps: dict = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    provenance: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    start: str = ""


def fit_hmm4_rt_vintage(archive_dir, start: str = "1999-07", end: str | None = None,
                        progress=None, **kw) -> RtVintageResult:
    vintages = list_vintages(archive_dir)
    if not vintages:
        raise FileNotFoundError(f"no fredmd_YYYY-MM.csv files in {archive_dir}")
    names = list(vintages)
    first_month = (pd.Period(names[0], "M") - 1).strftime("%Y-%m")
    last_month = (pd.Period(names[-1], "M") - 1).strftime("%Y-%m")
    start = max(start, first_month)
    end = min(end or last_month, last_month)
    months = pd.period_range(start, end, freq="M")
    probs, gg, pp, prov, gaps, reports = {}, {}, {}, {}, {}, {}
    for i, m in enumerate(months):
        month = str(m)
        v = vintage_for_month(month)
        if v not in vintages:
            gaps[month] = f"vintage {v} not in archive"
            continue
        pos = names.index(v)
        neighbour = vintages[names[pos - 1]] if pos > 0 else (vintages[names[1]] if len(names) > 1 else None)

        def loader(path, _n=neighbour):
            levels, tcodes, report = load_archive_vintage(path, neighbour=None if _n is None else str(_n))
            reports[v] = report
            return levels, tcodes

        t = m.to_timestamp()
        try:
            res = run_pipeline(str(vintages[v]), asof=str(m.to_timestamp(how="end").normalize()), loader=loader, **kw)
        except VintageError as e:
            reports[v] = {"refused": str(e)}
            gaps[month] = f"vintage {v} refused: {e}"
            continue
        if t not in res.hmm.probs_filtered.index:
            gaps[month] = f"vintage {v} carries no usable row for {month}"
            continue
        probs[t] = res.hmm.probs_filtered.loc[t]
        gg[t], pp[t] = float(res.G.loc[t]), float(res.P.loc[t])
        prov[t] = provenance(v)
        if progress:
            progress(i + 1, len(months), t)
    P = pd.DataFrame(probs).T.reindex(columns=REGIMES)
    P.index = pd.DatetimeIndex(P.index, name="date")
    return RtVintageResult(labels=P.idxmax(axis=1).rename("hmm_rt_vintage"), probs=P,
                           growth_gap=pd.Series(gg, name="growth_gap_rt_vintage", dtype=float),
                           inflation_gap=pd.Series(pp, name="inflation_gap_rt_vintage", dtype=float),
                           gaps=gaps, reports=reports, provenance=pd.Series(prov, dtype=object), start=start)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rtvintage.py -q`
Expected: 8 passed (each fit is 2–3 pipeline runs, well under a minute in total). If `test_gap_when_the_target_month_row_is_empty` fails because `_levels` keeps the blank row as NaN and the pipeline still labels it: the pipeline's `dropna(how="all")` in `_levels` already removes an all-empty row, so the month is genuinely absent — check the plant blanked *every* data cell of the last row (`raw.iloc[-1, 1:] = ""`).

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/rtvintage.py regime_v2/tests/test_rtvintage.py
git commit -m "feat(rtvintage): real-time-vintage walk-forward, month t from vintage t+1"
```

---

### Task 4: Comparison metrics (`summary_block`) and fig 12

**Files:**
- Modify: `regime_v2/regime_v2/rtvintage.py` (append)
- Modify: `regime_v2/regime_v2/figures.py` (append after `fig7_walkforward`, ~line 158)
- Modify: `regime_v2/regime_v2/publish.py:31-33` (`FIGURES` list)
- Test: `regime_v2/tests/test_rtvintage.py` (append), `regime_v2/tests/test_figures.py` (append)

**Interfaces:**
- Consumes: `RtVintageResult`, `figures.nber_lags`, `figures._strip`, `figures._shade`, `regimes.COLORS`, `regimes.run_lengths`.
- Produces:
  - `rtvintage.summary_block(rtv: RtVintageResult, labels_df: pd.DataFrame) -> dict` with keys: `window {start, end, n_months}`, `start_requested`, `gaps`, `n_gaps`, `n_reconstructed`, `n_published`, `agreement {overall, reconstructed, published}` (floats or None when n = 0), `crosstab {final_regime: {rt_regime: int}}`, `shares {final: {...}, rt: {...}}`, `mean_run_length {final: {...}, rt: {...}}`, `switches {final, rt}`, `gap_revision {growth: {mean, sd, corr, sign_agreement}, inflation: {...}}`, `nber_lags {final: [...], rt: [...]}` (records of `figures.nber_lags` for peaks ≥ window start), `loader {n_vintages, renamed: {old: count}, dropped: {series: count}, missing: {series: count}, refused: {vintage: reason}}`. `labels_df` must carry `hmm_walkforward`, `growth_gap`, `inflation_gap`.
  - `figures.fig12_rt_vintage(labels_df: pd.DataFrame, rtv: RtVintageResult, path: str) -> None` — two strips (final-vintage walk-forward above, real-time vintage below) over the rt window, disagreement months ticked between them, the reconstructed span shaded and labelled, NBER shading.
  - `publish.FIGURES` includes `"fig12_rt_vintage"`.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_rtvintage.py`:

```python
def _labels_df(index, final, g, p):
    df = pd.DataFrame(index=pd.DatetimeIndex(index, name="date"))
    df["hmm_walkforward"] = final
    df["growth_gap"] = g
    df["inflation_gap"] = p
    return df


def _rtv(index, labels, g, p, prov):
    probs = pd.DataFrame(0.0, index=pd.DatetimeIndex(index, name="date"), columns=REGIMES)
    for t, l in zip(probs.index, labels):
        probs.loc[t, l] = 1.0
    return RT.RtVintageResult(labels=pd.Series(labels, index=probs.index, name="hmm_rt_vintage"), probs=probs,
                              growth_gap=pd.Series(g, index=probs.index), inflation_gap=pd.Series(p, index=probs.index),
                              gaps={"2001-05": "vintage 2001-06 refused: x"},
                              reports={"2001-02": {"renamed": {"PPIFGS": "WPSFD49207"}, "dropped_check": ["CUMFNS"],
                                                   "dropped_tcode": [], "missing": ["HWI"], "n_growth": 20, "n_inflation": 13},
                                       "2001-06": {"refused": "x"}},
                              provenance=pd.Series(prov, index=probs.index), start="2001-01")


def test_summary_block_metrics_are_computed_over_the_overlap():
    idx = pd.date_range("2001-01-01", periods=6, freq="MS")
    final = ["Goldilocks", "Goldilocks", "Contraction", "Contraction", "Stagflation", "Stagflation"]
    rt_labels = ["Goldilocks", "Contraction", "Contraction", "Contraction", "Contraction", "Stagflation"]
    df = _labels_df(idx, final, [0.5, 0.4, -0.5, -0.6, -0.7, -0.8], [0.1, 0.0, -0.2, -0.3, 0.5, 0.6])
    rtv = _rtv(idx, rt_labels, [0.6, 0.3, -0.4, -0.7, -0.6, -0.9], [0.2, -0.1, -0.1, -0.4, 0.4, 0.7],
               ["reconstructed"] * 4 + ["published"] * 2)
    b = RT.summary_block(rtv, df)
    assert b["window"] == {"start": "2001-01", "end": "2001-06", "n_months": 6} and b["start_requested"] == "2001-01"
    assert b["agreement"]["overall"] == pytest.approx(4 / 6)
    assert b["agreement"]["reconstructed"] == pytest.approx(3 / 4) and b["agreement"]["published"] == pytest.approx(1 / 2)
    assert b["crosstab"]["Goldilocks"]["Contraction"] == 1 and b["crosstab"]["Stagflation"]["Contraction"] == 1
    assert b["shares"]["rt"]["Contraction"] == pytest.approx(4 / 6)
    assert b["switches"] == {"final": 2, "rt": 2}
    assert b["gap_revision"]["growth"]["sign_agreement"] == 1.0
    assert b["gap_revision"]["inflation"]["sign_agreement"] == pytest.approx(4 / 6)
    assert b["n_gaps"] == 1 and b["gaps"] == {"2001-05": "vintage 2001-06 refused: x"}
    assert b["n_reconstructed"] == 4 and b["n_published"] == 2
    assert b["loader"] == {"n_vintages": 2, "renamed": {"PPIFGS": 1}, "dropped": {"CUMFNS": 1},
                           "missing": {"HWI": 1}, "refused": {"2001-06": "x"}}
    assert [r["peak"] for r in b["nber_lags"]["final"]] == ["2001-03"]
    assert b["nber_lags"]["final"][0]["lag_months"] == 0 and b["nber_lags"]["rt"][0]["lag_months"] == 0


def test_summary_block_survives_an_empty_provenance_class():
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    df = _labels_df(idx, ["Goldilocks"] * 3, [0.1] * 3, [0.1] * 3)
    rtv = _rtv(idx, ["Goldilocks"] * 3, [0.1] * 3, [0.1] * 3, ["published"] * 3)
    b = RT.summary_block(rtv, df)
    assert b["agreement"]["reconstructed"] is None and b["agreement"]["published"] == 1.0
```

Append to `regime_v2/tests/test_figures.py`:

```python
def test_fig12_rt_vintage_draws_two_strips(tmp_path):
    import numpy as np
    from regime_v2 import figures as F, rtvintage as RT
    from regime_v2.regimes import REGIMES
    idx = pd.date_range("2010-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(0)
    final = rng.choice(REGIMES, 60)
    rt_l = final.copy(); rt_l[::7] = "Contraction"
    df = pd.DataFrame({"hmm_walkforward": final, "growth_gap": rng.normal(size=60), "inflation_gap": rng.normal(size=60)},
                      index=pd.DatetimeIndex(idx, name="date"))
    probs = pd.DataFrame(0.0, index=df.index, columns=REGIMES)
    rtv = RT.RtVintageResult(labels=pd.Series(rt_l, index=df.index, name="hmm_rt_vintage"), probs=probs,
                             growth_gap=df["growth_gap"], inflation_gap=df["inflation_gap"],
                             provenance=pd.Series(["reconstructed"] * 30 + ["published"] * 30, index=df.index), start="2010-01")
    out = tmp_path / "fig12.png"
    F.fig12_rt_vintage(df, rtv, str(out))
    assert out.exists() and out.stat().st_size > 10_000


def test_fig12_is_a_published_figure_name():
    from regime_v2 import publish
    assert "fig12_rt_vintage" in publish.FIGURES
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_rtvintage.py tests/test_figures.py -q -k "summary_block or fig12"`
Expected: FAIL — `AttributeError: ... has no attribute 'summary_block'` / `'fig12_rt_vintage'`; the `FIGURES` assertion fails.

- [ ] **Step 3: Implement**

Append to `regime_v2/regime_v2/rtvintage.py`:

```python
import numpy as np
from collections import Counter

from .figures import nber_lags
from .regimes import run_lengths


def _agree(a: pd.Series, b: pd.Series) -> float | None:
    return None if len(a) == 0 else float((a == b).mean())


def _revision(rt: pd.Series, fin: pd.Series) -> dict:
    d = pd.concat([rt.rename("rt"), fin.rename("fin")], axis=1).dropna()
    if len(d) < 3:
        return {"mean": None, "sd": None, "corr": None, "sign_agreement": None}
    diff = d["rt"] - d["fin"]
    return {"mean": float(diff.mean()), "sd": float(diff.std()),
            "corr": float(d["rt"].corr(d["fin"])),
            "sign_agreement": float((np.sign(d["rt"]) == np.sign(d["fin"])).mean())}


def summary_block(rtv: RtVintageResult, labels_df: pd.DataFrame) -> dict:
    """The comparison of the real-time-vintage label with the published walk-forward label,
    computed over their overlap. Everything here is reported, never thresholded."""
    j = rtv.labels.index.intersection(labels_df.index[labels_df["hmm_walkforward"].notna()])
    fin, rt = labels_df.loc[j, "hmm_walkforward"], rtv.labels.loc[j]
    prov = rtv.provenance.reindex(j)
    recon, pub = prov == "reconstructed", prov == "published"
    cross = pd.crosstab(fin, rt).reindex(index=REGIMES, columns=REGIMES, fill_value=0)
    renamed, dropped, missing, refused = Counter(), Counter(), Counter(), {}
    for v, rep in rtv.reports.items():
        if "refused" in rep:
            refused[v] = rep["refused"]
            continue
        renamed.update(rep["renamed"].keys())
        dropped.update(rep["dropped_check"] + rep["dropped_tcode"])
        missing.update(rep["missing"])
    lags_f, lags_r = nber_lags(fin), nber_lags(rt)
    in_window = lags_f["peak"] >= (j[0].strftime("%Y-%m") if len(j) else "9999")
    return {
        "window": {"start": j[0].strftime("%Y-%m") if len(j) else None,
                   "end": j[-1].strftime("%Y-%m") if len(j) else None, "n_months": int(len(j))},
        "start_requested": rtv.start,
        "gaps": dict(rtv.gaps), "n_gaps": len(rtv.gaps),
        "n_reconstructed": int(recon.sum()), "n_published": int(pub.sum()),
        "agreement": {"overall": _agree(fin, rt), "reconstructed": _agree(fin[recon], rt[recon]),
                      "published": _agree(fin[pub], rt[pub])},
        "crosstab": {r: {c: int(cross.loc[r, c]) for c in REGIMES} for r in REGIMES},
        "shares": {"final": fin.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0).round(4).to_dict(),
                   "rt": rt.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0).round(4).to_dict()},
        "mean_run_length": {"final": run_lengths(fin).round(2).to_dict(), "rt": run_lengths(rt).round(2).to_dict()},
        "switches": {"final": int((fin != fin.shift()).sum() - 1) if len(fin) else 0,
                     "rt": int((rt != rt.shift()).sum() - 1) if len(rt) else 0},
        "gap_revision": {"growth": _revision(rtv.growth_gap.reindex(j), labels_df.loc[j, "growth_gap"]),
                         "inflation": _revision(rtv.inflation_gap.reindex(j), labels_df.loc[j, "inflation_gap"])},
        "nber_lags": {"final": lags_f[in_window].to_dict(orient="records"),
                      "rt": lags_r[in_window].to_dict(orient="records")},
        "loader": {"n_vintages": len(rtv.reports), "renamed": dict(renamed), "dropped": dict(dropped),
                   "missing": dict(missing), "refused": refused},
    }
```

(`regimes.run_lengths` exists — it is what `build_summary` uses for `mean_run_length_months`; check its signature is `run_lengths(labels: pd.Series) -> pd.Series` and adapt the call if it differs.)

Append to `regime_v2/regime_v2/figures.py`:

```python
def fig12_rt_vintage(labels_df: pd.DataFrame, rtv, path: str) -> None:
    """Final-vintage walk-forward label above the real-time-vintage label, over the rt window;
    ticks where they disagree; the reconstructed-vintage span shaded and named."""
    j = rtv.labels.index
    fin = labels_df["hmm_walkforward"].reindex(j)
    fig, axes = plt.subplots(3, 1, figsize=(13, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1, 4]})
    _strip(axes[0], fin, COLORS, "Walk-forward label on the final vintage (published)")
    dis = (fin != rtv.labels) & fin.notna()
    axes[1].vlines(j[dis.to_numpy()], 0, 1, color="black", lw=0.8)
    axes[1].set_yticks([]); axes[1].set_ylim(0, 1)
    axes[1].set_title(f"months where the two labels differ: {int(dis.sum())} of {int(fin.notna().sum())}", fontsize=8)
    _strip(axes[2], rtv.labels, COLORS, "Label on the vintage available at the time (real-time vintage)")
    recon = rtv.provenance.reindex(j) == "reconstructed"
    if recon.any():
        a, b = j[recon.to_numpy()][0], j[recon.to_numpy()][-1] + pd.offsets.MonthEnd(0)
        for ax in axes:
            ax.axvspan(a, b, color="#c9a227", alpha=0.10, lw=0)
        axes[2].annotate("reconstructed vintages", (a, -0.08), xycoords=("data", "axes fraction"),
                         fontsize=7, color="#7a5c00", ha="left", va="top")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
```

In `regime_v2/regime_v2/publish.py`, add `"fig12_rt_vintage"` to the end of the `FIGURES` list.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_rtvintage.py tests/test_figures.py tests/test_publish.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/rtvintage.py regime_v2/regime_v2/figures.py regime_v2/regime_v2/publish.py regime_v2/tests/test_rtvintage.py regime_v2/tests/test_figures.py
git commit -m "feat(rtvintage): comparison metrics block and fig12 (final vs real-time-vintage labels)"
```

---

### Task 5: Wire the stage into `run.py` (columns, summary block, acceptance rows, fig 12, archive copy)

**Files:**
- Modify: `regime_v2/run.py` — `FIG_NAMES` (line 51), `main()` argument parser (~line 355), after the walk-forward (~line 432), label `parts` (~line 464), summary write.
- Modify: `regime_v2/regime_v2/acceptance.py:53-70` (`REPORT_ONLY`, `REPORT_RATIONALE`)
- Test: `regime_v2/tests/test_run.py` (append)

**Interfaces:**
- Consumes: `rtvintage.fit_hmm4_rt_vintage`, `rtvintage.summary_block`, `figures.fig12_rt_vintage`.
- Produces:
  - CLI `--vintage-archive DIR` (default `None`).
  - `run.run_rt_vintage(archive_dir, labels_df, staging_figs: Path, **kw) -> tuple[RtVintageResult | None, dict]` — returns `(rtv, block)`; `block` is `summary_block(...)` or `{"skipped": reason}`; never raises.
  - `run.archive_downloaded_vintage(path, archive_dir) -> bool` — copies `path` to `archive_dir/fredmd_YYYY-MM.csv` (name derived from the downloaded file's own name `fredmd_YYYY-MM.csv`) if absent; returns True when it wrote.
  - `regime_labels.csv` columns `hmm_rt_vintage`, `p_rt_<Regime>`, `growth_gap_rt_vintage`, `inflation_gap_rt_vintage` (present, empty, when the stage is skipped).
  - `summary["rt_vintage"]` = block.
  - Acceptance report-only rows `rt_vintage_agreement`, `rt_vintage_agreement_reconstructed`, `rt_vintage_agreement_published`, `rt_growth_gap_corr`, `rt_inflation_gap_corr` (NaN when skipped); `rt_pit_sharpe`, `rt_pit_longonly_sharpe` are added in Task 6.
  - `fig12_rt_vintage.png` in figs when the stage ran.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_run.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_run.py -q -k "rt_vintage or archive"`
Expected: FAIL — `unrecognized arguments: --vintage-archive`; `AttributeError: archive_downloaded_vintage`.

- [ ] **Step 3: Implement**

In `regime_v2/run.py`:

1. Imports: add `from regime_v2 import rtvintage` beside the other `regime_v2` imports.
2. `FIG_NAMES`: append `"fig12_rt_vintage"` (it is drawn only when the stage runs; `publish` copies whatever is in staging).
3. Add after `_placebo_block`:

```python
RT_LABEL_COLUMNS = ["hmm_rt_vintage", "growth_gap_rt_vintage", "inflation_gap_rt_vintage"]


def archive_downloaded_vintage(path, archive_dir) -> bool:
    """Keep the archive current: copy a downloaded vintage into it under its own name, once."""
    if not archive_dir:
        return False
    src = Path(path)
    dest = Path(archive_dir) / src.name
    if not src.name.startswith("fredmd_") or dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return True


def run_rt_vintage(archive_dir, labels_df: pd.DataFrame, staging_figs: Path, **kw):
    """The real-time-vintage stage (spec 2026-09-25). Returns (result, summary block); a failure
    anywhere becomes {"skipped": reason} and never changes the engine's exit code."""
    if not archive_dir:
        return None, {"skipped": "no vintage archive configured"}
    if not labels_df["hmm_walkforward"].notna().any():
        return None, {"skipped": "walk-forward disabled: nothing to compare the real-time-vintage label with"}
    try:
        rtv = rtvintage.fit_hmm4_rt_vintage(archive_dir, progress=lambda i, n, t: print(f"\rreal-time vintage {i}/{n} {t:%Y-%m}", end=""), **kw)
        print()
        if len(rtv.labels) == 0:
            return None, {"skipped": f"no month could be labelled: {rtv.gaps}"}
        block = rtvintage.summary_block(rtv, labels_df)
        figures.fig12_rt_vintage(labels_df, rtv, str(staging_figs / "fig12_rt_vintage.png"))
        return rtv, block
    except Exception as e:
        print(f"real-time-vintage stage skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return None, {"skipped": f"{type(e).__name__}: {e}"}


def rt_label_columns(rtv, index: pd.DatetimeIndex) -> list[pd.Series | pd.DataFrame]:
    """The published columns for the real-time-vintage label; empty when the stage did not run."""
    if rtv is None:
        empty = [pd.Series(pd.NA, index=index, name=c) for c in RT_LABEL_COLUMNS]
        return empty + [pd.DataFrame(pd.NA, index=index, columns=[f"p_rt_{r}" for r in R.REGIMES])]
    return [rtv.labels, rtv.growth_gap, rtv.inflation_gap, rtv.probs.add_prefix("p_rt_")]
```

4. Argument: `ap.add_argument("--vintage-archive", default=None, help="directory of fredmd_YYYY-MM.csv vintages; enables the real-time-vintage comparator")`.
5. After every branch that sets `path` from a download (`--vintage latest` and `--vintage YYYY-MM`), call `archive_downloaded_vintage(path, a.vintage_archive)`. Place a single call right after the `if not path: ap.error(...)` line: `if a.vintage: archive_downloaded_vintage(path, a.vintage_archive)`.
6. After `labels_df = labels_frame(res, extra=parts)` the stage needs `labels_df` with `hmm_walkforward`; so run it *after* that line and rejoin:

```python
    rtv, rt_block = run_rt_vintage(a.vintage_archive, labels_df, staging / "figs", **kw)
    for col in rt_label_columns(rtv, labels_df.index):
        labels_df = labels_df.join(col, how="left")
    summary["rt_vintage"] = rt_block
    if rt_block.get("skipped") is None:
        vals.update({"rt_vintage_agreement": rt_block["agreement"]["overall"],
                     "rt_vintage_agreement_reconstructed": rt_block["agreement"]["reconstructed"],
                     "rt_vintage_agreement_published": rt_block["agreement"]["published"],
                     "rt_growth_gap_corr": rt_block["gap_revision"]["growth"]["corr"],
                     "rt_inflation_gap_corr": rt_block["gap_revision"]["inflation"]["corr"]})
        table = acceptance.evaluate(vals)
        summary["acceptance_tests"] = table.reset_index().to_dict(orient="records")
        summary["acceptance_all_passed"] = bool(acceptance.all_passed(table))
        summary["acceptance_known_failures"] = {name: acceptance.KNOWN_FAILURES[name] for name in table.index
                                                if table.loc[name, "known_failure"] and not table.loc[name, "passed"]}
        summary["acceptance_blocking_failures"] = acceptance.blocking_failures(table)
```

This must sit *before* `labels_df.to_csv(staging / "output" / "regime_labels.csv")` and before `table.to_csv(...)`. `None` values in `vals` evaluate as NaN in `acceptance.evaluate` (it treats `None` as missing).

7. `acceptance.py`: append to `REPORT_ONLY`:
```python
               # real-time-vintage comparator (spec 2026-09-25): reported beside the published label
               "rt_vintage_agreement", "rt_vintage_agreement_reconstructed", "rt_vintage_agreement_published",
               "rt_growth_gap_corr", "rt_inflation_gap_corr", "rt_pit_sharpe", "rt_pit_longonly_sharpe"]
```
and to `REPORT_RATIONALE`:
```python
    "rt_vintage_agreement":
        "share of months 1999-07 on where the label from the vintage available at the time equals the "
        "published final-vintage walk-forward label; see rt_vintage_agreement_reconstructed / _published",
    "rt_pit_sharpe":
        "PIT max-Sharpe on the real-time-vintage label over the same window; compare pit_sharpe and static_6040_sharpe",
    "rt_pit_longonly_sharpe":
        "long-only PIT on the real-time-vintage label; compare pit_longonly_sharpe and static_6040_sharpe",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_run.py tests/test_acceptance_unit.py -q`
Expected: all PASS (the two new full runs cost ~40 s each). If `pd.testing.assert_frame_equal` fails only on dtype of the all-NA rt columns, compare after dropping them as the test already does — the assertion is on the *other* columns.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/run.py regime_v2/regime_v2/acceptance.py regime_v2/tests/test_run.py
git commit -m "feat(run): --vintage-archive stage: rt label columns, summary block, report rows, fig12, archive copy"
```

---

### Task 6: Backtest on the real-time-vintage label (asset stage)

**Files:**
- Modify: `regime_v2/run.py` — `run_assets` signature (line ~263) and body; the call site in `main`.
- Test: `regime_v2/tests/test_run.py` (append), `regime_v2/tests/test_portfolio.py` (append)

**Interfaces:**
- Consumes: `portfolio.backtest(returns, labels_frame, probs_rt, strategies=..., include_expost=False, cost_bp=...)`, `rtvintage.RtVintageResult`.
- Produces:
  - `run.rt_backtest_inputs(labels_df, rtv) -> tuple[pd.DataFrame, pd.DataFrame, int]` — a labels frame whose `hmm_walkforward` is `hmm_rt_vintage` held forward over gap months inside the rt window, the matching probabilities held the same way, and the number of held months.
  - `run_assets(..., rtv=None)`; `summary["assets"]["rt_vintage_backtest"]` = `{"perf": {cost_bp_0: {...}, cost_bp_10: {...}}, "counters": {...}, "held_months": int, "window": {"start", "end"}}` for strategies `PIT_MaxSharpe`, `PIT_LongOnly_MaxSharpe`, `Static_6040`; `None` when `rtv is None`.
  - Acceptance values `rt_pit_sharpe`, `rt_pit_longonly_sharpe` from `cost_bp_0`.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_run.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_run.py -q -k "rt_backtest or rt_vintage_backtest"`
Expected: FAIL — `AttributeError: rt_backtest_inputs`; `KeyError: 'rt_vintage_backtest'`.

- [ ] **Step 3: Implement**

In `regime_v2/run.py`, add after `rt_label_columns`:

```python
RT_BACKTEST_STRATEGIES = ["PIT_MaxSharpe", "PIT_LongOnly_MaxSharpe", "Static_6040"]


def rt_backtest_inputs(labels_df: pd.DataFrame, rtv) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Labels and probabilities for the backtest on the real-time-vintage label. Inside the rt
    window a gap month holds the previous month's label — what a reader would still have held —
    and the count of held months is reported; outside the window there is no label."""
    lf = labels_df.copy()
    win = (lf.index >= rtv.labels.index[0]) & (lf.index <= rtv.labels.index[-1])
    lab = rtv.labels.reindex(lf.index)
    held = int((lab.isna() & win).sum())
    lab[win] = lab[win].ffill()
    lf["hmm_walkforward"] = lab
    probs = rtv.probs.reindex(lf.index)
    probs[win] = probs[win].ffill()
    return lf, probs, held
```

In `run_assets`, add a parameter `rtv=None` at the end of the signature, and inside the `try`, after the `bts = {...}` line:

```python
        rt_bt = None
        if rtv is not None:
            lf_rt, probs_rt_v, held = rt_backtest_inputs(labels_df, rtv)
            rt_runs = {f"cost_bp_{c}": portfolio.backtest(rets, lf_rt, probs_rt_v, strategies=RT_BACKTEST_STRATEGIES,
                                                           include_expost=False, cost_bp=float(c)) for c in (0, 10)}
            rt_bt = {"perf": {k: v.perf.round(4).to_dict(orient="index") for k, v in rt_runs.items()},
                     "counters": rt_runs["cost_bp_0"].counters, "held_months": held,
                     "window": {"start": rt_runs["cost_bp_0"].params["start"],
                                "end": str(rt_runs["cost_bp_0"].returns.index[-1].date())}}
```

and in `block`: `"rt_vintage_backtest": rt_bt,`.

In `main`, pass `rtv=rtv` to `run_assets(...)`, and in the promotion `vals.update({...})` add:

```python
                             "rt_pit_sharpe": ((block.get("rt_vintage_backtest") or {}).get("perf", {}).get("cost_bp_0", {}).get("PIT_MaxSharpe", {}).get("sharpe", float("nan"))),
                             "rt_pit_longonly_sharpe": ((block.get("rt_vintage_backtest") or {}).get("perf", {}).get("cost_bp_0", {}).get("PIT_LongOnly_MaxSharpe", {}).get("sharpe", float("nan"))),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_run.py tests/test_portfolio.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/run.py regime_v2/tests/test_run.py
git commit -m "feat(assets): backtest on the real-time-vintage label, held over gap months, reported beside the others"
```

---

### Task 7: Site documents — placeholders, guard, contract, text

**Files:**
- Modify: `regime_v2/regime_v2/sitedocs.py` — the skipped-assets key list (~line 300), the assets branch end (~line 405), `_IF_LO_PLACEBO_RE` block (~line 445), `_apply_guard`, `render`.
- Modify: `docs/site/CONTRACT.md` (placeholder table, figures list, HTML rule)
- Modify: `docs/site/methodology.md:447-453` (Limitations first paragraph) and `:277` (after the fig 7 line)
- Modify: `docs/site/introduction.md` (one sentence in the "What refreshes" area or the walk-forward paragraph)
- Test: `regime_v2/tests/test_sitedocs.py` (append)

**Interfaces:**
- Consumes: `summary["rt_vintage"]`, `summary["assets"]["rt_vintage_backtest"]`.
- Produces placeholders (all strings): `rt.window_start`, `rt.window_end`, `rt.n_months`, `rt.n_gaps`, `rt.agreement` (`79%`), `rt.agreement_reconstructed`, `rt.agreement_published`, `rt.growth_corr` (`0.97`), `rt.inflation_corr`, `rt.nber_sentence`, `rt.pit`, `rt.pit_longonly`, `rt.pit10`, `rt.pit_longonly10`, `rt.held_months`, `skipped.rt_vintage` (`""` when the block published; otherwise the reason, `"real-time-vintage stage not run"` when absent). Guards `<!-- if:rt_vintage -->…<!-- endif:rt_vintage -->` and `<!-- ifnot:rt_vintage -->…<!-- endif:rt_vintage -->`.
- `rt.nber_sentence`: built from `nber_lags.final` / `.rt`: when every in-window peak has the same `first_low_growth_rt` under both labels → `"For the N recession peaks inside the window (YYYY, YYYY, …) the first low-growth call falls in the same month under both labels."`; otherwise `"The first low-growth call differs for K of the N recession peaks inside the window: <peak YYYY-MM: final YYYY-MM, real-time YYYY-MM; …>."`; `n/a` without the block.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_sitedocs.py`:

```python
def _rt_block(overall=0.787, recon=0.784, pub=0.791, same_peaks=True):
    lag_f = [{"peak": "2001-03", "first_low_growth_rt": "2001-03", "lag_months": 0, "censored": False},
             {"peak": "2007-12", "first_low_growth_rt": "2007-12", "lag_months": 0, "censored": False}]
    lag_r = [dict(r) for r in lag_f]
    if not same_peaks:
        lag_r[1]["first_low_growth_rt"] = "2008-02"; lag_r[1]["lag_months"] = 2
    return {"window": {"start": "1999-07", "end": "2026-07", "n_months": 324}, "start_requested": "1999-07",
            "gaps": {"2025-10": "x"}, "n_gaps": 1, "n_reconstructed": 185, "n_published": 139,
            "agreement": {"overall": overall, "reconstructed": recon, "published": pub},
            "crosstab": {}, "shares": {}, "mean_run_length": {}, "switches": {"final": 50, "rt": 48},
            "gap_revision": {"growth": {"mean": 0.07, "sd": 0.3, "corr": 0.97, "sign_agreement": 0.86},
                             "inflation": {"mean": -0.12, "sd": 0.3, "corr": 0.95, "sign_agreement": 0.89}},
            "nber_lags": {"final": lag_f, "rt": lag_r}, "loader": {}}


def _with_rt(pub, same_peaks=True):
    assets = dict(pub.summary["assets"], rt_vintage_backtest={
        "perf": {"cost_bp_0": {"PIT_MaxSharpe": {"sharpe": 1.179}, "PIT_LongOnly_MaxSharpe": {"sharpe": 0.841}, "Static_6040": {"sharpe": 1.051}},
                 "cost_bp_10": {"PIT_MaxSharpe": {"sharpe": 1.0}, "PIT_LongOnly_MaxSharpe": {"sharpe": 0.8}, "Static_6040": {"sharpe": 1.0}}},
        "counters": {}, "held_months": 1, "window": {"start": "2010-01-01", "end": "2026-08-01"}})
    return replace(pub, summary=dict(pub.summary, rt_vintage=_rt_block(same_peaks=same_peaks), assets=assets))


def test_rt_placeholders_come_from_the_block(pub):
    nums = sitedocs.numbers(_with_rt(pub))
    assert nums["rt.window_start"] == "1999-07" and nums["rt.n_months"] == "324" and nums["rt.n_gaps"] == "1"
    assert nums["rt.agreement"] == "79%" and nums["rt.agreement_reconstructed"] == "78%" and nums["rt.agreement_published"] == "79%"
    assert nums["rt.growth_corr"] == "0.97" and nums["rt.inflation_corr"] == "0.95"
    assert nums["rt.pit"] == "1.18" and nums["rt.pit_longonly"] == "0.84" and nums["rt.pit10"] == "1" and nums["rt.held_months"] == "1"
    assert nums["rt.nber_sentence"] == ("For the 2 recession peaks inside the window (2001, 2007) the first low-growth "
                                        "call falls in the same month under both labels.")
    assert nums["skipped.rt_vintage"] == ""
    nums2 = sitedocs.numbers(_with_rt(pub, same_peaks=False))
    assert nums2["rt.nber_sentence"].startswith("The first low-growth call differs for 1 of the 2 recession peaks")
    assert "2007-12: final 2007-12, real-time 2008-02" in nums2["rt.nber_sentence"]


def test_rt_placeholders_read_na_without_the_block(pub):
    nums = sitedocs.numbers(pub)                       # fixture run has no rt block
    for k in ("rt.window_start", "rt.agreement", "rt.nber_sentence", "rt.pit", "rt.pit_longonly"):
        assert nums[k] == "n/a", k
    assert nums["skipped.rt_vintage"] == "real-time-vintage stage not run"
    skipped = replace(pub, summary=dict(pub.summary, rt_vintage={"skipped": "no vintage archive configured"}))
    assert sitedocs.numbers(skipped)["skipped.rt_vintage"] == "no vintage archive configured"


def test_rt_guards_select_the_measured_or_the_stated_limitation():
    md = ("x <!-- if:assets -->y <!-- if:rt_vintage -->MEASURED<!-- endif:rt_vintage -->"
          "<!-- ifnot:rt_vintage -->STATED<!-- endif:rt_vintage --> z<!-- endif --> w")
    on = _md_text(sitedocs.render(md, {"skipped.assets": "", "skipped.rt_vintage": ""}, {}))
    off = _md_text(sitedocs.render(md, {"skipped.assets": "", "skipped.rt_vintage": "not run"}, {}))
    assert "MEASURED" in on and "STATED" not in on and "<!--" not in on
    assert "STATED" in off and "MEASURED" not in off and "<!--" not in off


def test_documents_state_the_limitation_or_the_measurement_never_both(pub):
    nums_off, nums_on = sitedocs.numbers(pub), sitedocs.numbers(_with_rt(pub))
    for name, path in DOC_PATHS.items():
        src = path.read_text(encoding="utf-8")
        text_off = _md_text(sitedocs.render(src, nums_off, {}))
        text_on = _md_text(sitedocs.render(src, nums_on, {}))
        assert "<!--" not in text_off and "<!--" not in text_on, name
        assert "79%" in text_on and "79%" not in text_off, name
        if name == "methodology":
            assert "ALFRED" in text_off and "ALFRED" not in text_on
            assert "reconstructed" in text_on
```

Also change the existing `test_no_value_is_empty_except_skipped_assets` so its exemption tuple reads
`("skipped.assets", "skipped.lo_placebo", "skipped.rt_vintage")`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_sitedocs.py -q -k "rt_"`
Expected: FAIL — `KeyError: 'rt.window_start'` etc.

- [ ] **Step 3: Implement sitedocs**

In `regime_v2/regime_v2/sitedocs.py`:

1. In the `if skipped_reason:` branch of `numbers`, nothing changes (rt keys are set after the assets branch for both cases).
2. After the assets `if/else` block (just before `return out` of `numbers`), add:

```python
    # -- rt.* — the real-time-vintage comparator (spec 2026-09-25). Absent block, or a
    # skipped stage, reads n/a and drives the if:rt_vintage / ifnot:rt_vintage guards.
    raw_rt = S.get("rt_vintage")
    rt = raw_rt if isinstance(raw_rt, dict) and raw_rt.get("skipped") is None else {}
    out["skipped.rt_vintage"] = ("" if rt else (raw_rt.get("skipped") if isinstance(raw_rt, dict) else "real-time-vintage stage not run"))
    win = rt.get("window") or {}
    out["rt.window_start"] = win.get("start") or _NA
    out["rt.window_end"] = win.get("end") or _NA
    out["rt.n_months"] = str(win["n_months"]) if "n_months" in win else _NA
    out["rt.n_gaps"] = str(rt["n_gaps"]) if "n_gaps" in rt else _NA
    agr = rt.get("agreement") or {}
    out["rt.agreement"] = _pct(agr.get("overall")) if agr.get("overall") is not None else _NA
    out["rt.agreement_reconstructed"] = _pct(agr.get("reconstructed")) if agr.get("reconstructed") is not None else _NA
    out["rt.agreement_published"] = _pct(agr.get("published")) if agr.get("published") is not None else _NA
    rev = rt.get("gap_revision") or {}
    out["rt.growth_corr"] = _num((rev.get("growth") or {}).get("corr")) if (rev.get("growth") or {}).get("corr") is not None else _NA
    out["rt.inflation_corr"] = _num((rev.get("inflation") or {}).get("corr")) if (rev.get("inflation") or {}).get("corr") is not None else _NA
    out["rt.nber_sentence"] = _rt_nber_sentence(rt.get("nber_lags")) if rt else _NA
    rb = (assets_blk.get("rt_vintage_backtest") or {}) if rt else {}
    p0, p10 = (rb.get("perf") or {}).get("cost_bp_0") or {}, (rb.get("perf") or {}).get("cost_bp_10") or {}
    out["rt.pit"] = _num(p0["PIT_MaxSharpe"]["sharpe"]) if "PIT_MaxSharpe" in p0 else _NA
    out["rt.pit_longonly"] = _num(p0["PIT_LongOnly_MaxSharpe"]["sharpe"]) if "PIT_LongOnly_MaxSharpe" in p0 else _NA
    out["rt.pit10"] = _num(p10["PIT_MaxSharpe"]["sharpe"]) if "PIT_MaxSharpe" in p10 else _NA
    out["rt.pit_longonly10"] = _num(p10["PIT_LongOnly_MaxSharpe"]["sharpe"]) if "PIT_LongOnly_MaxSharpe" in p10 else _NA
    out["rt.held_months"] = str(rb["held_months"]) if "held_months" in rb else _NA
```

3. Add the helper near `_placebo_sentence`:

```python
def _rt_nber_sentence(lags: dict | None) -> str:
    """Compare the first low-growth call after each in-window NBER peak under both labels."""
    if not lags or not lags.get("final"):
        return _NA
    fin = {r["peak"]: r.get("first_low_growth_rt") for r in lags["final"]}
    rt = {r["peak"]: r.get("first_low_growth_rt") for r in lags.get("rt", [])}
    peaks = list(fin)
    diff = [p for p in peaks if fin[p] != rt.get(p)]
    years = ", ".join(p[:4] for p in peaks)
    if not diff:
        return (f"For the {len(peaks)} recession peaks inside the window ({years}) the first low-growth "
                "call falls in the same month under both labels.")
    detail = "; ".join(f"{p}: final {fin[p] or 'none'}, real-time {rt.get(p) or 'none'}" for p in diff)
    return (f"The first low-growth call differs for {len(diff)} of the {len(peaks)} recession peaks inside "
            f"the window: {detail}.")
```

4. Guards. Replace the lo_placebo regex block and `_apply_guard` with a generalised form (keep the existing behaviour for `assets` and `lo_placebo`):

```python
_IF_ASSETS_RE = re.compile(r"<!--\s*if:assets\s*-->(.*?)<!--\s*endif\s*-->", re.DOTALL)
# Named guards nest inside if:assets, so each closes with its own tag: the if:assets pattern
# stops at the first bare `<!-- endif -->`, which a shared closing tag would cut short.
_NAMED_GUARDS = ("lo_placebo", "rt_vintage")
_LO_PLACEBO_ABSENT = "long-only placebo not computed for this run"


def _named_guard(text: str, name: str, present: bool) -> str:
    keep_if = re.compile(rf"<!--\s*if:{name}\s*-->(.*?)<!--\s*endif:{name}\s*-->", re.DOTALL)
    keep_ifnot = re.compile(rf"<!--\s*ifnot:{name}\s*-->(.*?)<!--\s*endif:{name}\s*-->", re.DOTALL)
    text = keep_if.sub((lambda m: m.group(1)) if present else "", text)
    return keep_ifnot.sub("" if present else (lambda m: m.group(1)), text)


def _apply_guard(text: str, skipped_assets: str, skipped_lo_placebo: str = "", skipped_rt_vintage: str = "") -> str:
    text = _named_guard(text, "lo_placebo", not skipped_lo_placebo)
    text = _named_guard(text, "rt_vintage", not skipped_rt_vintage)
    if skipped_assets:
        return _IF_ASSETS_RE.sub("", text)
    return _IF_ASSETS_RE.sub(lambda m: m.group(1), text)
```

and in `render`: `text = _apply_guard(markdown_text, nums.get("skipped.assets", ""), nums.get("skipped.lo_placebo", ""), nums.get("skipped.rt_vintage", ""))`.

5. `docs/site/CONTRACT.md`: add rows (before `skipped.lo_placebo`):

```
| `rt.window_start`, `rt.window_end`, `rt.n_months`, `rt.n_gaps` | `1999-07`, `2026-07`, `324`, `1` | the real-time-vintage comparator's window and the months without a usable vintage |
| `rt.agreement`, `rt.agreement_reconstructed`, `rt.agreement_published` | `79%`, `78%`, `79%` | share of months where the label from the vintage available at the time equals the published walk-forward label; split by vintage provenance (≤ 2014-12 reconstructed by the Fed from archived data, later published at the time) |
| `rt.growth_corr`, `rt.inflation_corr` | `0.97`, `0.95` | correlation of the real-time-vintage gap with the published gap |
| `rt.nber_sentence` | `For the 3 recession peaks inside the window (2001, 2007, 2020) the first low-growth call falls in the same month under both labels.` | one sentence, worded for match or mismatch, naming the peaks that differ |
| `rt.pit`, `rt.pit_longonly`, `rt.pit10`, `rt.pit_longonly10`, `rt.held_months` | `1.18`, `0.84`, `1.00`, `0.80`, `1` | PIT max-Sharpe and long-only Sharpe on the real-time-vintage label at 0 and 10 bp, and the gap months whose label was held for the backtest; always shown beside `bt.pit`, `bt.lo_pit` and `bt.sharpe_Static_6040` |
| `skipped.rt_vintage` | `` | empty when `summary.rt_vintage` published; otherwise the reason (`real-time-vintage stage not run` when absent). Text that depends on it goes inside `<!-- if:rt_vintage -->…<!-- endif:rt_vintage -->`; text that states the limitation instead goes inside `<!-- ifnot:rt_vintage -->…<!-- endif:rt_vintage -->` |
```

Add `fig12_rt_vintage` to the figures list sentence, and extend the HTML rule line to name `if:rt_vintage` / `ifnot:rt_vintage`.

6. `docs/site/methodology.md`: replace the first Limitations paragraph (lines 449–453) with:

```markdown
<!-- ifnot:rt_vintage -->
The walk-forward is real-time in estimation but not in data. Each month's model sees only data up to
that month, but it sees the *current* vintage of that data, not the vintage published then. Real
revisions to industrial production and payrolls would make the real-time gaps worse, not better;
reconstructing the pipeline over ALFRED vintages is the single change most likely to move the
headline agreement numbers, and it is out of scope here.
<!-- endif:rt_vintage -->
<!-- if:rt_vintage -->
The walk-forward is real-time in estimation; whether it is real-time in data is measured rather than
assumed. FRED-MD publishes every monthly vintage from 1999-08, so from {{rt.window_start}} each
month is also labelled from the vintage a reader actually had, by the same pipeline, and reported
beside the published label ({{rt.n_months}} months through {{rt.window_end}}; {{rt.n_gaps}}
without a usable vintage). Vintages up to 2014-12 were reconstructed later by the St. Louis Fed
from archived databases and are marked as such. The two labels agree in {{rt.agreement}} of months
({{rt.agreement_reconstructed}} on reconstructed vintages, {{rt.agreement_published}} on published
ones); the real-time gaps correlate {{rt.growth_corr}} (growth) and {{rt.inflation_corr}}
(inflation) with the published gaps. {{rt.nber_sentence}} On the real-time-vintage label the
point-in-time strategies earn {{rt.pit}} unconstrained and {{rt.pit_longonly}} long-only at zero
cost, beside {{bt.pit}} and {{bt.lo_pit}} on the published label and {{bt.sharpe_Static_6040}} for
static 60/40 over the same window ({{rt.pit10}} and {{rt.pit_longonly10}} at ten basis points).
The gap between the two label series is the measured effect of data revision on this pipeline; the
figure below shows where the two disagree.

![Published walk-forward label against the label from the vintage available at the time](fig:fig12_rt_vintage)
<!-- endif:rt_vintage -->
```

Both blocks sit inside the existing `<!-- if:assets -->…<!-- endif -->` section only if the Limitations section is inside it; check with `grep -n "if:assets\|endif" docs/site/methodology.md` — the asset section closes at line ~445 before "Limitations", so these blocks are *outside* `if:assets` and the `bt.*` placeholders inside the `if:rt_vintage` block must degrade gracefully: they read `n/a` when the asset stage was skipped, which is acceptable because `rt.pit` is also `n/a` then (the rt backtest lives in the asset stage). Also amend spec D8's paragraph in the Methodology "Motivation" or "Walk-forward" section if it states "final vintage" as a limitation elsewhere: `grep -n "vintage" docs/site/methodology.md` and add "(see Limitations for the real-time-vintage comparison)" where the walk-forward's data set is described.

7. `docs/site/introduction.md`: after the paragraph that introduces the walk-forward label (find with `grep -n "walk-forward" docs/site/introduction.md`), add:

```markdown
<!-- if:rt_vintage -->
A second check reads each month from the data as first published rather than as revised since:
from {{rt.window_start}} that label agrees with the published one in {{rt.agreement}} of months,
and {{rt.nber_sentence}}
<!-- endif:rt_vintage -->
```

(`rt.nber_sentence` starts with a capital; rewrite the join so it reads as two sentences: `… of months. {{rt.nber_sentence}}`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_sitedocs.py -q` and from the repo root `python -m pytest tests/test_pages_smoke.py -q`
Expected: all PASS, including `test_every_contract_key_is_present` (every new key produced) and the pages smoke test (`fig12_rt_vintage` is in `publish.FIGURES`, so the known-figures set accepts it).

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/sitedocs.py regime_v2/tests/test_sitedocs.py docs/site/CONTRACT.md docs/site/methodology.md docs/site/introduction.md
git commit -m "docs(site): report the real-time-vintage comparison from placeholders behind if:rt_vintage guards"
```

---

### Task 8: Dashboard

**Files:**
- Modify: `app.py` — after the "Full probability series" expander (~line 233), before `# ---------------- Zone 3`.
- Test: `tests/test_pages_smoke.py` (append; root suite)

**Interfaces:**
- Consumes: `pub.figures["fig12_rt_vintage"]`, `S["rt_vintage"]`, `S["assets"]["rt_vintage_backtest"]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pages_smoke.py`. The file already imports `AppTest` and defines `_page_text(at)`;
`app.py` reads `REGIME_OUTPUT_DIR` / `REGIME_FIGS_DIR` from the environment, which is how
`test_app_masthead_links_to_the_two_pages` points it at a run:

```python
def _run_app(monkeypatch, out, figs):
    monkeypatch.setenv("REGIME_OUTPUT_DIR", str(out))
    monkeypatch.setenv("REGIME_FIGS_DIR", str(figs))
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    return at


def test_app_renders_without_rt_block(published_dir, monkeypatch):
    """A run published before the real-time-vintage stage (the fixture) must not show the section."""
    out, figs = published_dir
    text = _page_text(_run_app(monkeypatch, out, figs))
    assert "vintage available at the time" not in text


def test_app_renders_rt_section_when_the_block_exists(published_dir, tmp_path, monkeypatch):
    import json, shutil
    out, figs = published_dir
    out2, figs2 = tmp_path / "output", tmp_path / "figs"
    shutil.copytree(out, out2); shutil.copytree(figs, figs2)
    s = json.loads((out2 / "summary.json").read_text())
    s["rt_vintage"] = {"window": {"start": "1999-07", "end": "2026-07", "n_months": 324}, "n_gaps": 1,
                       "agreement": {"overall": 0.787, "reconstructed": 0.784, "published": 0.791},
                       "gap_revision": {"growth": {"corr": 0.97}, "inflation": {"corr": 0.95}},
                       "n_reconstructed": 185, "n_published": 139, "switches": {"final": 50, "rt": 48}}
    (out2 / "summary.json").write_text(json.dumps(s))
    shutil.copy(figs2 / "fig7_walkforward.png", figs2 / "fig12_rt_vintage.png")
    text = _page_text(_run_app(monkeypatch, out2, figs2))
    assert "vintage available at the time" in text and "79%" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run (repo root): `python -m pytest tests/test_pages_smoke.py -q -k rt_`
Expected: the "without" test passes already; the "when the block exists" test FAILS on the missing text.

- [ ] **Step 3: Implement**

In `app.py`, after the "Full probability series" expander:

```python
rt = S.get("rt_vintage") or {}
if rt.get("skipped") is None and rt.get("window") and pub.figures.get("fig12_rt_vintage"):
    st.subheader("The same label from the vintage available at the time")
    st.image(str(pub.figures["fig12_rt_vintage"]))
    agr, rev = rt["agreement"], rt["gap_revision"]
    rb = ((S.get("assets") or {}).get("rt_vintage_backtest") or {}).get("perf", {}).get("cost_bp_0", {})
    bt_line = ""
    if rb:
        bt_line = (f" On that label the point-in-time strategies earn {rb['PIT_MaxSharpe']['sharpe']:+.2f} unconstrained "
                   f"and {rb['PIT_LongOnly_MaxSharpe']['sharpe']:+.2f} long-only at zero cost, beside "
                   f"{rb['Static_6040']['sharpe']:+.2f} for static 60/40.")
    st.caption(
        f"From {rt['window']['start']} each month is also labelled by the same pipeline from the FRED-MD vintage a "
        f"reader had at the time ({rt['window']['n_months']} months through {rt['window']['end']}; "
        f"{rt.get('n_gaps', 0)} without a usable vintage; vintages up to 2014-12 are the Fed's later reconstructions). "
        f"The two labels agree in {agr['overall']:.0%} of months ({agr['reconstructed']:.0%} reconstructed, "
        f"{agr['published']:.0%} published); the real-time gaps correlate {rev['growth']['corr']:.2f} (growth) and "
        f"{rev['inflation']['corr']:.2f} (inflation) with the published gaps.{bt_line}")
```

(`agr['reconstructed']` may be `None` when the archive has no reconstructed vintages: format with a helper `_pct_or_na = lambda x: 'n/a' if x is None else f'{x:.0%}'` and use it for all three.)

- [ ] **Step 4: Run the tests to verify they pass**

Run (repo root): `python -m pytest tests -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_pages_smoke.py
git commit -m "feat(app): show the real-time-vintage comparison when the run carries it"
```

---

### Task 9: Archive build script and the refresh wiring

**Files:**
- Create: `scripts/build_vintage_archive.py`
- Modify: `scripts/refresh_states.sh:140-147` (both `docker exec` commands gain `--vintage-archive /app/var/vintages`)
- Modify: `.gitignore` (add `regime_v2/data/vintages/`)
- Test: `regime_v2/tests/test_build_vintage_archive.py` (new), `tests/test_refresh_script.py` (append one assertion)

**Interfaces:**
- Produces: `scripts/build_vintage_archive.py` with `main(argv, fetch=urllib.request.urlopen) -> int` and helpers
  `ZIPS = [("historical_fred-md.zip", "1999-08", "2014-12"), ("historical-vintages-of-fred-md-2015-01-to-2025-12.zip", "2015-01", "2025-12")]`,
  `BASE = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/"`,
  `vintage_of(filename) -> str | None` (`"2015-01.csv"`, `"FRED-MD-2025m06.csv"`, `"FRED-MD_2024m10.csv"` → `"YYYY-MM"`),
  `extract_zip(zip_bytes, out_dir) -> list[str]` (vintages written, skipping existing),
  `fetch_monthly(vintages, out_dir, fetch) -> list[str]` (uses `run.fredmd_urls` order: `-rev-` first; `run.looks_like_fredmd`),
  `coverage(out_dir) -> dict` (`first`, `last`, `missing: [..]`).
  CLI: `--out DIR` (default `regime_v2/data/vintages`), `--through YYYY-MM` (default: current month), exit 0 on full coverage, 2 with the missing list printed when gaps remain, 1 on a download error (URL printed).

- [ ] **Step 1: Write the failing tests**

```python
# regime_v2/tests/test_build_vintage_archive.py
import io, sys, zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_vintage_archive as B   # noqa: E402

FRED = b"sasdate,INDPRO\nTransform:,5\n1/1/2020,1.0\n"


def _zip(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(n, FRED)
    return buf.getvalue()


def test_vintage_of_handles_the_three_filename_styles():
    assert B.vintage_of("2015-01.csv") == "2015-01"
    assert B.vintage_of("FRED-MD-2025m06.csv") == "2025-06"
    assert B.vintage_of("FRED-MD_2024m10.csv") == "2024-10"
    assert B.vintage_of("sub/FRED-MD_2024m10.csv") == "2024-10"
    assert B.vintage_of("ReadMe.txt") is None


def test_extract_zip_normalises_names_and_skips_existing(tmp_path):
    out = tmp_path / "v"
    out.mkdir(); (out / "fredmd_2015-01.csv").write_bytes(b"keep")
    written = B.extract_zip(_zip(["2015-01.csv", "FRED-MD-2015m02.csv", "ReadMe.txt"]), out)
    assert written == ["2015-02"]
    assert (out / "fredmd_2015-01.csv").read_bytes() == b"keep"
    assert (out / "fredmd_2015-02.csv").read_bytes() == FRED


def test_fetch_monthly_prefers_the_revised_file_and_skips_holding_pages(tmp_path):
    seen = []

    class R:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def fetch(url, timeout=60):
        seen.append(url)
        if "2026-rev-01" in url:
            return R(FRED)
        if "2026-02" in url:
            return R(b"<html>not yet</html>")
        raise OSError("404")

    got = B.fetch_monthly(["2026-01", "2026-02"], tmp_path, fetch)
    assert got == ["2026-01"] and (tmp_path / "fredmd_2026-01.csv").exists()
    assert not (tmp_path / "fredmd_2026-02.csv").exists()
    assert seen[0].endswith("2026-rev-01-md.csv")


def test_coverage_reports_gaps(tmp_path):
    for v in ("2020-01", "2020-02", "2020-04"):
        (tmp_path / f"fredmd_{v}.csv").write_bytes(FRED)
    assert B.coverage(tmp_path) == {"first": "2020-01", "last": "2020-04", "missing": ["2020-03"]}


def test_main_exit_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ZIPS", [("z.zip", "2020-01", "2020-02")])
    calls = {}

    class R:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def fetch(url, timeout=60):
        calls[url] = 1
        if url.endswith("z.zip"):
            return R(_zip(["2020-01.csv", "2020-02.csv"]))
        return R(FRED)
    assert B.main(["--out", str(tmp_path), "--through", "2020-03"], fetch=fetch) == 0
    assert B.coverage(tmp_path)["missing"] == []

    def broken(url, timeout=60):
        raise OSError("boom")
    assert B.main(["--out", str(tmp_path / "b"), "--through", "2020-03"], fetch=broken) == 1
```

Append to `tests/test_refresh_script.py` (root suite; `ROOT` is already defined there at line 17 and the
file's `run_refresh` fixture drives the script against stub docker — this test only reads the script):

```python
def test_refresh_passes_the_vintage_archive():
    """Both engine invocations name the archive on the volume so the rt stage runs on the Mini."""
    script = (ROOT / "scripts" / "refresh_states.sh").read_text(encoding="utf-8")
    assert script.count("--vintage-archive /app/var/vintages") == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_build_vintage_archive.py -q` (from `regime_v2/`) and `python -m pytest tests/test_refresh_script.py -q` (root)
Expected: FAIL — `ModuleNotFoundError: build_vintage_archive`; the count assertion is 0.

- [ ] **Step 3: Implement**

```python
# scripts/build_vintage_archive.py
"""One-off: build the FRED-MD vintage archive the real-time-vintage stage reads
(spec 2026-09-25 §1). Idempotent: existing files are never overwritten.

    python scripts/build_vintage_archive.py --out regime_v2/data/vintages
    docker exec states python scripts/build_vintage_archive.py --out /app/var/vintages

Exit 0 on full monthly coverage from the first zip's start through --through; 2 when
months are missing (listed); 1 when a download fails (URL printed).
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "regime_v2"))
from run import fredmd_urls, looks_like_fredmd  # noqa: E402

BASE = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/"
ZIPS = [("historical_fred-md.zip", "1999-08", "2014-12"),
        ("historical-vintages-of-fred-md-2015-01-to-2025-12.zip", "2015-01", "2025-12")]
_NAME = re.compile(r"(\d{4})[-m_](\d{2})\.csv$", re.IGNORECASE)


def vintage_of(filename: str) -> str | None:
    m = _NAME.search(Path(filename).name)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def extract_zip(zip_bytes: bytes, out_dir) -> list[str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            v = vintage_of(info.filename)
            if v is None or info.is_dir():
                continue
            dest = out_dir / f"fredmd_{v}.csv"
            if dest.exists():
                continue
            dest.write_bytes(z.read(info))
            written.append(v)
    return sorted(written)


def fetch_monthly(vintages, out_dir, fetch=urllib.request.urlopen) -> list[str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for v in vintages:
        dest = out_dir / f"fredmd_{v}.csv"
        if dest.exists():
            got.append(v); continue
        for url in fredmd_urls(v):
            try:
                with fetch(url, timeout=60) as r:
                    data = r.read()
            except Exception:
                continue
            if looks_like_fredmd(data):
                dest.write_bytes(data); got.append(v); break
    return got


def coverage(out_dir) -> dict:
    have = sorted(vintage_of(p.name) for p in Path(out_dir).glob("fredmd_*.csv"))
    if not have:
        return {"first": None, "last": None, "missing": []}
    want = [str(p) for p in pd.period_range(have[0], have[-1], freq="M")]
    return {"first": have[0], "last": have[-1], "missing": [v for v in want if v not in have]}


def main(argv=None, fetch=urllib.request.urlopen) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "regime_v2" / "data" / "vintages"))
    ap.add_argument("--through", default=date.today().strftime("%Y-%m"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    for name, first, last in ZIPS:
        need = [str(p) for p in pd.period_range(first, last, freq="M") if not (out / f"fredmd_{p}.csv").exists()]
        if not need:
            continue
        url = BASE + name
        try:
            with fetch(url, timeout=600) as r:
                data = r.read()
            written = extract_zip(data, out)
        except Exception as e:
            print(f"download failed: {url}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print(f"{name}: wrote {len(written)} vintages")
    monthly_start = (pd.Period(ZIPS[-1][2], "M") + 1).strftime("%Y-%m")
    monthly = [str(p) for p in pd.period_range(monthly_start, a.through, freq="M")]
    got = fetch_monthly(monthly, out, fetch)
    print(f"monthly files: {len(got)} of {len(monthly)} through {a.through}")
    cov = coverage(out)
    print(f"archive {cov['first']}..{cov['last']}; missing: {cov['missing'] or 'none'}")
    return 2 if cov["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/refresh_states.sh`: add `--vintage-archive /app/var/vintages \` to both `docker exec … run.py` commands (after `--figs-dir /app/var/figs`). `.gitignore`: add `regime_v2/data/vintages/`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_build_vintage_archive.py -q` (from `regime_v2/`) and `python -m pytest tests/test_refresh_script.py -q` (root)
Expected: all PASS. Note `test_main_exit_codes` expects exit 0 when the "monthly" months after the last zip month through `--through` are all served by `fetch` — with `ZIPS` patched to end `2020-02`, the monthly range is `2020-03` only, and `fetch` returns `FRED` for it.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_vintage_archive.py scripts/refresh_states.sh .gitignore regime_v2/tests/test_build_vintage_archive.py tests/test_refresh_script.py
git commit -m "feat(ops): vintage archive build script; the scheduled refresh names the archive on the volume"
```

---

### Task 10: Spec amendment, README, full suites, whole-branch review

**Files:**
- Modify: `docs/SPEC.md` — §4 D8 (append one sentence pointing at the decision-log entry), §5 (module contracts: `rtvintage.py`, `data.load_archive_vintage`), §6 (a "Stage 8 — real-time-vintage comparator" block with `[x]` items), §7 (`fig12_rt_vintage.png`), §10 decision log entry dated 2026-09-25.
- Modify: `README.md` (one paragraph: what `--vintage-archive` does and the build script).
- Modify: `status.md` (Done entry, Next steps: deploy + archive build on the Mini).

- [ ] **Step 1: Write the spec text**

§4, end of D8: `Amended 2026-09-25: a *comparator* label from the vintage available at the time is published beside it from 1999-07 (§6 Stage 8, §10); the primary label is unchanged.`

§10 entry:

```
- 2026-09-25 — Real-time-vintage comparator (design: docs/superpowers/specs/2026-09-25-realtime-vintage-label-design.md). FRED-MD publishes every monthly vintage from 1999-08 (1999–2014 reconstructed by the Fed from archived Haver data, 2015 on published at the time), so the paper's stated limitation — the walk-forward reads final data — is now measured: `rtvintage.fit_hmm4_rt_vintage` runs the unchanged pipeline on vintage t+1 cut at t for every month from 1999-07 and `summary.json["rt_vintage"]` reports agreement with `hmm_walkforward` (overall and by provenance), the cross-table, gap revisions, NBER calls under both labels, loader drops and the months without a usable vintage; the asset stage reruns PIT_MaxSharpe and PIT_LongOnly_MaxSharpe on that label. Archive vintages load through `data.load_archive_vintage` (renames PPIFGS/PPIFCG, never substitutes the NSA CUUR0000SA0L2, drops a series that fails the 0.98 check against the neighbouring vintage, refuses only on an anchor failure or a half-empty block). The stage runs only with `--vintage-archive DIR` and never changes the exit code. The published label and every existing number are unchanged (tested); documents show the measurement behind `if:rt_vintage` and the old limitation behind `ifnot:rt_vintage`. Spike numbers on vintage 2026-08: agreement 0.787, gap corr 0.97/0.95, identical NBER calls, backtest 0.77→1.18 unconstrained and 1.05→0.84 long-only with 45 of 199 months relabelled — reported as the sensitivity of the backtest to labels, never as a verdict. The shutdown files 2025-10..12 end in blank rows; 2025-10 has no vintage row and is a recorded gap.
```

§6 Stage 8 checklist (mark items `[x]` as the tasks above are complete): loader; fit; summary block; run.py wiring; asset backtest; sitedocs/guards/contract; app; build script + refresh; tests listed in spec §7.

`README.md`: add under the engine section: "`--vintage-archive DIR` enables the real-time-vintage comparator (`docs/superpowers/specs/2026-09-25-realtime-vintage-label-design.md`); build DIR once with `python scripts/build_vintage_archive.py --out DIR` (about 305 MB, not in git)."

- [ ] **Step 2: Run both full suites**

Run: `cd regime_v2 && python -m pytest -q` then `cd .. && python -m pytest -q tests`
Expected: all PASS (engine ≈ 8–9 min with the two extra full runs; root ≈ 1.5 min). Record the counts.

- [ ] **Step 3: Commit**

```bash
git add docs/SPEC.md README.md status.md
git commit -m "docs: spec Stage 8 and decision log for the real-time-vintage comparator"
```

- [ ] **Step 4: Whole-branch review**

Use superpowers:requesting-code-review with base `add1018` (the spec commit) and head `HEAD`; fix Important findings; re-run the affected tests; commit.

- [ ] **Step 5: Merge and deploy (after the user's go-ahead)**

```bash
git checkout master && git merge --no-ff rt-vintage -m "Merge rt-vintage: real-time-vintage comparator label from 1999-07"
git push origin master
ssh jameswalsh@JHCW-mini.local 'zsh -lc "cd ~/apps/states && git pull --ff-only && docker compose up -d --build"'
ssh jameswalsh@JHCW-mini.local 'zsh -lc "docker exec states python scripts/build_vintage_archive.py --out /app/var/vintages"'
ssh jameswalsh@JHCW-mini.local 'zsh -lc "docker exec -w /app/regime_v2 states python run.py --vintage 2026-08 --out-dir /app/var/output --figs-dir /app/var/figs --returns-cache /app/var/returns_yfinance.parquet --refresh-returns --vintage-archive /app/var/vintages"'
```

Then confirm on the site: the Methodology Limitations paragraph shows the measurement, fig 12 renders, and `summary.json["rt_vintage"]["agreement"]["overall"]` ≈ 0.79.
