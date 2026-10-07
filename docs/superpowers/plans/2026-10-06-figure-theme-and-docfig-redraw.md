# Figure Theme and Doc-Figure Redraw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Engine PNGs (figures 1–12 and the seven doc figures) carry the lazyeconomist.com palette, and a container start redraws the doc figures from the published run so a code-only deploy never serves stale ones.

**Architecture:** A new Streamlit-free module `regime_v2/regime_v2/theme.py` owns the palette tokens, two rcParams dicts (cream ground for saved PNGs, transparent for the app's inline charts) and a `themed` decorator that wraps a figure function in `matplotlib.rc_context`. `figures.py` and `docfigs.py` import their colours from it and decorate every figure function; the app's `site_theme.py` re-exports the same tokens so there is one source. `publish.redraw_doc_figures(out_dir, figs_dir)` factors the tail of `run.py` (load the published run, draw the doc figures, write the `doc_figures` key back) so both the engine and a new `scripts/redraw_doc_figures.py` call it; `docker/entrypoint.sh` runs the script whenever a published run already exists.

**Tech Stack:** Python 3.12, matplotlib (Agg), Pillow (already a matplotlib dependency) for pixel checks, pytest, POSIX sh entrypoint.

**Spec:** Design approved in conversation on 2026-10-06 (bounded change; no spec file). The binding statements are: (1) the four regime colours in `regimes.COLORS` are unchanged; (2) tab10/tab20 stay for the challenger clusters and the eleven backtest strategies; (3) fonts stay DejaVu Sans (web fonts are not installed in the container); (4) saved PNGs get the cream ground explicitly, the app's inline charts stay transparent; (5) the twelve engine figures are not regenerated at start (they need fitted objects); (6) a failure of the redraw never blocks Streamlit from starting and never changes `run.py`'s exit code.

## Global Constraints

- Every commit message ends with the two trailer lines:
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JZwEFBhS9cHprXhP5y6jr1`.
- Palette tokens, verbatim from `.streamlit/config.toml` and `site_theme.py`: `BG = "#fbfaf7"`, `BG_SOFT = "#f4f2ec"`, `INK = "#1a1a1a"`, `INK_SOFT = "#4a4a4a"`, `INK_FAINT = "#8a8780"`, `RULE = "#e8e4dc"`, `ACCENT = "#b8410e"`, `ACCENT_SOFT = "#f5e6dd"`, `GROWTH_C = "#2C7FB8"`, `INFL_C = "#D95F0E"`.
- `regimes.COLORS` is not edited. `matplotlib.use("Agg")` stays at the top of `figures.py` and `docfigs.py`. `DPI = 130` stays.
- No new dependencies. No network in tests.
- Engine tests run from `regime_v2/` (`pytest.ini` sets `pythonpath = .`); root tests from the repo root. Commands below assume the repo's venv python at `.venv/Scripts/python.exe` (Windows); on the Mini plain `python`.
- `docs/site/CONTRACT.md` is unchanged: no placeholder or figure name changes.

## Review Focus

1. A published run whose `summary.json` predates the `doc_figures` key (any run before 2026-09-05): `redraw_doc_figures` must add the key rather than fail. Pinned by Task 4's `test_redraw_doc_figures_restores_pngs_and_summary_key`.
2. A volume with `summary.json` but no doc PNGs at all (figs dir wiped, or an image rebuilt after the figures were drawn elsewhere): the redraw recreates all seven that the run supports. Same test (all seven PNGs deleted before the call).
3. A themed figure function that raises midway must not leave the cream rcParams active for the next caller (the app's charts would turn opaque). Pinned by Task 1's `test_themed_restores_rcparams_after_an_exception`.
4. Importing `site_theme` must still leave the app's inline charts transparent after the tokens move. Pinned by Task 1's root test `test_site_theme_keeps_inline_charts_transparent`.
5. A doc figure drawn with `fig.suptitle`/`fig.text` and a `tight_layout(rect=...)` (look-ahead, loadings) must still save on the cream ground: the ground is set by rcParams, not by `tight_layout`. Pinned by Task 3's corner-pixel check over every doc PNG.

---

### Task 1: `theme.py` and the `site_theme` re-export

**Files:**
- Create: `regime_v2/regime_v2/theme.py`
- Modify: `site_theme.py` (lines 7–25: the token block and the `plt.rcParams.update`)
- Test: `regime_v2/tests/test_theme.py` (new), `tests/test_app_smoke.py` (one new test at the end)

**Interfaces:**
- Produces: `theme.BG, BG_SOFT, INK, INK_SOFT, INK_FAINT, RULE, ACCENT, ACCENT_SOFT, GROWTH_C, INFL_C` (str hex), `theme.SHADE_ALPHA = 0.15`, `theme.RC` (dict, cream ground), `theme.RC_INLINE` (dict, transparent ground), `theme.CMAP` (a `LinearSegmentedColormap` from `BG` to `ACCENT`, name `"lazyeconomist"`), `theme.themed(fn)` decorator, `theme.tokens_from_streamlit_config(path) -> dict[str, str]`.
- Consumed by Tasks 2 and 3 (`figures.py`, `docfigs.py`) and by `site_theme.py`.

- [ ] **Step 1: Write the failing tests**

`regime_v2/tests/test_theme.py`:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from regime_v2 import theme

ROOT = Path(__file__).resolve().parents[2]


def test_tokens_match_the_streamlit_theme_file():
    cfg = theme.tokens_from_streamlit_config(ROOT / ".streamlit" / "config.toml")
    assert cfg["primaryColor"] == theme.ACCENT
    assert cfg["backgroundColor"] == theme.BG
    assert cfg["secondaryBackgroundColor"] == theme.BG_SOFT
    assert cfg["textColor"] == theme.INK
    assert cfg["borderColor"] == theme.RULE


def test_rc_dicts_differ_only_in_the_ground():
    assert theme.RC["figure.facecolor"] == theme.BG
    assert theme.RC["savefig.facecolor"] == theme.BG
    assert theme.RC_INLINE["figure.facecolor"] == "none"
    assert theme.RC_INLINE["savefig.facecolor"] == "none"
    shared = {k: v for k, v in theme.RC.items() if k not in ("figure.facecolor", "savefig.facecolor")}
    assert shared == {k: v for k, v in theme.RC_INLINE.items() if k not in ("figure.facecolor", "savefig.facecolor")}
    assert theme.RC["axes.edgecolor"] == theme.RULE and theme.RC["text.color"] == theme.INK


def test_themed_applies_inside_and_restores_after():
    before = plt.rcParams["figure.facecolor"]

    @theme.themed
    def inside():
        return plt.rcParams["figure.facecolor"], plt.rcParams["axes.edgecolor"]

    assert inside() == (theme.BG, theme.RULE)
    assert plt.rcParams["figure.facecolor"] == before


def test_themed_restores_rcparams_after_an_exception():
    before = plt.rcParams["figure.facecolor"]

    @theme.themed
    def boom():
        raise RuntimeError("mid-draw failure")

    with pytest.raises(RuntimeError):
        boom()
    assert plt.rcParams["figure.facecolor"] == before


def test_themed_keeps_the_wrapped_name():
    @theme.themed
    def fig99_example(path):
        return path

    assert fig99_example.__name__ == "fig99_example"


def test_cmap_runs_from_ground_to_accent():
    from matplotlib.colors import to_hex
    assert theme.CMAP.name == "lazyeconomist"
    assert to_hex(theme.CMAP(0.0)) == theme.BG and to_hex(theme.CMAP(1.0)) == theme.ACCENT
```

Append to `tests/test_app_smoke.py`:

```python
def test_site_theme_keeps_inline_charts_transparent():
    # The tokens moved to regime_v2.theme; the app must still draw transparent charts on the
    # cream page, and must use the same hex values the engine PNGs use.
    import matplotlib.pyplot as plt
    import site_theme
    from regime_v2 import theme
    assert plt.rcParams["figure.facecolor"] == "none"
    for name in ("BG", "BG_SOFT", "INK", "INK_SOFT", "INK_FAINT", "RULE", "ACCENT", "ACCENT_SOFT",
                 "GROWTH_C", "INFL_C"):
        assert getattr(site_theme, name) == getattr(theme, name), name
```

- [ ] **Step 2: Run them to confirm they fail**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_theme.py -q`
Expected: ImportError, `cannot import name 'theme'`.

Run (from repo root): `.venv\Scripts\python.exe -m pytest tests/test_app_smoke.py::test_site_theme_keeps_inline_charts_transparent -q`
Expected: FAIL at `from regime_v2 import theme`.

- [ ] **Step 3: Write `regime_v2/regime_v2/theme.py`**

```python
"""The lazyeconomist.com palette for every matplotlib figure the engine saves.

One source for the site's tokens (the Streamlit side reads the same values from
.streamlit/config.toml; site_theme.py re-exports these for the app's inline charts).
Nothing here imports Streamlit, so the engine and its tests stay free of it.
Fonts are left at matplotlib's DejaVu Sans: the site's web fonts are loaded by the
browser and are not installed where the PNGs are drawn.
"""
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, BG_SOFT = "#fbfaf7", "#f4f2ec"
INK, INK_SOFT, INK_FAINT, RULE = "#1a1a1a", "#4a4a4a", "#8a8780", "#e8e4dc"
ACCENT, ACCENT_SOFT = "#b8410e", "#f5e6dd"
GROWTH_C, INFL_C = "#2C7FB8", "#D95F0E"
SHADE_ALPHA = 0.15          # NBER recession bands: INK_FAINT at this alpha

# A sequential map from the page ground to the accent, for the transition-matrix heat map.
CMAP = LinearSegmentedColormap.from_list("lazyeconomist", [BG, ACCENT])

_RC_SHARED = {
    "axes.facecolor": "none", "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": INK_SOFT, "axes.titlesize": 10, "axes.edgecolor": RULE, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK_FAINT, "ytick.color": INK_FAINT, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": RULE, "grid.alpha": 0.9, "grid.linewidth": 0.6, "savefig.edgecolor": "none",
}
# Saved PNGs: an explicit cream ground, because the files are also downloaded and viewed alone.
RC = dict(_RC_SHARED, **{"figure.facecolor": BG, "savefig.facecolor": BG})
# The app's inline charts: transparent, so they sit on the page's own ground.
RC_INLINE = dict(_RC_SHARED, **{"figure.facecolor": "none", "savefig.facecolor": "none"})


def themed(fn):
    """Run a figure function under the saved-PNG rcParams and restore the previous ones after,
    whether it returns or raises (matplotlib.rc_context does the restoring)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return fn(*args, **kwargs)
    return wrapper


def tokens_from_streamlit_config(path) -> dict[str, str]:
    """The `key = "value"` pairs of the [theme] table in a Streamlit config.toml, read without
    a TOML dependency: enough for the test that pins these constants to that file."""
    out: dict[str, str] = {}
    in_theme = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_theme = line == "[theme]"
            continue
        if in_theme and "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = v.strip('"')
    return out
```

- [ ] **Step 4: Point `site_theme.py` at it**

Replace lines 7–25 of `site_theme.py` (from `import matplotlib.pyplot as plt` through the closing `})` of `plt.rcParams.update(...)` and the `GROWTH_C, INFL_C = ...` line) with:

```python
import matplotlib.pyplot as plt
import streamlit as st

# ---- site theme: the lazyeconomist.com landing page tokens (fixed light) ----------
# The Streamlit side of the theme lives in .streamlit/config.toml. The hex tokens and the
# rcParams live in regime_v2.theme (one source for the engine's PNGs and these inline
# charts); the inline charts stay transparent so they sit on the page's own ground.
from regime_v2.theme import (  # noqa: E402  (app.py puts regime_v2 on the path first)
    ACCENT, ACCENT_SOFT, BG, BG_SOFT, GROWTH_C, INFL_C, INK, INK_FAINT, INK_SOFT, RC_INLINE, RULE,
)

INK_MUTED, SURFACE = INK_FAINT, BG
FIG_W = 12
plt.rcParams.update(RC_INLINE)
```

Leave `SITE_CSS`, `_masthead`, `_site_pages_nav` and everything after untouched. `app.py`'s import list on lines 47–50 keeps working because every name it imports is still defined in `site_theme`.

- [ ] **Step 5: Run the tests**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_theme.py -q` → 6 passed.
Run (from root): `.venv\Scripts\python.exe -m pytest tests/test_app_smoke.py tests/test_pages_smoke.py -q` → all pass (the published_dir fixture takes ~90 s).

- [ ] **Step 6: Commit**

```bash
git add regime_v2/regime_v2/theme.py regime_v2/tests/test_theme.py site_theme.py tests/test_app_smoke.py
git commit -m "feat(theme): one palette module for engine PNGs and the app's inline charts"
```
(with the two trailer lines from Global Constraints)

---

### Task 2: Restyle `figures.py` (figures 1–12)

**Files:**
- Modify: `regime_v2/regime_v2/figures.py` (whole module; every figure function)
- Test: `regime_v2/tests/test_figures.py`

**Interfaces:**
- Consumes: `theme.themed, BG, INK, INK_SOFT, INK_FAINT, ACCENT, ACCENT_SOFT, GROWTH_C, INFL_C, SHADE_ALPHA` from Task 1.
- Produces: unchanged public signatures `fig1_factors_gaps … fig12_rt_vintage`, `nber_lags`, `strategy_colors`, `_palette`, `_strip`, `_shade`, `_primary` (app.py imports `strategy_colors`; run.py calls all twelve).

- [ ] **Step 1: Write the failing tests**

Add to `regime_v2/tests/test_figures.py`, after `_ok`:

```python
import re

from PIL import Image

from regime_v2 import theme


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _corner(path):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((0, 0))


def test_every_engine_figure_saves_on_the_cream_ground(ctx, tmp_path):
    res, free, gmm, wf = ctx
    p = lambda n: str(tmp_path / n)
    F.fig1_factors_gaps(res, p("fig1.png"))
    F.fig2_regime_timeline(res, wf, free, gmm, p("fig2.png"))
    F.fig3_state_space(res, wf, p("fig3.png"))
    F.fig4_hmm_probabilities(res, wf, p("fig4.png"))
    F.fig5_revisions(res, p("fig5.png"))
    F.fig6_classifier_comparison(res, free, gmm, p("fig6.png"))
    F.fig7_walkforward(res, wf, p("fig7.png"))
    for n in range(1, 8):
        assert _corner(p(f"fig{n}.png")) == _hex_rgb(theme.BG), f"fig{n}"


def test_figures_module_has_no_colour_literals():
    # Every colour comes from regime_v2.theme or regimes.COLORS; the restyle must not erode.
    src = (Path(F.__file__)).read_text(encoding="utf-8")
    assert not re.search(r'"#[0-9a-fA-F]{6}"', src)
    assert not re.search(r'color="(grey|gray|black|white)"', src)


def test_themed_figure_leaves_rcparams_alone(ctx, tmp_path):
    import matplotlib.pyplot as plt
    res, _, _, wf = ctx
    before = plt.rcParams["figure.facecolor"]
    F.fig3_state_space(res, wf, str(tmp_path / "x.png"))
    assert plt.rcParams["figure.facecolor"] == before
```

`from pathlib import Path` goes with the imports at the top. In `test_fig12_rt_vintage_draws_two_strips`, after the size assertion add:

```python
    assert _corner(str(out)) == _hex_rgb(theme.BG)
```

Also extend the asset figures: add after `test_fig12_rt_vintage_draws_two_strips`:

```python
def test_asset_figures_save_on_the_cream_ground(tmp_path):
    idx = pd.date_range("2010-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.005, 0.03, (36, 3)), index=idx,
                        columns=["PIT_MaxSharpe", "Static_6040", "InSample_MaxSharpe_expost"])
    F.fig10_backtest_wealth(rets, str(tmp_path / "f10.png"))
    w = pd.DataFrame(rng.normal(0, 0.3, (36, 4)), index=idx, columns=list("ABCD"))
    F.fig11_pit_weights(w, str(tmp_path / "f11.png"))
    path_df = pd.DataFrame({"mu": rng.normal(0.05, 0.01, 36), "sigma": rng.normal(0.1, 0.01, 36)}, index=idx)
    F.fig9_mixture_6040(path_df, str(tmp_path / "f9.png"))
    table = pd.DataFrame({"ann_ret": [0.05, 0.02, -0.01, 0.03], "se_ann_ret": [0.01] * 4, "n": [20, 15, 10, 5]},
                         index=pd.MultiIndex.from_product([["SPY"], R.REGIMES], names=["asset", "regime"]))
    F.fig8_regime_returns(table, str(tmp_path / "f8.png"))
    for n in (8, 9, 10, 11):
        assert _corner(str(tmp_path / f"f{n}.png")) == _hex_rgb(theme.BG), n
```

- [ ] **Step 2: Run them to confirm they fail**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_figures.py -q`
Expected: the new tests fail (corner pixel is white `(255, 255, 255)`; the literal scan finds `"#1c7ed6"`; `theme` import works from Task 1).

- [ ] **Step 3: Restyle the module**

Edits to `regime_v2/regime_v2/figures.py`:

1. Docstring line 1 becomes: `"""Figures 1-12 (spec §7). 130 dpi, Agg, NBER shading, regime colours from regimes.COLORS, everything else from theme.py (the lazyeconomist.com palette); every figure function is wrapped in theme.themed."""` (keep the rest of the docstring).
2. Imports: add `from .theme import ACCENT, ACCENT_SOFT, GROWTH_C, INFL_C, INK, INK_FAINT, INK_SOFT, SHADE_ALPHA, themed  # noqa: E402` after the `.regimes` import.
3. `_shade`: `color=INK_FAINT, alpha=SHADE_ALPHA` in place of `color="grey", alpha=0.18`.
4. Decorate each of `fig1_factors_gaps, fig2_regime_timeline, fig3_state_space, fig4_hmm_probabilities, fig5_revisions, fig6_classifier_comparison, fig7_walkforward, fig8_regime_returns, fig9_mixture_6040, fig10_backtest_wealth, fig11_pit_weights, fig12_rt_vintage` with `@themed` on the line above `def`. `nber_lags`, `strategy_colors`, `_palette`, `_strip`, `_primary`, `_shade` are not decorated (they draw nothing on their own or are called inside a themed function).
5. Colour replacements, every occurrence:
   - `"#1c7ed6"` → `GROWTH_C` (fig1 level line, fig9 expected-return line)
   - `"#e8590c"` → `INFL_C` (fig1 trend line, fig9 volatility line)
   - `"#2b8a3e"` → `ACCENT` (fig1 gap line)
   - `color="grey"` → `color=INK_FAINT` (zero lines and axis cross-hairs in fig1, fig3, fig5, fig6, fig8, fig11)
   - fig5: give the two lines explicit colours: real-time `color=GROWTH_C`, ex-post `color=INK_FAINT` (keep `alpha=0.8` on the ex-post line).
   - fig12: `color="black"` on the disagreement ticks → `color=INK`; reconstructed span `color="#c9a227", alpha=0.10` → `color=ACCENT, alpha=0.08`; annotation `color="#7a5c00"` → `color=ACCENT`.
   - fig10 legend: keep `framealpha=0.9` but the frame is off by rcParams (`legend.frameon False`); add `frameon=True` to that one legend call so the eleven entries stay readable over the curves.
6. Nothing else changes: titles, sizes, `DPI`, `tight_layout`, tab10/tab20 palettes.

- [ ] **Step 4: Run the tests**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_figures.py -q` → all pass.
Then the suites that call figures through `run.py`: `..\.venv\Scripts\python.exe -m pytest tests/test_run.py tests/test_publish.py -q` → pass.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/figures.py regime_v2/tests/test_figures.py
git commit -m "style(figures): figures 1-12 on the site palette via theme.themed"
```
(with the trailer lines)

---

### Task 3: Restyle `docfigs.py` (the seven doc figures)

**Files:**
- Modify: `regime_v2/regime_v2/docfigs.py` (constants block lines 28–33; every `_draw_*`; `write_doc_figures` docstring)
- Test: `regime_v2/tests/test_docfigs.py`

**Interfaces:**
- Consumes: `theme.themed, BG, BG_SOFT, INK, INK_SOFT, INK_FAINT, RULE, ACCENT, ACCENT_SOFT, GROWTH_C, INFL_C, CMAP` from Task 1.
- Produces: unchanged `DOC_FIGURES`, `write_doc_figures(pub, figs_dir)`, `_draw_*` names (tests and `publish.py` import them).

- [ ] **Step 1: Write the failing tests**

Add to `regime_v2/tests/test_docfigs.py`:

```python
import re
from pathlib import Path

from PIL import Image

from regime_v2 import theme


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _corner(path):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((0, 0))


def test_every_doc_figure_saves_on_the_cream_ground(published_dir, tmp_path):
    out, figs = published_dir
    pub = P.load_published(out, figs)
    result = docfigs.write_doc_figures(pub, tmp_path)
    drawn = {k: v for k, v in result.items() if v is not None}
    assert set(drawn) >= ENGINE_ONLY
    for name, path in drawn.items():
        assert _corner(path) == _hex_rgb(theme.BG), name


def test_docfigs_module_has_no_colour_literals():
    src = Path(docfigs.__file__).read_text(encoding="utf-8")
    assert not re.search(r'"#[0-9a-fA-F]{6}"', src)
    assert not re.search(r'(color|fc|ec)="(grey|gray|black|white)"', src)
    assert 'cmap="Blues"' not in src
```

- [ ] **Step 2: Run them to confirm they fail**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_docfigs.py -q`
Expected: the two new tests fail (white corners; `"#212529"` found).

- [ ] **Step 3: Restyle the module**

Edits to `regime_v2/regime_v2/docfigs.py`:

1. Docstring sentence "Same visual language as figures.py: regime colours from `regimes.COLORS`, everything else a light neutral palette, 130 dpi, Agg." → "Same visual language as figures.py: regime colours from `regimes.COLORS`, everything else from `theme.py` (the lazyeconomist.com palette), every drawer wrapped in `theme.themed`, 130 dpi, Agg."
2. Replace the constants block

```python
INK = "#212529"
LINE = "#495057"
NEUTRAL = "#adb5bd"
NEUTRAL_LIGHT = "#eef2f6"
BLUE = "#1c7ed6"
ORANGE = "#e8590c"
```
with
```python
from .theme import (  # noqa: E402
    ACCENT, BG, BG_SOFT, CMAP, GROWTH_C, INFL_C, INK, INK_FAINT, INK_SOFT, RULE, themed,
)

# Local names kept so the drawing code reads as before; each is a theme token.
LINE = INK_SOFT            # arrows, box edges, reference lines, secondary text
NEUTRAL = INK_FAINT        # hysteresis bands, placebo histogram bars, look-ahead step bars
NEUTRAL_LIGHT = BG_SOFT    # pipeline stage boxes
BLUE = GROWTH_C            # positive loadings, achievable Sharpe bars
ORANGE = INFL_C            # negative loadings
```
3. Decorate `_draw_pipeline, _draw_quadrants, _draw_timing, _draw_lookahead, _draw_placebo, _draw_loadings, _draw_transition` with `@themed`.
4. Literal replacements:
   - `_draw_quadrants` star: `color="black"` → `color=INK`, `edgecolor="white"` → `edgecolor=BG`.
   - `_draw_lookahead` callout bbox: `fc="white"` → `fc=BG`.
   - `_draw_placebo`: `edgecolor="white"` → `edgecolor=BG`; the real-value line `color=ORANGE` → `color=ACCENT` (the accent marks the real statistic against the grey null).
   - `_draw_transition`: `cmap="Blues"` → `cmap=CMAP`; cell text `color="white" if v > 0.5 else INK` → `color=BG if v > 0.5 else INK`.
   - Any remaining `color="grey"`/`"gray"`/`"black"`/`"white"` → `INK_FAINT`/`INK`/`BG` by the same rule (grey → `INK_FAINT`, black → `INK`, white → `BG`).
5. Nothing else changes: layouts, numbers, titles, `write_doc_figures` logic.

- [ ] **Step 4: Run the tests**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_docfigs.py tests/test_sitedocs.py -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add regime_v2/regime_v2/docfigs.py regime_v2/tests/test_docfigs.py
git commit -m "style(docfigs): the seven doc figures on the site palette; ground-to-accent heat map"
```
(with the trailer lines)

---

### Task 4: `publish.redraw_doc_figures`, the redraw script, and the entrypoint

**Files:**
- Modify: `regime_v2/regime_v2/publish.py` (add one function after `load_published`)
- Modify: `regime_v2/run.py` (lines ~678–691: the doc-figure block at the end of `main`)
- Create: `scripts/redraw_doc_figures.py`
- Modify: `docker/entrypoint.sh`
- Test: `regime_v2/tests/test_publish.py`, `tests/test_refresh_schedule.py`, `tests/test_redraw_script.py` (new)

**Interfaces:**
- Consumes: `publish.load_published(out_dir, figs_dir)`, `publish.PublishedMissing`, `docfigs.write_doc_figures(pub, figs_dir)`, `publish.FILES["summary"]`.
- Produces: `publish.redraw_doc_figures(out_dir, figs_dir) -> dict[str, str | None]` — redraws the doc figures from the published run on disk, writes `summary["doc_figures"]` (file names, not paths) back to `summary.json`, returns that mapping; raises `PublishedMissing` when there is no published run; any other exception propagates (callers decide). `scripts/redraw_doc_figures.py` exits 0 on success, 1 on `PublishedMissing` or any failure (message on stderr), and accepts `--out-dir`/`--figs-dir` defaulting to `REGIME_OUTPUT_DIR`/`REGIME_FIGS_DIR` then `regime_v2/output`/`regime_v2/figs`.

- [ ] **Step 1: Write the failing tests**

Append to `regime_v2/tests/test_publish.py`:

```python
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
```
(`import pytest` at the top of the file if absent.)

New `tests/test_redraw_script.py` (root):

```python
"""scripts/redraw_doc_figures.py: the entrypoint's redraw of the doc figures from a published run."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "redraw_doc_figures.py"


def test_script_redraws_from_a_published_run(published_dir):
    out, figs = published_dir
    (figs / "doc_pipeline.png").unlink(missing_ok=True)
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(out), "--figs-dir", str(figs)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert (figs / "doc_pipeline.png").exists()
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["doc_figures"]["doc_pipeline"] == "doc_pipeline.png"
    assert "doc figures redrawn" in r.stdout


def test_script_exits_1_without_a_published_run(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(tmp_path / "o"), "--figs-dir", str(tmp_path / "f")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1
    assert "no published run" in r.stderr


def test_script_reads_the_container_env(tmp_path, monkeypatch):
    # The entrypoint passes no flags; the defaults are the Dockerfile's REGIME_* variables.
    import os
    env = dict(os.environ, REGIME_OUTPUT_DIR=str(tmp_path / "o"), REGIME_FIGS_DIR=str(tmp_path / "f"))
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 1 and str(tmp_path / "o") in r.stderr
```

Append to `tests/test_refresh_schedule.py`:

```python
def test_entrypoint_redraws_doc_figures_when_a_run_is_published():
    text = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    head, _, tail = text.partition("else")
    assert "run.py data/fredmd_2026-07.csv" in head            # first start: the pinned run, as before
    assert "scripts/redraw_doc_figures.py" in tail              # later starts: redraw from the volume
    assert "scripts/redraw_doc_figures.py" not in head
    assert "|| echo" in tail                                   # a failure never blocks streamlit
    assert text.rstrip().endswith('--server.port="${STREAMLIT_SERVER_PORT:-8505}"')
```
(`ROOT` is already defined in that file; check and reuse.)

- [ ] **Step 2: Run them to confirm they fail**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_publish.py -q -k redraw` → AttributeError, no `redraw_doc_figures`.
Run (from root): `.venv\Scripts\python.exe -m pytest tests/test_redraw_script.py tests/test_refresh_schedule.py -q` → script missing (returncode 2), entrypoint test fails on the `else` partition.

- [ ] **Step 3: Add `redraw_doc_figures` to `publish.py`**

After `load_published`:

```python
def redraw_doc_figures(out_dir, figs_dir) -> dict[str, "str | None"]:
    """Redraw the seven documentation figures from the run published in `out_dir` and record
    their file names under `summary["doc_figures"]`.

    The doc figures are drawn from the published bundle, not from fitted objects, so they can
    be refreshed without an engine run: run.py calls this at the end of every run, and the
    container entrypoint calls it on every start so a code-only deploy never serves PNGs
    drawn by an older docfigs.py. Raises PublishedMissing when there is nothing published;
    other failures propagate to the caller.
    """
    from .docfigs import write_doc_figures   # local: docfigs is imported lazily to keep this module light
    out_dir, figs_dir = Path(out_dir), Path(figs_dir)
    pub = load_published(out_dir, figs_dir)
    paths = write_doc_figures(pub, figs_dir)
    names = {k: (v.name if v is not None else None) for k, v in paths.items()}
    summary_path = out_dir / FILES["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["doc_figures"] = names
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return names
```
(`publish.py` already imports `json` and `Path`; `from .docfigs import DOC_FIGURES` already exists at module level, so the local import may instead be a module-level `from .docfigs import DOC_FIGURES, write_doc_figures` — either is fine; pick the module-level one if no circular import results, which it does not: docfigs imports only `.regimes` and `.theme`.)

- [ ] **Step 4: Make `run.py` call it**

Replace the block at the end of `main` (from the comment `# Doc figures (docs/site/CONTRACT.md): drawn on every run` through the `except Exception as e:` print) with:

```python
    # Doc figures (docs/site/CONTRACT.md): drawn on every run, whether the asset stage
    # published, was skipped, or was not requested (the two asset-dependent figures are
    # then simply absent). They are drawn from the published bundle on disk, so the same
    # call serves the container entrypoint (scripts/redraw_doc_figures.py). A failure here
    # must never change the exit code: output/ and figs/ are already published.
    try:
        summary["doc_figures"] = redraw_doc_figures(out_dir, figs_dir)
    except Exception as e:
        print(f"doc figures failed: {type(e).__name__}: {e}", file=sys.stderr)
    return 0
```
and add `redraw_doc_figures` to the existing `from regime_v2.publish import ...` line (or `from regime_v2 import publish` usage, whichever `run.py` has for `load_published`; keep the style of that import). Remove the now-unused `docfigs` name from `run.py`'s imports if nothing else in `run.py` uses it (grep first: `docfigs.` elsewhere in `run.py`).

- [ ] **Step 5: Write `scripts/redraw_doc_figures.py`**

```python
"""Redraw the documentation figures from the published run, without an engine run.

The container entrypoint calls this on every start that finds a published run on the volume,
so a deploy that only changed docfigs.py (or theme.py) shows its figures at once. The twelve
engine figures need fitted objects and are refreshed by the next engine run instead.

Exit 0 when the figures were redrawn, 1 when there is no published run or the redraw failed
(the message goes to stderr; the entrypoint prints it and starts the app regardless).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "regime_v2"))

from regime_v2.publish import PublishedMissing, redraw_doc_figures  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.environ.get("REGIME_OUTPUT_DIR", str(ROOT / "regime_v2" / "output")))
    ap.add_argument("--figs-dir", default=os.environ.get("REGIME_FIGS_DIR", str(ROOT / "regime_v2" / "figs")))
    a = ap.parse_args(argv)
    try:
        names = redraw_doc_figures(a.out_dir, a.figs_dir)
    except PublishedMissing as e:
        print(f"no published run to redraw from: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"doc figures not redrawn: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    drawn = sorted(k for k, v in names.items() if v)
    skipped = sorted(k for k, v in names.items() if not v)
    print(f"doc figures redrawn in {a.figs_dir}: {', '.join(drawn)}" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Update `docker/entrypoint.sh`**

```sh
#!/bin/sh
# First start: publish the pinned vintage (offline-capable) so the site is never empty.
# Later starts: redraw the documentation figures from the published run on the volume, so a
# deploy that changed only how they are drawn shows them at once (the engine figures wait for
# the next refresh). Neither step may stop the app from starting.
set -e
mkdir -p /app/var
if [ ! -f /app/var/output/summary.json ]; then
  echo "no published run on the volume; publishing the pinned vintage"
  (cd /app/regime_v2 && python run.py data/fredmd_2026-07.csv --out-dir /app/var/output --figs-dir /app/var/figs \
      --returns-cache /app/var/returns_yfinance.parquet) || echo "pinned run failed; the app will show the empty state"
else
  python scripts/redraw_doc_figures.py || echo "doc figures not regenerated; serving the ones already published"
fi
exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.port="${STREAMLIT_SERVER_PORT:-8505}"
```
(The script reads `REGIME_OUTPUT_DIR`/`REGIME_FIGS_DIR` from the Dockerfile's ENV, so no flags.) Keep LF line endings (check with `git diff --stat`; the file is a shell script run by `sh` in the image).

- [ ] **Step 7: Run the tests**

Run (from `regime_v2/`): `..\.venv\Scripts\python.exe -m pytest tests/test_publish.py tests/test_docfigs.py tests/test_run.py -q` → pass.
Run (from root): `.venv\Scripts\python.exe -m pytest tests/test_redraw_script.py tests/test_refresh_schedule.py -q` → pass.

- [ ] **Step 8: Commit**

```bash
git add regime_v2/regime_v2/publish.py regime_v2/run.py scripts/redraw_doc_figures.py docker/entrypoint.sh \
        regime_v2/tests/test_publish.py tests/test_redraw_script.py tests/test_refresh_schedule.py
git commit -m "feat(ops): redraw the doc figures from the published run at container start"
```
(with the trailer lines)

---

### Task 5: Documents

**Files:**
- Modify: `docs/SPEC.md` (§7 closing line; §10 new entry; §12 step 2)
- Modify: `status.md` (the "Next steps" list and the dated progress section)

**Interfaces:** none; text only.

- [ ] **Step 1: SPEC §7**

Replace the closing line of §7

`All figures: 130 dpi, `matplotlib.use("Agg")`, NBER recessions shaded from the `NBER` list in `run.py`, regime colours from `regimes.COLORS`.`

with

`All figures: 130 dpi, `matplotlib.use("Agg")`, NBER recessions shaded from the `NBER` list, regime colours from `regimes.COLORS`, every other colour and the rcParams from `regime_v2/regime_v2/theme.py` (the lazyeconomist.com palette: cream ground `#fbfaf7`, ink `#1a1a1a`, rust accent `#b8410e`, growth blue `#2C7FB8`, inflation orange `#D95F0E`), applied per figure by the `theme.themed` decorator; `site_theme.py` re-exports the same tokens for the dashboard's inline charts, which stay transparent. Fonts are matplotlib's DejaVu Sans (the site's web fonts are not installed where the PNGs are drawn). The seven documentation figures are drawn from the published bundle, so `publish.redraw_doc_figures` refreshes them without an engine run; the container entrypoint calls it on every start that finds a published run.`

- [ ] **Step 2: SPEC §10 entry**

Append under §10 Decision log:

`- 2026-10-06 — Figures on the site palette; doc figures redrawn at container start. `theme.py` is the one source for the engine PNGs' colours and rcParams (`site_theme.py` re-exports it for the app). The regime colours, tab10 for the challenger clusters and tab20 for the eleven strategies are unchanged; the ad-hoc Open Color blues/oranges/greys are gone, the transition heat map runs from the page ground to the accent, and saved PNGs carry the cream ground explicitly because they are also downloaded as files. Fonts stay DejaVu Sans: vendoring the web fonts into the image is a separate decision. The doc figures were already drawn from the published bundle; `publish.redraw_doc_figures(out_dir, figs_dir)` now factors that tail of `run.py` and `scripts/redraw_doc_figures.py` calls it from `docker/entrypoint.sh` whenever a run is published on the volume, so a deploy that touches only `docfigs.py`/`theme.py` shows its figures at once. The twelve engine figures need fitted objects and wait for the next engine run; a redraw failure prints and never blocks Streamlit.`

- [ ] **Step 3: SPEC §12**

In §12's numbered deployment list, extend step 2 (`docker compose up -d --build` …) with: `On every later start the entrypoint redraws the seven doc figures from the volume (seconds) before serving; the engine figures change only on a refresh.`

- [ ] **Step 4: status.md**

In the "Next steps, in order" list remove item 1 (`**Small items:** doc-figure regeneration in the container entrypoint; figure restyle.`) and renumber. Add to the dated progress section, above the paragraph that begins "Everything above is merged", a bullet:

`- **Figures on the site palette; doc figures redrawn at start** (2026-10-06). `regime_v2/regime_v2/theme.py` holds the palette and rcParams for every saved PNG (`themed` decorator on all nineteen figure functions), `site_theme.py` re-exports it; regime colours unchanged, DejaVu Sans kept. `publish.redraw_doc_figures` + `scripts/redraw_doc_figures.py` + the entrypoint's else branch refresh the doc figures without an engine run.`

- [ ] **Step 5: Commit**

```bash
git add docs/SPEC.md status.md
git commit -m "docs: spec §7/§10/§12 and status for the figure theme and the doc-figure redraw"
```
(with the trailer lines)

---

## Self-review notes

- Spec coverage: tokens/one source (T1), decorator and cream ground (T1–T3), regime/tab palettes untouched (T2 §3.6, T3 §3.5), DejaVu (T1 docstring, T5), redraw function + script + entrypoint with non-blocking failure (T4), engine figures not regenerated (T4 script docstring, T5), docs (T5).
- Type consistency: `redraw_doc_figures` returns `dict[str, str | None]` of file names in T4 and is what `run.py` stores and the script prints; `themed` is a plain decorator everywhere; `CMAP` is defined in T1 and used in T3.
- Review Focus items 1–5 each name their pinning test (T4 restore test, T1 exception test, T1 root transparency test, T3 corner check).
