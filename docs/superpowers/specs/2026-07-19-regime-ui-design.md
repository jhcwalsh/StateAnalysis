# Regime Dashboard UI — Design Spec

**Date:** 2026-07-19
**Status:** Approved (design); implementation not started
**Approach:** Streamlit viewer over notebook exports (Option A of three considered)

## Purpose

A local, single-user UI over `Macro_Regime_Analysis.ipynb` serving three jobs
in priority order:

1. **Glanceable status** — current regime, soft probabilities, data vintage
2. **Exploration** — one live parameter: hysteresis theta
3. **Presentation** — the paper's tables, figures, and backtest in a readable page

Non-goals: multi-user access, hosting, auth, live parameters other than theta,
any change to the notebook's economics.

## Requirements (from brainstorm)

- Local machine only (`streamlit run app.py`, localhost)
- Opens instantly from cached results of the last notebook run
- Refresh button triggers a full notebook re-execution (~5 min); user decides
  when data is stale (FRED updates ~monthly)
- Theta is the only interactive parameter. Moving it relabels regimes
  instantly from cached gap series. All other knobs (COVID window, asset
  universe, backtest settings) change only via notebook config + Refresh.

## Architecture

The notebook remains the single source of truth for all economics. The app is
a viewer. Exactly one piece of logic is shared, via a module imported by both:

```
regime_core.py      shared logic, moved VERBATIM from notebook cell 20:
                      _hysteretic_sign(series, theta)   (Schmitt trigger)
                      QUAD_BY_SIGN                      (sign-pair -> quadrant)
                      assign_quadrants(g, p, theta)
                      duration_stats(labels)            (~60 lines total)

app.py              Streamlit viewer (~250 lines), no economics beyond
                    calling regime_core for theta relabeling

ui_data/            gitignored; written by notebook, read by app
```

### Notebook changes (deliberately minimal)

1. Cell 20: delete the local trigger definitions, `from regime_core import ...`;
   `assign_quadrants` takes an explicit `theta` (module code cannot see the
   notebook's `HYSTERESIS_THETA` global), so its two call sites gain the arg.
   Numbers must be bit-identical after the switch (verified — see Testing).
2. Cell 51: paper-convention constants (`CODE_TO_PAPER`, `PAPER_DISPLAY`,
   `PAPER_COLORS`, `PAPER_ORDER`) move to `regime_core` and are imported —
   the app needs them for badges/colors, and duplicating them would drift.
3. Cell 60 (Cell J): one-line call-site change — `assign_quadrants(pit_g,
   pit_p, HYSTERESIS_THETA)`.
4. One new export cell at the end of the notebook writes `ui_data/`.
5. `.gitignore` gains `ui_data/`.

No other notebook cell changes. A small `ui_io.py` (data loading, schema
guard, refresh runner) sits between `ui_data/` and `app.py` so those pieces
are unit-testable without Streamlit.

### Data contract (`ui_data/`)

| File | Contents |
|---|---|
| `gaps.parquet` | full-sample gap series (`g_gap`, `p_gap`), PIT endpoint series (`pit_g`, `pit_p`), factors |
| `labels.parquet` | `quad` (full-sample), `pit_quad`, GMM cluster labels |
| `probs.parquet` | quadrant probabilities, paper convention (`probs_paper`) |
| `returns.parquet` | monthly simple returns (`ret_bt`) |
| `backtest.parquet` | `bt_returns` strategy return series |
| `tables.xlsx` | existing `macro_regime_results.xlsx` (Tables 1–3) |
| `meta.json` | schema_version, run timestamp, theta used, COVID window, sample ranges, current-regime summary, PIT/final agreement stats, backtest perf summary |

PNG figures already exported by Cell H are read from the project root.

Writes go to `ui_data/.tmp/` then atomically rename into place, so a failed
run never leaves half-written data; the app keeps serving the previous run.

## UI layout (single page, four zones)

1. **Status header** — current-regime badge colored with `PAPER_COLORS`; the
   four soft probabilities as a horizontal bar; current gap values with the
   theta band marked; data-vintage line from `meta.json` ("FRED through
   2026-06 · run 2026-07-19 · PIT/final agreement 64.6%").
2. **Explore** — theta slider ∈ [0, 1], default and marker at the run's
   value. Live updates: regime timeline strip, occupancy table, average
   duration / switch count. Fixed note stating that tables and backtest below
   are pinned to run theta and require Refresh.
3. **Results** — tabs: Regime returns (Table 1) · Correlations (Table 2, with
   SPY–AGG spotlight) · Portfolios (Table 3) · State space (PNG) · Backtest
   (wealth curves, performance table, look-ahead decomposition).
4. **Refresh** — button runs `jupyter nbconvert --execute` as a subprocess
   (30-min timeout), spinner with elapsed time, reloads caches on success,
   shows tail of log on failure. Button disabled while a run is in flight
   (session-state guard).

The live/pinned boundary is always labeled in the UI: theta relabeling
(labels, occupancy, durations, timeline) is instant and client-driven;
regime-conditional moments, tables, and backtest are pinned to the run.

## Error handling

- No `ui_data/` on launch → friendly empty state: "No cached run found —
  press Refresh (≈5 min)". No stack traces.
- `meta.json` schema_version mismatch → refuse to render data, instruct
  re-running the notebook. Never misrender old-format data.
- Refresh subprocess failure → previous data stays live; log tail shown.
- Missing individual file → treat as schema mismatch (same message).

## Testing

1. **`tests/test_regime_core.py`** (pytest):
   - causality: trigger on a prefix equals prefix of trigger on full series
   - theta=0 equals memoryless sign classification
   - the synthetic flip-count case (noisy series: 8 switches at theta=0,
     2 at theta=0.25)
2. **Bit-identity check** (one-off, on the cell-20 import switch): execute the
   notebook immediately BEFORE the switch to capture a same-day baseline of
   `quad` value counts, average duration, and switch count (as of 2026-07-19:
   73/120/104/79 at theta=0.50, 4.6 mo, 81 switches — but the check compares
   run-to-run, not against these hardcoded values, since FRED data moves).
   Execute again after the switch; any difference fails the refactor.
3. **App smoke test**: launch headless against a fixture `ui_data/`, assert
   the status header renders.

## Dependencies

Add to `requirements.txt`: `streamlit`, `pyarrow` (parquet), `pytest`.

## Alternatives considered

- **B — full pipeline extraction to a module**: cleanest long-term; rejected
  for now as a high-risk refactor of a stable, committed, paper-backing
  notebook. Option A creates the `regime_core.py` seam B would grow from.
- **C — self-contained HTML + JS theta slider**: zero-server and shareable,
  but a static page cannot run the pipeline, failing the refresh-button
  requirement.

## Build order (for the implementation plan)

1. `regime_core.py` + tests + notebook cell-20 import switch + bit-identity check
2. Notebook export cell + `ui_data/` contract + gitignore
3. `app.py` zones 1–2 (status, explore)
4. `app.py` zones 3–4 (results, refresh)
5. Smoke test, README note on launching
