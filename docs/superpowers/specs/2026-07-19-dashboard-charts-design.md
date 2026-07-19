# Dashboard Charts: Factors, Probabilities, State Space

**Date:** 2026-07-19
**Status:** Approved
**Extends:** `2026-07-19-regime-ui-design.md` (Zone 3, Results tabs)

## Goal

Three new charts in the Results zone of `app.py`, all read from the existing
`ui_data/` bundle. No changes to the notebook, the data contract, or `ui_io`.

## Tab layout

Factors · Probabilities · Regime returns · Correlations · Portfolios ·
State space · Backtest — two new tabs; State space upgraded from static PNG
to a live chart.

## Charts

**Factors tab** — two stacked time-series panels from `gaps.parquet`:

- Top: the classification inputs `g_gap` and `p_gap`, a zero line, and the
  live ±θ dead band shaded (θ from the Zone 2 slider).
- Bottom: the underlying factor levels `growth_factor` and
  `inflation_factor`.

**Probabilities tab** — stacked area chart of the four regime probabilities
over time from `probs.parquet`, in `PAPER_COLORS` / `PAPER_ORDER`. Caption
notes these are the run's marginalized GMM probabilities, pinned to run θ
(they cannot be re-labeled live).

**State space tab** — live scatter of `g_gap` vs `p_gap`:

- Each month colored by the **live θ slider's** regime labels (reuses the
  `live` series Zone 2 already computes via `regime_core.assign_quadrants`).
- Quadrant reference lines at zero; the ±θ dead band shaded.
- Latest month highlighted with a distinct marker and date label.
- The notebook's static `state_space_regimes.png` moves into a collapsed
  expander below as the published-run reference.

## Testing

Extend `tests/test_app_smoke.py`: the app renders without exception with the
fixture bundle and the new tabs are present. Re-verify against real
`ui_data/` with the AppTest harness before committing.
