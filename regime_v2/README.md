# regime_v2 — growth/inflation regime engine

Source of truth: `../docs/SPEC.md`. Reference numbers: `../docs/regime_v2_prototype.zip`.

    ../.venv/Scripts/python.exe -m pip install -r requirements.txt
    ../.venv/Scripts/python.exe run.py data/fredmd_2026-07.csv        # full run, ~90 s (walk-forward)
    ../.venv/Scripts/python.exe run.py data/fredmd_2026-07.csv --no-walkforward --skip-robustness   # ~1 min
    ../.venv/Scripts/python.exe run.py --vintage 2026-08               # download then run
                                                                        # (URL pattern verified 2026-09-04 against the
                                                                        # FRED-MD page; not yet exercised from this
                                                                        # machine — a manual download into
                                                                        # data/fredmd_YYYY-MM.csv works as a fallback)
    ../.venv/Scripts/python.exe -m pytest -q                           # ~2 min
    RUN_SLOW=1 ../.venv/Scripts/python.exe -m pytest -q tests/test_acceptance.py   # full walk-forward acceptance

Outputs: `output/regime_labels.csv` (primary label = `hmm_walkforward`; join on `available_at`),
`output/summary.json` (transition matrix rows = from-state), `output/acceptance.csv`, `figs/fig1..fig7.png`,
`data/README.md` (data sheet). A run that fails any acceptance threshold exits 1 and leaves `output.staging/`.

After a successful publish, `run.py` runs the asset stage (Stage 5-6) unless `--no-assets` is given: it
loads/caches the 11-ETF universe (`--returns-cache PATH`, default `data/returns_yfinance.parquet`;
`--refresh-returns` forces a re-download), joins returns to labels strictly through `available_at`, and writes
`output/regime_returns.csv`, `output/regime_corr_<regime>.csv`, `output/backtest_returns.csv`,
`output/portfolio_weights.csv` and `figs/fig8..fig11.png`. It also runs a 60/40 achievable backtest at 0bp and
10bp cost, a PIT/oracle/in-sample look-ahead decomposition, and (unless `--skip-placebo`, `--placebo-n N`
default 200) a label-shuffle backtest placebo; a network or cache failure is caught and recorded as
`summary.json["assets"]["skipped"]` without changing the engine's exit code. These diagnostics are appended
to `output/acceptance.csv` as `op="report"` rows (never thresholds): `pit_sharpe`, `oracle_sharpe`,
`insample_sharpe`, `label_lookahead`, `moment_lookahead`, `growth_share_6040`, `backtest_placebo_pct`.

Modules: `data` (masked FRED-MD), `factors` (masked PCA), `trend` (one-sided gaps), `regimes` (hysteresis
quadrants, constrained HMM, challengers), `pipeline` (the composition), `walkforward`, `placebo`,
`acceptance`, `figures`, `nber`, `assets` (Stage 5 universe/returns/regime moments), `portfolio` (Stage 6
mean-variance strategies and the achievable backtest).
