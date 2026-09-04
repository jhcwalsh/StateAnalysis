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

Modules: `data` (masked FRED-MD), `factors` (masked PCA), `trend` (one-sided gaps), `regimes` (hysteresis
quadrants, constrained HMM, challengers), `pipeline` (the composition), `walkforward`, `placebo`,
`acceptance`, `figures`, `nber`.
