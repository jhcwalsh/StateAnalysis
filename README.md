# StateAnalysis — Macro Regime Engine

The site classifies the US macro economy into four regimes — Contraction, Goldilocks, Overheating,
Stagflation — from walk-forward filtered HMM probabilities published with a one-month lag, and shows what
that labeling is actually worth as a portfolio-timing signal. The honest answer: not much. The point-in-time
max-Sharpe strategy scores 0.75 (0.61 after 10 bp of trading costs) against a static 60/40 benchmark's 1.05
over 2010–2026. The headline in-sample number, 1.59, is mostly an artifact of look-ahead — 0.77 of the gap
comes from using full-sample moments, only 0.06 from knowing the labels themselves in advance — and two
placebo tests confirm it: shuffling the regime labels puts the real Sharpe spread at the 32nd percentile of
1000 run-preserving shuffles and the real backtest at the 25th percentile of 200 random relabelings, i.e.
below the median shuffle on both.
The regime timing signal is not there once look-ahead is removed. Every regime number on the site is shown
beside its benchmark so that comparison is unavoidable, not buried in a caption.

## Run locally

    .venv/Scripts/python.exe -m pip install -r requirements.txt
    cd regime_v2 && ../.venv/Scripts/python.exe run.py data/fredmd_2026-07.csv   # ~3 min; add --no-assets to skip the ETF stage
    .venv/Scripts/python.exe -m streamlit run app.py                            # from the repo root

## Refresh

The dashboard has a Refresh button at the bottom of the page. It reruns the engine for a given vintage — the
input defaults to the previous month — and republishes `regime_v2/output/`. If the run fails any acceptance
check, the last published outputs are kept and the dashboard keeps serving them.

## Layout

- `regime_v2/` — the regime engine. See `regime_v2/README.md` and `docs/SPEC.md` for the full design.
- `app.py` — the Streamlit dashboard, reading `regime_v2/output/` through `regime_v2/regime_v2/publish.py`.
- `pages/` — the two document pages, Introduction and Methodology. Their text lives in `docs/site/*.md`;
  `regime_v2/regime_v2/sitedocs.py` fills every `{{placeholder}}` from the published `summary.json` and
  every `![caption](fig:NAME)` from `regime_v2/figs/`, so the documents always agree with the dashboard.
  The rules are in `docs/site/CONTRACT.md`. Each page offers a self-contained HTML download that prints to PDF.
- `site_theme.py` — the lazyeconomist.com palette, CSS, masthead and page nav shared by every page.
- `tests/` — dashboard and page tests.

## Deployment

The site is deployed to the Mac Mini via Docker from `~/apps/states`, serving at
`states.lazyeconomist.com`; see `docs/SPEC.md` §12 for the full setup and checklist and the `deploy states`
command that ships a new build.

## Tests

    .venv/Scripts/python.exe -m pytest tests -q                        # dashboard, ~1 min
    cd regime_v2 && ../.venv/Scripts/python.exe -m pytest -q           # engine, ~4 min
