# Site documents — the shared contract

Two documents render as pages of the States app (`pages/1_Introduction.py`,
`pages/2_Methodology.py`) from Markdown sources in this directory:

- `docs/site/introduction.md` — a plain-language introduction (900–1300 words).
- `docs/site/methodology.md` — the methodology paper (3000–4500 words, with references).

Both are rendered by `regime_v2/regime_v2/sitedocs.py`, which (1) loads the published run
through `regime_v2.publish.load_published`, (2) builds the `numbers` dictionary below,
(3) substitutes placeholders, (4) resolves figure markers, and (5) can export the whole
document as one self-contained HTML file (images embedded) for download and printing.

## Placeholders

Write `{{key}}` anywhere in the Markdown. Unknown keys render as `[missing: key]` so they are
caught by tests. All values are pre-formatted strings.

| key | example | meaning |
|---|---|---|
| `run.vintage` | `fredmd_2026-08.csv` | the FRED-MD vintage file the run used |
| `run.vintage_month` | `2026-08` | the vintage's month (from the file name) |
| `run.asof` | `2026-07` | last labelled month, YYYY-MM |
| `run.asof_long` | `July 2026` | same, spelled out |
| `run.date` | `2026-09-05` | run date (UTC) |
| `run.label_source` | `walk-forward filtered` | label column the app shows |
| `current.regime` | `Overheating` | current regime name |
| `current.month_long` | `July 2026` | month of the current label |
| `current.prob_<Regime>` | `55%` | one per regime: `prob_Contraction`, `prob_Goldilocks`, `prob_Overheating`, `prob_Stagflation` |
| `current.growth_gap`, `current.inflation_gap` | `+0.05` | signed, 2 dp |
| `current.quadrant` | `Contraction` | deterministic quadrant rule's label |
| `sample.start` | `1969-01` | first month with a complete pair of standardised gaps (the FRED-MD panel itself begins 1959-01) |
| `sample.n_months` | `810` | months in the labelled sample |
| `sample.wf_start` | `1978-12` | first walk-forward month |
| `sample.wf_n` | `572` | walk-forward months |
| `params.window` | `240` | trend window, months |
| `params.theta` | `0.50` | hysteresis band |
| `params.persistence` | `10` | HMM Dirichlet diagonal prior κ |
| `params.eps` | `0.5` | HMM Dirichlet off-diagonal ε |
| `params.lag` | `1` | publication lag, months |
| `params.mask` | `2020-03 to 2020-12` | COVID estimation mask |
| `params.k_outlier` | `10` | outlier rule multiple of the IQR |
| `panel.n_growth`, `panel.n_inflation` | `22`, `13` | series per block |
| `hmm.expected_duration_<Regime>` | `9.3 months` | 1/(1−p_ii), one per regime |
| `hmm.mean_run_<Regime>` | `6.1 months` | mean realised run length, one per regime |
| `hmm.share_<Regime>` | `31%` | share of walk-forward months, one per regime |
| `hmm.transition_table` | markdown table | 4×4 transition matrix, 2 dp |
| `hmm.filtered_vs_smoothed` | `79%` | agreement of filtered and smoothed labels |
| `hmm.max_prob_share` | `41%` | share of months with max probability > 0.95 |
| `acc.n_tests`, `acc.n_passed` | `16`, `15` | acceptance rows with thresholds, and passes |
| `acc.known_failures` | `non_nber_contraction_hmm` | comma-separated, or `none` |
| `acc.table` | markdown table | name, value, op, threshold, status for every thresholded row |
| `acc.<name>` | `0.90` | the value of any acceptance row by name, e.g. `acc.gfc_contraction_hmm`, `acc.trunc_2015_agreement_hmm` |
| `nber.mean_lag`, `nber.median_lag` | `2.1`, `2` | months from NBER peak to first low-growth label (uncensored) |
| `nber.n_peaks`, `nber.n_censored` | `9`, `1` | |
| `nber.n_in_window` | `6` | peaks with a measured lag, i.e. inside the walk-forward window — the peaks `mean_lag`/`median_lag` are computed over. Peaks before the window opened are excluded, so `n_peaks` and `n_in_window` differ and both must be shown. |
| `assets.window_start`, `assets.window_end`, `assets.n_months` | `2007-08`, `2026-08`, `229` | |
| `assets.universe` | `SPY (Equity_US), …` | the 11 ETFs |
| `assets.n_<Regime>` | `36` | months per regime in the asset window |
| `assets.growth_share`, `assets.r2` | `51%`, `0.008` | 60/40 regression |
| `assets.spread_pct` | `32` | Sharpe-spread placebo percentile |
| `assets.spread_ord` | `32nd` | the same percentile as an English ordinal — write `{{assets.spread_ord}} percentile`, never `{{assets.spread_pct}}th` |
| `assets.spread_n` | `1000` | shuffles actually drawn (the length of the null array) |
| `bt.start`, `bt.min_obs` | `2010-01`, `15` | |
| `bt.perf0`, `bt.perf10` | markdown tables | strategy × ann_ret, ann_vol, sharpe, maxdd, turnover at 0 bp and 10 bp |
| `bt.sharpe_<Strategy>` | `0.77` | 0 bp Sharpe, one per strategy the engine runs (`regime_v2.portfolio.STRATEGIES + EXPOST`, which `sitedocs.STRATEGIES` is): `PIT_MaxSharpe`, `PIT_MinVar`, `ProbWeighted_MaxSharpe`, `Oracle_MaxSharpe`, `Static_6040`, `EqualWeight`, `PIT_LongOnly_MaxSharpe`, `PIT_RiskParity`, `Oracle_LongOnly_MaxSharpe`, `InSample_MaxSharpe_expost`, `InSample_LongOnly_expost` |
| `bt.sharpe10_<Strategy>` | `0.61` | the same at 10 bp |
| `bt.insample`, `bt.oracle`, `bt.pit`, `bt.moment_lookahead`, `bt.label_lookahead`, `bt.total_lookahead` | `1.51`, `0.70`, `0.77`, `+0.81`, `-0.07`, `+0.74` | look-ahead decomposition, unconstrained long-short family |
| `bt.lo_insample`, `bt.lo_oracle`, `bt.lo_pit`, `bt.lo_moment_lookahead`, `bt.lo_label_lookahead`, `bt.lo_total` | `0.98`, `0.71`, `0.68`, `+0.27`, `+0.03`, `+0.30` | the same decomposition for the long-only family (`InSample_LongOnly_expost`, `Oracle_LongOnly_MaxSharpe`, `PIT_LongOnly_MaxSharpe`); formatted like their unconstrained counterparts, and `n/a` for a run whose summary has no `lookahead_longonly` block |
| `bt.placebo_pct`, `bt.placebo_n` | `25`, `200` | backtest placebo |
| `bt.placebo_ord` | `25th` | `bt.placebo_pct` as an English ordinal — write `{{bt.placebo_ord}} percentile`, never `{{bt.placebo_pct}}th` |
| `bt.placebo_direction`, `assets.spread_direction` | `below` | `below` when the respective percentile (`bt.placebo_pct`, `assets.spread_pct`) is under 50, `above` otherwise |
| `bt.placebo_sentence` | `Both sit below the fiftieth percentile: more than half of the random relabelings beat the real one.` | one sentence stating the direction of both placebos, worded for either outcome (both below, both above, or split); when the backtest placebo is absent, states the Sharpe-spread placebo alone |
| `bt.counters` | `pit_maxsharpe_fallback 35, …` | fallback and guard counts |
| `skipped.assets` | `` | empty when the asset stage published; otherwise the reason (`asset stage not run` when the summary has no `assets` block at all). Text that depends on the asset stage must be wrapped in `<!-- if:assets -->…<!-- endif -->` so it is dropped when the stage was skipped. |

## Figures

Write `![caption](fig:NAME)` on its own line. `NAME` is one of the engine figures
(`fig1_factors_gaps`, `fig2_regime_timeline`, `fig3_state_space`, `fig4_hmm_probabilities`,
`fig5_revisions`, `fig6_classifier_comparison`, `fig7_walkforward`, `fig8_regime_returns`,
`fig9_mixture_6040`, `fig10_backtest_wealth`, `fig11_pit_weights`) or one of the documentation
figures the engine writes on every run (`regime_v2/regime_v2/docfigs.py`):

| name | shows |
|---|---|
| `doc_pipeline` | box-and-arrow schematic: FRED-MD vintage → t-code transforms and 10×IQR rule → growth and inflation blocks → one-factor PCA with EM → cumulated diffusion index → one-sided trend gap → constrained 4-state HMM (filtered) → labels with `available_at` → asset tables and backtest |
| `doc_quadrants` | the growth-gap × inflation-gap plane with the four named quadrants, the ±θ hysteresis band, and the current month marked |
| `doc_timing` | timeline of one month: data for month t, label t published on the first day of t+1, return of month t+1 paired with label t (descriptive tables), decision at end of d using label d−1 and earning d+1 (backtest) |
| `doc_lookahead` | waterfall: in-sample Sharpe → minus moment look-ahead → oracle → minus label look-ahead → PIT, with the Static_6040 Sharpe as a reference line |
| `doc_placebo` | two panels: histogram of the Sharpe-spread null (1000 shuffles) and of the backtest-placebo null (200 shuffles), each with the real value marked and its percentile |
| `doc_loadings` | horizontal bars of the growth and inflation factor loadings by series |
| `doc_transition` | 4×4 heatmap of the HMM transition matrix with expected durations |

A missing figure renders as a visible `[missing figure: NAME]` line, never an exception.

## Style

- Regime names exactly `Contraction`, `Goldilocks`, `Overheating`, `Stagflation`; the colours are
  the engine's (`regime_v2.regimes.COLORS`), never restated as hex in prose.
- Say what the app shows: the app's tabs are Factor gaps, Factor levels, Probabilities, State
  space, Regime returns, Correlations, Portfolios, Backtest, Acceptance, Figures. Refer to them
  by those names when pointing the reader to the app.
- The spec (`docs/SPEC.md`) is the authority on every method statement: D1–D11 and §6 Stages 1–6.
  The paper cites the spec's decisions by number where it helps (e.g. "D11").
- Honesty rule: every regime-timing number appears beside its benchmark; the placebo direction is
  stated; "in-sample" and "ex-post" are never presented as achievable.
- Author: The Lazy Economist (no personal byline). Date: `{{run.date}}`. Reproducibility note:
  `github.com/jhcwalsh/StateAnalysis`, engine `regime_v2`.
- Headings: `#` for the title only, `##` sections, `###` subsections. No HTML except the
  `<!-- if:assets -->` guard.
