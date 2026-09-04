# SPEC — regime_v2: growth/inflation regime engine (rebuild)

This file is the source of truth for Claude Code sessions on this project. Read it fully before touching code; keep the "Decisions" and "Acceptance tests" sections in sync with the code. It lives at `docs/SPEC.md`.

Revised 2026-09-03 after a review of the spec and an empirical review of the prototype (see §10). The revision changes what "real-time" means in this project, tightens the HMM, and reinstates three decisions from the existing notebook.

## 1. Purpose

Rebuild the empirical pipeline behind the paper *Growth, Inflation, and Asset Prices: A Regime-Based Empirical Framework* so that:

1. regime labels are economically credible (the GFC is a Contraction, 2021–22 is high-inflation),
2. the paper's primary label for month *t* uses only information available at *t*. This is delivered by the walk-forward path (Stage 4), not by the full-sample path. The full-sample path is an ex-post comparator and is always labelled as such,
3. the probabilistic layer produces filtered probabilities that move, and a transition matrix with no hard zeros that can feed portfolio construction,
4. the whole thing re-runs from one command against a fresh FRED-MD vintage.

Non-goals for this build: forecasting regime transitions; ML classifiers; the asset/portfolio layer beyond the contract in §6 (Stages 5–6 land after the asset universe is confirmed); ALFRED vintage reconstruction.

## 2. Starting point

### 2.1 Prototype

A working prototype exists at `docs/regime_v2_prototype.zip` (Stages 0–3). Unpack it into the repo and treat it as the reference implementation, not sacred code. It ships no tests and no requirements file.

Its run on the FRED-MD 2026-07 vintage (690 months, 1969-01..2026-06) produced, on **smoothed** full-sample labels: GFC → Contraction 90%; 2021-06..2022-12 → high-inflation 100%; every NBER recession month low-growth; 7% Contraction false-alarm rate; expected durations 11/40/16/14 months.

The 2026-09-03 empirical review re-ran the prototype and measured the following. These numbers motivate the revisions in §4 and §8.

| Check | Result |
|---|---|
| Labels on overlap after truncating the sample at 2015-12 | quadrants 93.6% agree, HMM 92.9% |
| Same, truncated at 2007-12 | quadrants 86.5%, HMM 72.2% |
| Growth loadings, full vs truncated | corr 0.93–0.95 |
| Emission-only HMM (κ = 0) agreement with quadrants | 81% (34 of 120 Contraction months become Goldilocks) |
| Filtered vs smoothed label agreement | 84% |
| Filtered vs quadrant agreement | 66% (smoothed: 71%) |
| Share of months with max prob > 0.95 | filtered 45%, smoothed 61% |
| First filtered Contraction call in the GFC | 2008-10 (NBER peak 2007-12) |
| Growth series surviving the outlier rule in 2020-04 | 3 of 22 |
| Pooled growth variance, with / without 2020 | 0.67 / 0.44 |
| Goldilocks → Contraction transition probability | exactly 0 |
| 2024-01..2026-06 mean growth gap under 5 trend variants | −0.14 to −0.41 (never Goldilocks) |

Known prototype defects to fix in Stage 1–3 hardening: the transition matrix in `summary.json` is transposed (`to_dict()` default orient); the EM convergence test compares SVD scores whose sign can flip between iterations, so it never converges and always runs 50 iterations; the "1973-75 & 1979-82" test spans 1973-11..1982-11 as one block; free-HMM states are printed under quadrant names they do not deserve; loadings ride on `DataFrame.attrs`, which pandas drops on most operations; counter-cyclical sign flips are redundant given the anchor step.

```
regime_v2/
  regime_v2/
    __init__.py
    data.py        # Stage 1
    factors.py     # Stage 2a
    trend.py       # Stage 2b
    regimes.py     # Stage 3
    walkforward.py # Stage 4
    placebo.py     # Stage 4
  run.py           # end-to-end driver
  tests/
  data/            # FRED-MD vintages (gitignored except a pinned sample)
  figs/  output/   # generated
  README.md
  requirements.txt
```

Environment: Windows + PyCharm for development. Python ≥ 3.11. Deps: `pandas numpy scipy statsmodels scikit-learn hmmlearn matplotlib pytest`. `hmmlearn` is not in the repo's current `.venv`; add it to `requirements.txt`. Deploy path is push-to-GitHub then pull on the Mac Mini (§12), so nothing may depend on local absolute paths or on Windows; all paths relative to repo root or passed as CLI args. `run.py` must resolve `figs/` and `output/` relative to its own location, not the working directory.

### 2.2 Relationship to the existing repo

The repo already contains `Macro_Regime_Analysis.ipynb`, `regime_core.py`, `app.py` (Streamlit dashboard), `ui_io.py`, the `ui_data/` contract and `tests/`. That pipeline loads individual FRED series via `pandas_datareader` with a hand-built wage splice; there is no FRED-MD or ALFRED loader to reuse (this answers the old §9 Q1). FRED-MD sidesteps the quarterly-ULC and wage-splice traps that cost time in July.

Three decisions from the July sessions are carried into regime_v2 as D9–D11: the COVID estimation mask, causal hysteresis for the deterministic quadrants, and reporting data-driven challengers under their own names with marginalisation to quadrants.

**regime_v2 replaces the original pipeline** (decided 2026-09-03, §10). Migration, done as Stage 7 after Stage 4 is green:

- `Macro_Regime_Analysis.ipynb` is retired. Its point-in-time track, COVID mask, and hysteresis logic survive as D8–D10; its in-sample tables do not.
- The Streamlit dashboard (`app.py`, `ui_io.py`) is kept and re-pointed at `output/regime_labels.csv` and `output/summary.json`. The `ui_data/` parquet contract and the notebook-driven Refresh button are replaced by `run.py --vintage`.
- `regime_core.py` is folded into `regime_v2/regimes.py` (the hysteresis trigger moves verbatim); Q-numbered names, `PAPER_COLORS`, and `PAPER_ORDER` are deleted and the §3 names and colours are used everywhere.
- `tests/test_regime_core.py` moves with the trigger; `tests/test_ui_io.py` is rewritten against the new outputs.
- Generated outputs listed in `.gitignore` (`macro_regime_results.xlsx`, the three PNGs, `ui_data/`) are removed from the repo root.

Until Stage 7 lands, the Q-number ban in §11 applies to `regime_v2/` only; after it, repo-wide.

## 3. Vocabulary (use these exact names everywhere — code, figures, tables, paper)

| Regime | growth gap | inflation gap | Name in existing repo code |
|---|---|---|---|
| `Contraction` | < 0 | < 0 | `Q4_Recession` |
| `Goldilocks` | > 0 | < 0 | `Q1_Goldilocks` |
| `Overheating` | > 0 | > 0 | `Q2_Overheating` |
| `Stagflation` | < 0 | > 0 | `Q3_Stagflation` |

Never use Q1–Q4 in regime_v2. The paper currently has three incompatible Q-numberings; the code must not add a fourth. The existing repo's `PAPER_ORDER` names map one-to-one as shown; any bridge to the dashboard translates through this table.

Colours (fixed): Contraction `#3b5bdb`, Goldilocks `#2b8a3e`, Overheating `#f08c00`, Stagflation `#e03131`. These differ from `regime_core.PAPER_COLORS`; the dashboard adopts these if and when it consumes regime_v2 output.

Label types (use these words exactly): **smoothed** = full-sample forward-backward posterior (ex-post); **filtered** = forward-only posterior from a full-sample fit; **walk-forward** = filtered posterior from a model re-estimated on data ≤ *t*. Only the walk-forward label is real-time.

## 4. Decisions (settled — do not reopen without a note in §10)

D1. **Data:** FRED-MD monthly, current vintage, 1959→. McCracken–Ng t-codes. Outlier rule: `|x − median| > 10 × IQR → NaN` before factor extraction. In the walk-forward path the median and IQR are computed on data ≤ *t*. Unit labour costs excluded from the monthly factor (quarterly series).

D2. **Blocks:** growth = real activity (IP, employment, unemployment, claims, retail/consumption, income, housing; 22 series); inflation = CPI/PCE variants, PPI, avg hourly earnings (13 series). Lists live in `data.py`. Counter-cyclical series are not sign-flipped; PCA is invariant to column sign and the anchor in D3 fixes orientation.

D3. **Factors:** first PC per block, EM imputation of NaN cells, sign anchored to `INDPRO` / `CPIAUCSL`. The EM convergence test must be sign-invariant (compare the rank-1 reconstruction, or align the loading sign to the previous iteration before differencing). Loadings are returned explicitly, not via `.attrs`. Inflation factor is cumulated into a partial-sum diffusion index (an inflation-*rate* level). Growth factor is used as-is (already a growth *rate*). A month in which fewer than half of a block's series survive the outlier rule (2020-04 has 3 of 22) is flagged in `n_series` and excluded from estimation under D9.

D4. **Gaps are built on rates, not levels.** growth gap = 3-month MA of growth factor − trailing 120-month mean; inflation gap = 3-month MA of inflation diffusion index − trailing 120-month mean. Standardised by expanding-window std computed under the D9 mask. Rationale: a recursive Hamilton filter on the cumulated growth *level* labelled the whole 2010s as Contraction because trend growth slowed. Keep `hamilton_recursive` and `onesided_hp` in `trend.py` for the robustness table only. The trend window is chosen on the revision statistics (§6 Stage 2), never on whether the current label looks right.

D5. **No two-sided estimates in the labelling path.** HMM forward-backward smoothing is two-sided. Smoothed probabilities may appear only in comparison figures and must be captioned "smoothed (ex-post)". Any full-sample quantity lives in a function whose name ends in `_expost`.

D6. **Primary classifier = constrained 4-state HMM.** Emission means fixed and **symmetric**: `(±c_g, ±c_p)` where `c_g`, `c_p` are the mean absolute standardised gaps under the D9 mask, so the emission-only decision boundaries coincide with the axes and the HMM is genuinely a persistence smoother over the quadrant rule. One pooled within-quadrant covariance, computed under the D9 mask, also fixed. Only start and transition probabilities are estimated. Dirichlet prior on the transition matrix `1 + ε + κ·I` with ε = 0.5 off-diagonal pseudo-count and κ = 10 default, so no transition can be estimated as exactly zero. (hmmlearn's tied-covariance M-step divides by zero when means are frozen and clips `prior − 1 + counts` at zero — freezing covariances and adding ε are workarounds, not preferences.) Acceptance test: emission-only (κ = 0) agreement with quadrants ≥ 0.95.

D7. **Challengers, reported not used:** (a) deterministic quadrants with causal hysteresis (D10) — the transparent reference; (b) GMM on the same inputs; (c) free HMM (free means, full covariances). (b) and (c) find a crisis-vs-normal split rather than four sign quadrants; the paper reports this as a finding. They are reported under descriptive centroid names (e.g. `Normal`, `Crisis`, `HighInflation`), never under quadrant names, and are bridged to quadrants by marginalisation: `P(quadrant q | t) = Σ_k P(cluster k | t) · P(quadrant q | cluster k)`. No one-to-one cluster→quadrant map (it was non-injective in July and is non-injective in the prototype).

D8. **The paper's primary label is the walk-forward filtered label.** Recursive re-estimation on final-vintage data (Ince & Papell 2013), refit each month on data ≤ *t*, filtered probability stored for *t*. The full-sample smoothed label is the ex-post comparator for the revision figure. Full ALFRED vintage reconstruction is out of scope for v2. Stages 1–3 are full-sample and exist to fix the model form; no figure from them goes in the paper except as an ex-post comparison.

D9. **COVID estimation mask.** Months 2020-03..2020-12 are excluded from every *estimation* step (outlier thresholds, block standardisation, PCA loadings, expanding std, emission centroids and covariance, HMM transition fit) and are still transformed, classified and reported. Carried over from the notebook, where the same fix un-collapsed the regimes. In the prototype, 2020 inflates the pooled growth variance by 50% and produces a −5.6 / +4.3 SD whipsaw in the growth gap.

D10. **Deterministic quadrants use causal hysteresis.** The Schmitt trigger from `regime_core.hysteretic_sign` (state flips only when the gap crosses ±θ) is the persistence device for the quadrant challenger, with θ reported alongside. A memoryless sign rule is kept only as the θ = 0 case. No post-hoc run merging or minimum-duration rules anywhere.

D11. **Publication lag.** A month-*t* label is not available at *t*. `regime_labels.csv` carries an `available_at` column (first day of the month in which the FRED-MD vintage containing month *t* is published; default *t* + 1 month). The asset layer joins on `available_at`, never on `date`.

## 5. Module contracts

Keep these signatures stable; other modules and the paper's figure scripts depend on them.

```python
# data.py
def build_blocks(path: str, k_outlier: float = 10.0, asof: str | None = None,
                 mask: tuple[str, str] | None = COVID_MASK) -> dict
# -> {"growth": DataFrame, "inflation": DataFrame, "outliers": DataFrame[date, series],
#     "missing_series": list[str], "tcodes": Series, "estimation_mask": Series[bool]}
# asof truncates the raw panel at that month before any statistic is computed.

# factors.py
def pca_factor_em(block: DataFrame, anchor: str, est_mask: Series, n_iter=50, tol=1e-6) -> tuple[DataFrame, Series]
# -> (DataFrame[factor, diffusion, n_series], loadings)
def pca_factor_expanding(block, anchor, est_mask, min_obs=120) -> tuple[DataFrame, DataFrame]   # Stage 4
# -> (same shape, loadings per month)

# trend.py
def make_gap(y: Series, method: str, est_mask: Series, **kw) -> DataFrame   # columns level, trend, gap_raw, gap
# methods: "smoothed_trailing" (default), "trailing_mean", "trailing_median", "hamilton", "onesided_hp"
def revision_stats(first: Series, final: Series) -> dict  # corr, noise_to_signal_rmse, sign_agreement, n

# regimes.py
REGIMES = ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
def quadrant_labels(g: Series, p: Series, theta: float = 0.0) -> Series          # hysteresis when theta > 0
def fit_hmm4(g, p, est_mask, persistence=10.0, eps=0.5, seed=0, constrained=True) -> HMMResult
# HMMResult: labels_filtered, labels_smoothed_expost, probs_filtered, probs_smoothed_expost, model, state_map, means
def fit_gmm4(g, p, est_mask, seed=0) -> (labels, probs, model, cluster_names, quadrant_profile)   # challenger
def transition_table(model, state_map) -> DataFrame    # rows = from, cols = to; write with orient="index"
def expected_duration(tm) -> Series
def run_lengths(labels) -> Series

# walkforward.py (Stage 4)
def fit_hmm4_walkforward(path, min_obs=240, **kw) -> (labels_rt, probs_rt, transmat_by_month)
# re-runs build_blocks(asof=t) -> factors -> gaps -> fit_hmm4 for each t; stores the filtered prob for t.
```

Outputs written by `run.py`: `output/regime_labels.csv` (index date; available_at, growth_gap, inflation_gap, quadrant, quadrant_theta, hmm_walkforward, hmm_filtered, hmm_smoothed_expost, hmm_free, gmm, p_<regime>×4 from the walk-forward), `output/outliers_removed.csv`, `output/summary.json`, `figs/fig1..fig7.png`. The transition matrix in `summary.json` is written row = from-state.

## 6. Stages and tasks

Work in this order. Each stage ends with `pytest` green and the acceptance tests in §8 re-run.

**Stage 1 — data (prototype done; harden).**
- [ ] Add `requirements.txt` and `tests/` to the package; nothing in the prototype is under test.
- [ ] Add `asof` truncation and the D9 estimation mask to `build_blocks`; outlier median/IQR computed under the mask and on data ≤ asof.
- [ ] Remove the counter-cyclical sign flip (D2).
- [ ] Add `data/README.md` data sheet generator: every series, t-code, block, number of outlier cells removed, first/last date. Called from `run.py`.
- [ ] Pin one vintage in-repo (`data/fredmd_2026-07.csv`) for tests; download others via `--vintage YYYY-MM`. Verify the FRED-MD download URL against the St. Louis Fed site before baking it in; the pattern in the first draft of this spec was unverified.
- [ ] Test: outlier count in 2020 ≥ 20; no series in both blocks; all listed series present; last vintage month fully populated for both blocks (it is in 2026-07: 20/22 and 13/13).

**Stage 2 — factors and gaps (prototype done; harden).**
- [ ] Sign-invariant EM convergence check; return loadings explicitly (D3).
- [ ] Test: growth factor correlation with `INDPRO` transformed > 0.7; inflation factor with `CPIAUCSL` > 0.7.
- [ ] Test: sign anchor stable under truncation at 2007-12 and 2015-12 (factor correlation > 0.99 on overlap).
- [ ] Test (trend step only): `make_gap` on a fixed factor series truncated at 2015-12 equals the full-sample result on the overlap to 1e-10. This is the only exact real-time test; the end-to-end test is the tolerance test in Stage 4.
- [ ] Add `--trend-window` CLI arg (default 120) and `trailing_median`; produce the robustness table: labels under windows 120/180/240, mean vs median, smoothed_trailing vs hamilton. Report label agreement matrix, revision statistics per variant, and the GFC Contraction share per variant (measured: 0.90 at 120-mean, 0.80 for every other variant).
- [ ] Choose the trend window on noise-to-signal and sign agreement in `revision_stats`; record the choice in §10. The 2024–26 low-growth reading is a finding, not a bug to tune away: under all five variants the mean growth gap over 2024–26 is negative and Goldilocks appears in at most one month.

**Stage 3 — regime models (prototype done; extend).**
- [ ] Symmetric fixed means, masked pooled covariance, ε off-diagonal prior (D6). Return both filtered and smoothed probabilities; the label column defaults to filtered.
- [ ] Fix the transposed transition matrix in `summary.json` (`orient="index"`).
- [ ] Hysteresis in `quadrant_labels` (D10), θ from `--theta`, default taken from the notebook's `HYSTERESIS_THETA`.
- [ ] Implement `fit_gmm4` (sklearn `GaussianMixture`, 4 components, full covariance) reported under descriptive names with a `quadrant_profile` = empirical P(quadrant | cluster) and marginalised quadrant probabilities (D7). Same treatment for the free HMM.
- [ ] Split the "1973-75 & 1979-82" diagnostic into two windows (1973-11..1975-03, 1980-01..1982-11) or drop it.
- [ ] Add fig 6: 2×2 panel of state-space scatters coloured by quadrants / constrained HMM / free HMM / GMM, same axes, challengers under their own names.
- [ ] Add `summary.json` fields: per-classifier agreement with quadrants, emission-only agreement, filtered-vs-smoothed agreement, regime counts, share of months with max prob > 0.95 (filtered and smoothed separately), expected durations, transition matrix (row = from), min off-diagonal transition probability.
- [ ] Test: constrained HMM means equal the symmetric targets after fit (they must not have moved). Test: no transition probability below 1e-3.

**Stage 4 — walk-forward and validation harness (new; produces the paper's primary label).**
- [ ] `pca_factor_expanding`: re-estimate loadings each month on data ≤ *t* under the mask, sign-anchored (drop the separate previous-month alignment; the anchor is sufficient and unambiguous).
- [ ] `fit_hmm4_walkforward`: from `min_obs=240`, rerun the whole pipeline with `asof=t` and store the *filtered* probability for month *t* and the transition matrix at *t*. This is the real-time label and the `p_<regime>` columns.
- [ ] End-to-end real-time tolerance test: labels from `asof=2015-12` vs full-sample on the overlap ≥ 0.90 agreement (measured 0.93 for quadrants, 0.93 for HMM in the prototype); `asof=2007-12` ≥ 0.80 (measured 0.87 / 0.72; the HMM fails today and the symmetric means and mask are expected to help — if not, this is a paper finding).
- [ ] Revision figure (fig 7): walk-forward label vs full-sample smoothed label over time; month-level agreement and lag (months) at each NBER peak and trough. Report the 2008-10 GFC call explicitly.
- [ ] Placebo: `placebo.py` — shuffle regime labels within-sample 1,000 times preserving run-length distribution (block shuffle), recompute any downstream statistic, return null distribution and the real statistic's percentile. Generic: takes a callable.
- [ ] Block bootstrap (12-month blocks) helper for CIs on regime-conditional statistics.
- [ ] Re-run the whole §8 table on walk-forward labels; those are the numbers the paper quotes.

**Stage 5–6 — asset layer (blocked on §9 answers).** Contract only for now:
```python
# assets.py
def load_returns(source: str, tickers: list[str], start: str) -> DataFrame   # monthly total returns
def regime_conditional_table(returns, labels, bootstrap) -> DataFrame       # n, ann_ret, ann_vol, sharpe, maxdd, hit, SEs
def conditional_corr(returns, labels) -> dict[str, DataFrame]
def mixture_moments(mu_by_regime, cov_by_regime, probs_t) -> (mu_t, Sigma_t)  # law of total variance
def growth_share_6040(returns_6040, growth_gap, inflation_gap) -> float       # replaces the unsourced "95%+"
```
All joins to labels use `available_at` (D11). The ETF universe has common history only from 2007-07 (see the July notes), so regime-conditional moments will be estimated on one recession and one inflation spike while the transition matrix is estimated from 1969; the table must report *n* per regime in the asset window and the bootstrap CIs.

**Stage 7 — replace the original pipeline (after Stage 4 is green; see §2.2).**
- [ ] Move `hysteretic_sign` into `regime_v2/regimes.py` with its test; delete `regime_core.py`.
- [ ] Re-point `app.py` / `ui_io.py` at `output/regime_labels.csv` and `output/summary.json`; remove the notebook Refresh path and the `ui_data/` contract; rewrite `tests/test_ui_io.py`.
- [ ] Retire `Macro_Regime_Analysis.ipynb` and the root-level generated files; update `README.md` and `.gitignore`.
- [ ] Repo-wide grep for `Q1|Q2|Q3|Q4` returns nothing; `pytest -q` green; `python run.py data/fredmd_2026-07.csv` rebuilds everything the dashboard needs.
- [ ] Deployment checklist in §12 completed on the Mini.

## 7. Figures (the paper pulls these directly; keep filenames)

- `fig1_factors_gaps.png` — level, recursive trend, real-time gap; growth and inflation panels; NBER shading.
- `fig2_regime_timeline.png` — four strips: hysteresis quadrants, walk-forward HMM (primary), free HMM, GMM. Challenger strips use their own names and colours, not the regime palette.
- `fig3_state_space.png` — scatter coloured by walk-forward HMM labels, n per regime in legend.
- `fig4_hmm_probabilities.png` — stacked **filtered** probabilities (walk-forward). A smoothed version, if shown, is a separate panel captioned "smoothed (ex-post)".
- `fig5_revisions.png` — real-time vs ex-post growth gap with corr / noise-to-signal / sign agreement in title.
- `fig6_classifier_comparison.png` — 2×2 state-space panel (Stage 3).
- `fig7_walkforward.png` — walk-forward vs full-sample smoothed labels (Stage 4), NBER lags annotated.

All figures: 130 dpi, `matplotlib.use("Agg")`, NBER recessions shaded from the `NBER` list in `run.py`, regime colours from `regimes.COLORS`.

## 8. Acceptance tests (must pass before any figure goes into the paper)

Implemented in `run.py` → `summary.json["acceptance_tests"]`; mirror them in `tests/test_acceptance.py`. Thresholds are stated with a rationale so they are not re-fitted to whatever the next run produces. Every history test is evaluated on the **walk-forward filtered** label; the smoothed values are recorded alongside for the revision discussion.

| Test | Threshold | Rationale |
|---|---|---|
| GFC 2008-09..2009-06 labelled Contraction | ≥ 0.8 | Deepest post-war contraction; allows a 2-month lag in a 10-month window |
| GFC 2008-09..2009-06 labelled Contraction (hysteresis quadrants) | ≥ 0.7 | Same, one extra month of slack for a memoryless-ish rule |
| COVID 2020-03..2020-06 labelled Contraction | ≥ 0.5 | 4-month window; May–June rebound is ambiguous by construction |
| 2021-06..2022-12 labelled Overheating or Stagflation | ≥ 0.9 | Inflation was above every trend definition for the whole window |
| NBER recession months labelled Contraction or Stagflation | ≥ 0.9 | Low growth must dominate recessions |
| Non-NBER months labelled Contraction (false alarm) | ≤ 0.10 | Comparable to the false-positive rate of standard recession indicators |
| Share of months with max filtered prob > 0.95 | ≤ 0.75 | Probabilities must move; measured 0.45 filtered, 0.61 smoothed |
| Emission-only (κ = 0) agreement with quadrants | ≥ 0.95 | D6: the HMM must be persistence over quadrants, not a relabelling (prototype: 0.81) |
| Constrained-HMM means unchanged after fit | exact | D6 |
| Minimum transition probability | ≥ 1e-3 | No impossible transitions for the portfolio layer (prototype: exact 0) |
| Trend-step real-time property (Stage 2) | exact | Only the trend step can be exact |
| End-to-end label agreement, `asof=2015-12` vs full | ≥ 0.90 | Measured 0.93 in the prototype |
| End-to-end label agreement, `asof=2007-12` vs full | ≥ 0.80 | Measured 0.87 quadrants / 0.72 HMM; a known failure until D6/D9 land |
| Filtered vs smoothed label agreement | reported, no threshold | Measured 0.84; goes in the paper as the cost of real-time labelling |
| Seed invariance | exact | Same labels for seeds 0, 1, 2 |

Prototype values (smoothed, full-sample, pre-revision): 0.90, 0.90, 0.75, 1.00, 1.00, 0.07, 0.61. Filtered values from the same fit: GFC 0.90, high-inflation 1.00, share > 0.95 0.45.

## 9. Open questions

1. ~~Existing data loaders~~ — answered: none reusable (§2.2).
2. The nine-asset validation universe: tickers/indices and return source (S&P Global MCP, yfinance, other), and whether returns are total or price. Note the 2007-07 common-history constraint (§6).
3. Should the three-axis taxonomy (growth/inflation/policy + liquidity modifier) consume these labels, or stay separate? Affects whether `regime_labels.csv` needs a `policy` column stub.
4. Trend window / estimator to adopt (Stage 2 robustness table will inform this; decided on revision statistics, not on the 2024–26 label).
5. ~~Replace or run alongside the notebook~~ — answered: replace (§2.2, Stage 7).
6. Default θ for hysteresis quadrants: reuse the notebook's `HYSTERESIS_THETA` or re-tune on the FRED-MD gaps?
7. Hosting on lazyeconomist.com (§12): is the dashboard served live (Streamlit behind a reverse proxy, at which path or subdomain) or published as static HTML and figures on a schedule? Which process supervisor runs it on the Mini, and what triggers the monthly refresh when a new FRED-MD vintage is out?

## 10. Decision log

- 2026-09-03 — Rate-based gaps adopted over level-based Hamilton filter (2010s drift artefact). See D4.
- 2026-09-03 — Constrained HMM adopted as primary; free HMM and GMM demoted to challengers after they recovered crisis-vs-normal states rather than sign quadrants. See D6–D7.
- 2026-09-03 — Covariances frozen in constrained HMM because of hmmlearn tied-covariance bug with frozen means. See D6.
- 2026-09-03 (revision) — Walk-forward filtered label made the paper's primary label; Stages 1–3 demoted to model-form fixing and ex-post comparison. Reason: prototype's smoothed labels are two-sided, and end-to-end truncation flips 7–28% of labels. See D5, D8.
- 2026-09-03 (revision) — HMM emission means made symmetric about the axes and an ε off-diagonal transition prior added. Reason: prototype's centroid means relabel 19% of months before any persistence, and the fitted Goldilocks→Contraction probability was exactly zero. See D6.
- 2026-09-03 (revision) — COVID estimation mask, causal hysteresis, and challenger marginalisation reinstated from the July notebook work as D9–D11. Reason: the first draft silently dropped them; the prototype shows 2020 inflating the pooled growth variance by 50%.
- 2026-09-03 (revision) — "Exact" end-to-end real-time test replaced by tolerance tests; exactness kept for the trend step only. Reason: re-estimated loadings make exactness impossible.
- 2026-09-03 (revision) — 2024–26 reading reclassified from "trend-window bug" to "low-growth finding". Reason: negative mean growth gap under all five trend variants.
- 2026-09-03 — regime_v2 replaces the notebook pipeline rather than running alongside it; migration is Stage 7. The engine and dashboard become part of lazyeconomist.com, hosted on the Mac Mini (§12).
- (add entries here; never edit D1–D11 silently)

## 11. Conventions for Claude Code sessions

- Run `python run.py data/fredmd_2026-07.csv && pytest -q` after every change to `regime_v2/`.
- Do not introduce look-ahead in the labelling path. If you need a full-sample quantity, it goes in a function whose name ends in `_expost` and it may only be used for comparison figures. HMM smoothing counts as look-ahead.
- Every estimation step takes the estimation mask; grep for `.mean()`, `.std()`, `.median()`, `.quantile(` in `regime_v2/` and check each is masked or documented as ex-post.
- Named regimes only inside `regime_v2/`. Grep `regime_v2/` for `Q1|Q2|Q3|Q4` before committing; it should return nothing. The existing `regime_core.py` / `app.py` names are translated through the §3 table until Stage 7 removes them; after Stage 7 the grep runs repo-wide.
- Prefer editing the existing modules over adding parallel ones. New estimators go behind the `method=` switch in `make_gap` or as a new `fit_*` in `regimes.py` with the same return type.
- Nothing Windows-specific: no backslash paths, no `.exe` references outside `README.md` setup notes, no reliance on the case-insensitive filesystem. It has to run unchanged on the Mini.
- Commit messages reference the stage and task, e.g. `stage4: walk-forward HMM refit`.

## 12. Deployment: lazyeconomist.com on the Mac Mini

The regime engine and its dashboard are a component of **lazyeconomist.com**, hosted on the Mac Mini. What is settled:

- **Flow:** develop on Windows, push to `github.com/jhcwalsh/StateAnalysis`, pull on the Mini. The Mini never runs anything that is not in git.
- **Refresh:** `python run.py --vintage YYYY-MM` downloads the vintage, rebuilds `output/` and `figs/`, and re-runs the acceptance tests. A refresh that fails any §8 test must not replace the previously published outputs; it exits non-zero and leaves the last good `output/` in place.
- **Published surface:** the dashboard (`app.py`) reading `output/`, plus the seven figures and `regime_labels.csv` as downloadable files. Nothing on the site is produced by the notebook.
- **Secrets:** none required. FRED-MD is a public CSV; `FRED_API_KEY` is no longer needed once the notebook is retired.
- **Pinned vintage:** `data/fredmd_2026-07.csv` stays in git so the site can rebuild without network access and the tests are reproducible on the Mini.

Serving mechanics (reverse proxy, path or subdomain, process supervisor, refresh scheduling) are §9 Q7 and are recorded here once decided. Do not add a `Dockerfile`, `launchd` plist, or proxy config until then.
