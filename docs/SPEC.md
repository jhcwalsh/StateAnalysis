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
    assets.py      # Stage 5
    portfolio.py   # Stage 6
  run.py           # end-to-end driver
  tests/
  data/            # FRED-MD vintages (gitignored except a pinned sample)
  figs/  output/   # generated
  README.md
  requirements.txt
```

Environment: Windows + PyCharm for development. Python ≥ 3.11. Deps: `pandas numpy scipy statsmodels scikit-learn hmmlearn matplotlib pytest`. `hmmlearn` is not in the repo's current `.venv`; add it to `requirements.txt`. Deploy path is push-to-GitHub then pull on the Mac Mini (§12), so nothing may depend on local absolute paths or on Windows; all paths relative to repo root or passed as CLI args. `run.py` must resolve `figs/` and `output/` relative to its own location, not the working directory. A full run with the walk-forward takes about 90 seconds on the pinned vintage.

### 2.2 Relationship to the existing repo

The repo contained `Macro_Regime_Analysis.ipynb`, `regime_core.py`, `app.py` (Streamlit dashboard), `ui_io.py`, the `ui_data/` contract and `tests/`. That pipeline loads individual FRED series via `pandas_datareader` with a hand-built wage splice; there is no FRED-MD or ALFRED loader to reuse (this answers the old §9 Q1). FRED-MD sidesteps the quarterly-ULC and wage-splice traps that cost time in July.

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

D3. **Factors:** first PC per block, EM imputation of NaN cells, sign anchored to `INDPRO` / `CPIAUCSL`. The EM convergence test must be sign-invariant (compare the rank-1 reconstruction, or align the loading sign to the previous iteration before differencing). Loadings are returned explicitly, not via `.attrs`. Inflation factor is cumulated into a partial-sum diffusion index (an inflation-*rate* level). Growth factor is used as-is (already a growth *rate*). A month in which fewer than half of a block's series survive the outlier rule (2020-04 has 3 of 22) is flagged in `n_series` and excluded from estimation under D9. The factor is scaled by the estimation-row std but **not demeaned**; a drift term Σ_j l_j·μ_j/sd_j is added so that `cumsum(factor)` is the loading-weighted cumulated raw series up to a constant. Demeaning on the estimation sample turned the sample mean into a drift in the diffusion index and a 0.3 SD sample-dependent offset in the inflation gap (2026-09-04). EM convergence is judged on the sign-aligned loadings; every row is scored by the regression of its observed cells on the converged loadings.

D4. **Gaps are built on rates, not levels.** growth gap = 3-month MA of growth factor − trailing 240-month mean; inflation gap = 3-month MA of inflation diffusion index − trailing 240-month mean (240 adopted 2026-09-04 on the Stage 2 revision statistics; 120 was the prototype default). Standardised by expanding-window std computed under the D9 mask. Rationale: a recursive Hamilton filter on the cumulated growth *level* labelled the whole 2010s as Contraction because trend growth slowed. Keep `hamilton_recursive` and `onesided_hp` in `trend.py` for the robustness table only. The trend window is chosen on the revision statistics (§6 Stage 2), never on whether the current label looks right.

D5. **No two-sided estimates in the labelling path.** HMM forward-backward smoothing is two-sided. Smoothed probabilities may appear only in comparison figures and must be captioned "smoothed (ex-post)". Any full-sample quantity lives in a function whose name ends in `_expost`.

D6. **Primary classifier = constrained 4-state HMM.** Emission means fixed and **symmetric**: `(±c_g, ±c_p)` where `c_g`, `c_p` are the mean absolute standardised gaps under the D9 mask, so the emission-only decision boundaries coincide with the axes and the HMM is genuinely a persistence smoother over the quadrant rule. One pooled within-quadrant covariance, computed under the D9 mask, also fixed. Only start and transition probabilities are estimated. Dirichlet prior on the transition matrix `1 + ε + κ·I` with ε = 0.5 off-diagonal pseudo-count and κ = 10 default, so no transition can be estimated as exactly zero. (hmmlearn's tied-covariance M-step divides by zero when means are frozen and clips `prior − 1 + counts` at zero — freezing covariances and adding ε are workarounds, not preferences.) Acceptance test: emission-only (κ = 0) agreement with quadrants ≥ 0.95. The pooled covariance is stored **diagonal**; with symmetric means this makes the emission-only boundaries exactly the axes (measured emission-only agreement 1.00). The prototype's off-diagonal term was 0.002.

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
def make_gap(y: Series, method: str, est_mask: Series, min_obs_std: int = 60, **kw) -> DataFrame   # columns level, trend, gap_raw, gap
# methods: "smoothed_trailing" (default), "trailing_mean", "trailing_median", "hamilton", "onesided_hp"
def revision_stats(first: Series, final: Series) -> dict  # corr, noise_to_signal_rmse, sign_agreement, n

# regimes.py
REGIMES = ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
def quadrant_labels(g: Series, p: Series, theta: float = 0.0) -> Series          # hysteresis when theta > 0
def fit_hmm4(g, p, est_mask, persistence=10.0, eps=0.5, seed=0, constrained=True) -> HMMResult
# HMMResult: labels_filtered, labels_smoothed_expost, probs_filtered, probs_smoothed_expost, model, state_map, means
def fit_free_hmm4(g, p, est_mask, persistence=10.0, eps=0.5, seed=0) -> HMMResult   # challenger, free means/covariances
def describe_state(mean: ndarray, k: int) -> str   # descriptive centroid name, not a quadrant name
def fit_gmm4(g, p, est_mask, seed=0) -> GMMResult   # challenger
# GMMResult: labels, probs, model, cluster_names, quadrant_profile, quadrant_probs
def quadrant_profile(cluster_labels, quad_labels, clusters=None) -> DataFrame   # empirical P(quadrant | cluster)
def marginalise(cluster_probs: DataFrame, profile: DataFrame) -> DataFrame   # P(quadrant | t) = Σ_k P(cluster k | t)·P(quadrant | cluster k)
def transition_table(transmat: ndarray, state_names: dict[int, str]) -> DataFrame    # rows = from, cols = to; write with orient="index"
def expected_duration(tm) -> Series
def run_lengths(labels) -> Series

# walkforward.py (Stage 4)
def fit_hmm4_walkforward(path, min_obs=240, step=1, start=None, end=None, progress=None, **kw) -> WalkForwardResult
# WalkForwardResult: labels_rt, probs_rt, growth_gap_rt, inflation_gap_rt, transmat_by_month
# re-runs build_blocks(asof=t) -> factors -> gaps -> fit_hmm4 for each t; stores the filtered prob for t.

# acceptance.py
KNOWN_FAILURES: dict[str, str]   # test name -> reason; declared failures reported but non-blocking
def blocking_failures(table: DataFrame) -> list[str]
def all_passed(table: DataFrame) -> bool

# figures.py
def nber_lags(labels_rt: Series) -> DataFrame   # columns: peak, first_low_growth_rt, lag_months, censored

# assets.py (Stage 5)
UNIVERSE: dict[str, str]   # ticker -> display name; the 11 ETFs (SPY, VEA, EEM, AGG, TLT, LQD, HYG, VNQ, GLD, DBC, TIP)
def load_returns(source="yfinance", tickers=None, start="2000-01-01", cache=None, refresh=False, fetch=None) -> DataFrame
# monthly simple total returns from adjusted closes; index = month start; columns = display names
# fetch: injectable downloader (tickers, start) -> daily price frame; the tests never hit the network
def align_to_available(returns: DataFrame, labels: DataFrame, col: str, strict: bool = False) -> DataFrame
# joins the return of month r to the label row whose available_at <= r (strict=True: < r) (D11);
# returns [label, label_date, *assets] — label_date is the month the label describes
def regime_conditional_table(returns, labels, col="hmm_walkforward", n_boot=1000, block=12, seed=0) -> DataFrame
# index (asset, regime); columns n, ann_ret, ann_vol, sharpe, maxdd, hit, se_ann_ret, se_sharpe
def conditional_corr(returns, labels, col="hmm_walkforward") -> dict[str, DataFrame]
def moments_by_label(aligned: DataFrame, min_obs: int = 1) -> (dict[str, Series], dict[str, DataFrame])
# annualised mu, cov per label from an already-aligned panel (label/label_date + asset columns);
# the one implementation, shared by regime_moments and portfolio.backtest
def regime_moments(returns, labels, col, strict=False, min_obs=1) -> (dict[str, Series], dict[str, DataFrame])
def mixture_moments(mu_by_regime, cov_by_regime, probs_t: Series) -> (Series, DataFrame)  # law of total variance
def mixture_path(mu_by_regime, cov_by_regime, probs: DataFrame, weights: Series) -> DataFrame  # columns mu, sigma per month
def growth_share_6040(aligned: DataFrame) -> dict
# aligned columns r6040, growth_gap, inflation_gap (already joined/aligned by the caller);
# OLS of the 60/40 return on both gaps; returns r2, growth_share (LMG/Shapley split of R2), inflation_share, n
def sharpe_spread_placebo(aligned: DataFrame, n=1000, seed=0) -> dict
# aligned columns label, r6040; placebo() on max-min regime Sharpe (as implemented; the two Series-argument
# signatures originally specified were replaced by this single aligned-frame form, controller decision 2026-09-04)

# portfolio.py (Stage 6)
def mv_weights(mu: Series, Sigma: DataFrame, objective="max_sharpe", rf=0.0, leverage_cap=3.0) -> (Series, dict)
# unconstrained long-short, pinv; returns (weights, flags{negsum, rank_deficient})
STRATEGIES = ["PIT_MaxSharpe", "PIT_MinVar", "ProbWeighted_MaxSharpe", "Oracle_MaxSharpe", "Static_6040", "EqualWeight"]
EXPOST = ["InSample_MaxSharpe_expost"]   # full-sample moments + smoothed labels; measures look-ahead
def backtest(returns, labels_frame, probs_rt, start="2010-01-01", min_regime_obs=15, cost_bp=0.0,
             leverage_cap=3.0, strategies=None, include_expost=True) -> BacktestResult
# strategies=None means STRATEGIES; include_expost appends EXPOST to whatever was asked for.
# BacktestResult: returns (DataFrame, one column per strategy), weights (dict[str, DataFrame]),
#                 turnover (DataFrame), perf (DataFrame: ann_ret, ann_vol, sharpe, maxdd, turnover),
#                 counters (dict: pit_maxsharpe_fallback, pit_minvar_fallback, oracle_fallback, pw_fallback,
#                 insample_fallback, negsum, rank_deficient), params (dict: start, min_regime_obs, cost_bp, leverage_cap)
# There is no separate insample_sharpe(): the in-sample number is the InSample_MaxSharpe_expost row of perf.
def lookahead_decomposition(perf: DataFrame) -> dict
# reads the InSample_MaxSharpe_expost / Oracle_MaxSharpe / PIT_MaxSharpe rows of one perf table;
# returns insample_sharpe, oracle_sharpe, pit_sharpe, moment_lookahead, label_lookahead, total
def backtest_placebo(returns, labels_frame, probs_rt, n=200, seed=0, **kw) -> dict   # PIT_MaxSharpe Sharpe vs shuffled labels
```

Outputs written by `run.py`: `output/regime_labels.csv` (index date; columns as built: `available_at, growth_gap, inflation_gap, quadrant, quadrant_theta, hmm_filtered, hmm_smoothed_expost, hmm_walkforward, quadrant_walkforward, hmm_free, gmm, p_Contraction, p_Goldilocks, p_Overheating, p_Stagflation`), `output/outliers_removed.csv`, `output/summary.json`, `output/acceptance.csv`, `figs/fig1..fig7.png`; with the asset stage: `output/regime_returns.csv`, `output/regime_corr_<Regime>.csv`, `output/backtest_returns.csv`, `output/portfolio_weights.csv`, `figs/fig8..fig11.png`, and `summary.json["assets"]`. The transition matrix in `summary.json` is written row = from-state. `quadrant` is the full-sample hysteresis label; `quadrant_walkforward` applies the same rule to the walk-forward gaps and is what §8 evaluates.

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
- [x] Add `--trend-window` CLI arg (default 240 since 2026-09-04) and `trailing_median`; produce the robustness table: labels under windows 120/180/240, mean vs median, smoothed_trailing vs hamilton. Report label agreement matrix, revision statistics per variant, and the GFC Contraction share per variant (measured: 0.90 at 120-mean, 0.80 for every other variant).
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
- [ ] End-to-end real-time tolerance test: labels from `asof=2015-12` vs full-sample on the overlap ≥ 0.90 agreement (measured 0.93 for quadrants, 0.93 for HMM in the prototype); `asof=2007-12` ≥ 0.80 (prototype measured 0.87 / 0.72; after the D3 drift fix the HMM passes at 0.953 and the quadrants at 0.955).
- [ ] Revision figure (fig 7): walk-forward label vs full-sample smoothed label over time; month-level agreement and lag (months) at each NBER peak and trough. Report the 2008-10 GFC call explicitly.
- [ ] Placebo: `placebo.py` — shuffle regime labels within-sample 1,000 times preserving run-length distribution (block shuffle), recompute any downstream statistic, return null distribution and the real statistic's percentile. Generic: takes a callable.
- [ ] Block bootstrap (12-month blocks) helper for CIs on regime-conditional statistics.
- [ ] Re-run the whole §8 table on walk-forward labels; those are the numbers the paper quotes.

**Stage 5 — asset layer (decided 2026-09-04; §9 Q2 answered).**
- [ ] `assets.UNIVERSE`: the notebook's 11 ETFs and display names (SPY Equity_US, VEA Equity_DevelopedExUS, EEM Equity_EM, AGG US_Aggregate_Bonds, TLT US_Long_Treasury, LQD Corp_IG, HYG Corp_HY, VNQ REITs, GLD Gold, DBC Commodities, TIP TIPS). Common history from 2007-07 (VEA binds); DBC not PDBC (July note).
- [ ] `load_returns`: yfinance adjusted close (`auto_adjust=True`), month-end resample, simple returns, index shifted to month start. Cached to `data/returns_yfinance.parquet` (gitignored); `--refresh-returns` re-downloads. A pinned fixture `data/returns_fixture.parquet` (the same 11 assets, tracked) is what tests use; no test touches the network.
- [ ] **Timing (D11, strictly):** every join uses `available_at`. Label t is available on the first day of t+1, so the return of month t+1 is paired with label t. There is no contemporaneous pairing anywhere in the asset layer; the descriptive tables and the portfolio layer share this alignment.
- [ ] `regime_conditional_table`, `conditional_corr`, `regime_moments`, `mixture_moments`, `mixture_path` (60/40 expected return and vol over time from the walk-forward filtered probabilities), `growth_share_6040` (replaces the unsourced "95%+"), `sharpe_spread_placebo`.
- [ ] Outputs: `output/regime_returns.csv`, `output/regime_corr_<Regime>.csv`, `summary.json["assets"]` (window, n per regime inside the asset window, growth share, placebo percentile, skipped flag with reason if the download failed).
- [ ] Figures: fig 8 conditional annualised returns per asset and regime with bootstrap CIs; fig 9 probability-weighted 60/40 expected return and vol over time, NBER shaded.
- [ ] Tests on the fixture: alignment uses `available_at` (a label dated t never meets return t); table n per regime sums to the window; bootstrap SEs positive and seed-stable; mixture reduces to the single-regime moments when probs are one-hot; growth share in [0, 1].

**Stage 6 — portfolios and the achievable backtest (decided 2026-09-04).**
- [ ] `mv_weights`: the notebook's optimiser with its guards (pseudo-inverse; negative-sum and rank-deficiency counted, never silenced; gross-leverage cap 3.0). Unconstrained long-short is kept for continuity of the look-ahead decomposition; long-only is a later option.
- [ ] `backtest`: decision at the end of month d using labels available by then (label d-1 under a 1-month lag), weights earn month d+1, expanding-window regime moments from returns <= d aligned by `available_at`. Strategies: `PIT_MaxSharpe`, `PIT_MinVar`, `ProbWeighted_MaxSharpe` (mixture moments on the walk-forward filtered probabilities; the §1 purpose-3 strategy), `Oracle_MaxSharpe` (full-sample smoothed labels: label look-ahead), `Static_6040`, `EqualWeight`. Fallback to 60/40 when the current regime has fewer than `min_regime_obs = 15` paired months. Start 2010-01. Turnover tracked; `cost_bp` applied per unit turnover and reported at 0 and 10 bp.
- [ ] Look-ahead decomposition as the notebook's Cell L: in-sample (full-sample moments, smoothed labels) -> oracle -> PIT; `moment_lookahead`, `label_lookahead`, `total`. Reported in `summary.json["assets"]["lookahead"]` and as report-only rows in the acceptance table (`pit_sharpe`, `oracle_sharpe`, `insample_sharpe`, `label_lookahead`, `moment_lookahead`, `growth_share_6040`, `backtest_placebo_pct`); no thresholds.
- [ ] `backtest_placebo`: 200 label shuffles (each reruns the backtest), percentile of the real PIT Sharpe.
- [ ] Outputs: `output/backtest_returns.csv`, `output/portfolio_weights.csv` (PIT_MaxSharpe and ProbWeighted weights by month); figures fig 10 wealth curves (log) with drawdown panel, fig 11 PIT max-Sharpe weights over time.
- [ ] `run.py`: an `--assets/--no-assets` stage (default on) that runs after the engine publishes. A download failure or a missing cache skips the stage, records `summary.json["assets"]["skipped"]` with the reason, and does not change the engine's exit code.
- [ ] Tests on the fixture: no return of month r is ever paired with a label of month >= r; weights sum to 1 (or -1 with the counter incremented); fallback fires below `min_regime_obs`; a synthetic panel with a planted regime premium gives PIT Sharpe above 60/40 and oracle >= PIT; cost 10 bp lowers every active strategy's return by turnover x 0.001 exactly; the decomposition terms sum.

**Stage 7 — replace the original pipeline (after Stage 4 is green; see §2.2).**
- [x] Move `hysteretic_sign` into `regime_v2/regimes.py` with its test; delete `regime_core.py`.
- [x] Re-point `app.py` / `ui_io.py` at `output/regime_labels.csv` and `output/summary.json`; remove the notebook Refresh path and the `ui_data/` contract; rewrite `tests/test_ui_io.py`.
- [x] Retire `Macro_Regime_Analysis.ipynb` and the root-level generated files; update `README.md` and `.gitignore`.
- [x] Repo-wide grep for `Q1|Q2|Q3|Q4` returns nothing; `pytest -q` green; `python run.py data/fredmd_2026-07.csv` rebuilds everything the dashboard needs.
- [x] Deployment checklist in §12 completed on the Mini. (deployment checklist executed on the Mini 2026-09-04: cloned to ~/apps/states, image built, pinned run published on the volume, Streamlit on localhost:8505 beside regimes:8503, terrarium:8502, test:8501; Cloudflare published application route `states.lazyeconomist.com` -> HTTP localhost:8505 added on the lazyeconomist.com tunnel; https://states.lazyeconomist.com answers 200. Port 8505 confirmed. Remaining: press Refresh once with the current vintage; add 8505 = states to the MacMiniHosting port table), the tunnel route still has to be published in Cloudflare Zero Trust)

## 7. Figures (the paper pulls these directly; keep filenames)

- `fig1_factors_gaps.png` — level, recursive trend, real-time gap; growth and inflation panels; NBER shading.
- `fig2_regime_timeline.png` — four strips: hysteresis quadrants, walk-forward HMM (primary), free HMM, GMM. Challenger strips use their own names and colours, not the regime palette.
- `fig3_state_space.png` — scatter coloured by walk-forward HMM labels, n per regime in legend.
- `fig4_hmm_probabilities.png` — stacked **filtered** probabilities (walk-forward). A smoothed version, if shown, is a separate panel captioned "smoothed (ex-post)".
- `fig5_revisions.png` — real-time vs ex-post growth gap with corr / noise-to-signal / sign agreement in title.
- `fig6_classifier_comparison.png` — 2×2 state-space panel (Stage 3).
- `fig7_walkforward.png` — walk-forward vs full-sample smoothed labels (Stage 4), NBER lags annotated.
- `fig8_regime_returns.png` — annualised return per asset and regime with block-bootstrap CIs (Stage 5).
- `fig9_mixture_6040.png` — probability-weighted 60/40 expected return and vol over time from walk-forward probabilities (Stage 5).
- `fig10_backtest_wealth.png` — log wealth curves for the six strategies with a drawdown panel (Stage 6).
- `fig11_pit_weights.png` — PIT max-Sharpe weights by month (Stage 6).

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
| End-to-end label agreement, `asof=2007-12` vs full | ≥ 0.80 | Prototype measured 0.72; passes at 0.953 after the D3 drift fix |
| Filtered vs smoothed label agreement | reported, no threshold | Measured 0.84; goes in the paper as the cost of real-time labelling |
| Seed invariance | exact | Same labels for seeds 0, 1, 2 |

**Declared known failures.** `acceptance.KNOWN_FAILURES` lists thresholds that are reported in every table and in `summary.json` but do not block publishing. Each carries a reason. Adding or removing one is a §10 decision. Currently declared: `non_nber_contraction_hmm`.

Measured walk-forward table (2026-07 vintage, 571 months from 1978-12, 240-month window):

| Test | Measured |
|---|---|
| GFC Contraction (HMM) | 0.90 |
| GFC Contraction (hysteresis quadrants) | 0.80 |
| COVID Contraction | 0.75 |
| 2021-06..2022-12 high inflation | 1.00 |
| NBER low growth | 0.98 |
| Non-NBER Contraction | 0.233 (declared known failure; 0.201 on full-sample filtered labels; 0.207 / 0.159 at the old 120 window) |
| Share max filtered prob > 0.95 | 0.408 |
| Emission-only agreement | 1.00 |
| Means unmoved | 0 |
| Min transition prob | 0.0055 |
| Trend-step real-time | 0 |
| Truncation 2015 HMM / quad | 0.986 / 0.996 |
| Truncation 2007 HMM / quad | 0.966 / 0.976 |
| Seed invariance | 0 |
| Filtered vs smoothed agreement | 0.786 |

## 9. Open questions

1. ~~Existing data loaders~~ — answered: none reusable (§2.2).
2. ~~Asset universe and return source~~ — answered 2026-09-04: the notebook's 11-ETF universe via yfinance adjusted close (total return), common history from 2007-07; S&P Global indices remain a later upgrade behind `load_returns(source=...)`.
3. Should the three-axis taxonomy (growth/inflation/policy + liquidity modifier) consume these labels, or stay separate? Affects whether `regime_labels.csv` needs a `policy` column stub.
4. ~~Trend window~~ — answered 2026-09-04: **240-month trailing mean** (best revision statistics: N/S 0.239, sign agreement 0.916; medians and Hamilton markedly noisier). Truncation agreement at 240: HMM 0.986 / 0.966, quadrants 0.996 / 0.976.
5. ~~Replace or run alongside the notebook~~ — answered: replace (§2.2, Stage 7).
6. Default θ for hysteresis quadrants: reuse the notebook's `HYSTERESIS_THETA` or re-tune on the FRED-MD gaps?
7. ~~Hosting on lazyeconomist.com~~ — answered 2026-09-04: served live as the Streamlit `app.py` in a Docker container on the Mac Mini (OrbStack), app dir `~/apps/states`, Cloudflare Tunnel route `states.lazyeconomist.com` -> `localhost:8505` (port confirmed 2026-09-04 when the route was created). Refresh is the in-app button running `run.py --vintage`. Deploy is push-to-GitHub then the user's `deploy states` ssh function (git pull + `docker compose up -d --build`).
8. ~~`non_nber_contraction_hmm`~~ — answered 2026-09-04: the 0.10 threshold is **not relaxed** and the metric is not redefined; it stays a declared known failure, reported in every table (walk-forward 0.233 at the 240 window, 0.201 on full-sample filtered labels).

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
- 2026-09-04 — D3 amended: factor scaled, not demeaned, with drift term. Reason: demeaning on the estimation sample turned the sample mean into a drift in the diffusion index and a 0.3 SD sample-dependent offset in the inflation gap. See D3. (controller ruling during execution; confirm)
- 2026-09-04 — D6 amended: diagonal pooled covariance. See D6. (controller ruling during execution; confirm)
- 2026-09-04 — EM convergence judged on sign-aligned loadings; every row scored by the regression of its observed cells on the converged loadings; guards added for zero-variance columns and zero-denominator rows. (controller ruling during execution; confirm)
- 2026-09-04 — Declared known failures mechanism added to §8; `non_nber_contraction_hmm` declared (walk-forward 0.207): the excess months are 1991-04..1993-10, 1986, 2024-07..2026-02, i.e. below-trend growth with below-trend inflation. Threshold not relaxed; user decision pending (§9 Q8).
- 2026-09-04 — `quadrant_profile` fills clusters that never win an argmax uniformly so marginalisation conserves mass.
- 2026-09-04 — `nber_lags` flags left-censored lags. Measured walk-forward lags at NBER peaks 1980-01, 1981-07, 1990-07, 2001-03, 2007-12, 2020-02: 0 months (low growth = Contraction or Stagflation already called at the peak; peaks before 1978-12 are outside the walk-forward).
- 2026-09-04 — Block bootstrap start range corrected to include the last valid block.
- 2026-09-04 — Stage 1–4 implemented per `docs/superpowers/plans/2026-09-03-regime-v2-engine.md`; full run 82 s.
- 2026-09-04 — Parked for the final review: `standardise_expanding`/`revision_stats` zero-std division; `labels_frame` column-collision error; `run_pipeline` early-`asof` opaque error.
- 2026-09-04 — Trend window 240 adopted (D4, §9 Q4) on the Stage 2 revision statistics; acceptance table re-run, all thresholds pass except the declared known failure. User decision.
- 2026-09-04 — `non_nber_contraction_hmm` threshold kept at 0.10 and the metric not redefined (§9 Q8); it remains a declared, reported, non-blocking known failure. User decision.
- 2026-09-04 — FRED-MD download URL pattern verified against the St. Louis Fed page: `/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/YYYY-MM-md.csv`. The download itself could not be exercised from the dev machine (connection timed out); first exercise on the Mini.
- 2026-09-04 — Stages 5–6 designed and adopted (§6): 11-ETF yfinance universe; strict `available_at` timing for every asset join; unconstrained long-short MV with a 3x gross cap (continuity with the July decomposition); transaction costs as a parameter reported at 0 and 10 bp; backtest start 2010-01; probability-weighted max-Sharpe strategy added; backtest placebo with 200 shuffles. User approved the design; the four numbered choices were the controller's defaults.
- 2026-09-04 — Stages 5–6 implemented per `docs/superpowers/plans/2026-09-04-regime-v2-assets.md`; wired into `run.py` as a post-publish asset stage (Task 7). Measured on the pinned vintage (`data/fredmd_2026-07.csv`, walk-forward filtered labels, window 2007-08..2026-09, 230 months): PIT Sharpe 0.75, oracle Sharpe 0.82, in-sample (ex-post) Sharpe 1.58 (moment look-ahead +0.76, label look-ahead +0.07); growth share of the 60/40 regime-conditional R^2 51.3% (R^2 0.008, n=230); Sharpe-spread placebo percentile 31.9 (real spread 0.93 vs 1000 run-preserving shuffles); PIT max-Sharpe backtest placebo percentile 25.0 (real Sharpe 0.75 vs 200 shuffles). Asset stage wall-clock ~100 s (cached returns, 200-shuffle placebo included) on top of ~8 s for the engine alone; the full real run (engine + walk-forward + asset stage) took 2m57s.
- 2026-09-04 (revision, Task 7 code review) — Two fixes to the above: (1) `run_assets` now wraps its *entire* body in one try/except, not just `assets.load_returns` — a failure anywhere in the stage (a computation, a figure write) after `publish()` has already swapped `output/` into place must not change the engine's exit code, matching §6 Stage 6's rule; (2) `assets.returns_to_monthly` now drops a final calendar month whose last observed price date is before that month's last business day (`pd.offsets.BMonthEnd`), since a download run mid-month (e.g. the 4th) otherwise reports a 2-3-trading-day "month" as a completed one. `data/returns_fixture.parquet` was regenerated via a fresh yfinance download and now correctly ends 2026-08-01 (was 2026-09-01, a 3-day partial month). Re-measured on the same pinned vintage after regenerating the default returns cache (window 2007-08..2026-08, 229 months, one fewer than before because the partial month is gone): PIT Sharpe 0.75, oracle Sharpe 0.82, in-sample Sharpe 1.59 (moment look-ahead +0.77, label look-ahead +0.06); growth share 51.4% (R^2 0.008, n=229); Sharpe-spread placebo percentile 31.9; backtest placebo percentile 24.5. All within noise of the first measurement; no threshold or `KNOWN_FAILURES` entry changed. Suite-wide, an autouse `conftest.py` fixture now monkeypatches `assets._download_yfinance` to raise, so no test (present or future) can reach the live network regardless of its cache/fetch arguments.
- 2026-09-04 (revision, whole-branch review of `regime-v2-assets`) — Nine fixes, no threshold and no `KNOWN_FAILURES` change. (1) The asset stage no longer substitutes labels: with the walk-forward disabled it copied `hmm_filtered` into `hmm_walkforward` and published full-sample labels under PIT names, so `pit_sharpe` was not point-in-time. It now skips with `"walk-forward disabled: the asset stage needs real-time labels"` and `summary.json["assets"]["label_column"]` is gone. (2) The stage publishes all-or-nothing via `output/.assets_staging` and `figs/.assets_staging` after clearing the previous run's artefacts, so a mid-stage failure can no longer leave this run's early files beside the previous run's late ones; promoting the metrics into `acceptance.csv` is inside the same guard. (3) `acceptance.csv` gained three report-only benchmark rows — `static_6040_sharpe`, `pit_sharpe_10bp`, `growth_share_6040_r2` — and per-name rationales for `pit_sharpe`, `growth_share_6040` and `backtest_placebo_pct`. Measured: PIT max-Sharpe 0.751 vs Static_6040 **1.051** at 0 bp, and 0.611 vs 1.050 at 10 bp — the achievable strategy underperforms plain 60/40, which the bare `pit_sharpe` row did not show. (4) fig9's title now reads "full-sample regime moments x walk-forward probabilities (descriptive, not achievable)" and its x-axis is `available_at`, not the label date. (5) The `BMonthEnd` partial-month rule of the entry above is superseded: `returns_to_monthly` now drops the last calendar month present unless a later month has observations (in practice always the final month). `BMonthEnd` is holiday-blind — a month whose last trading day precedes the last business day (Memorial Day, Good Friday, a Saturday month-end) looked partial — while a calendar-day tolerance keeps months that really are short. (6) `pit_fallback` split into `pit_maxsharpe_fallback` / `pit_minvar_fallback` (35 / 35 on the pinned vintage). (7) `portfolio._moments` folded into `assets.moments_by_label`, used by both. (8) Housekeeping: unused `REGIMES` import and `run_assets(res=...)` removed, mid-file `noqa: E402` placebo imports hoisted (no cycle), bootstrap-SE and `maxdd` semantics documented. (9) §5 re-synced with the code. Re-measured on the pinned vintage, `data/returns_fixture.parquet` regenerated (still 229 rows, 2007-08..2026-08; values differ by ≤2.8e-6 from vendor adjusted-close revisions): PIT Sharpe 0.751, oracle 0.816, in-sample 1.588 (moment look-ahead +0.772, label look-ahead +0.064); growth share 51.4% (R² 0.008, n=229); Sharpe-spread placebo percentile 31.9; backtest placebo percentile 24.5 — identical to the previous measurement. Full run 3m00s; suite 101 passed, 1 skipped in 2m00s.
- 2026-09-04 — Stage 7 designed and approved: dashboard reads `regime_v2/output/` directly through a small loader; label vocabulary is the spec's four names; in-app refresh runs `run.py`; Docker deployment per §12; notebook, `regime_core.py`, `ui_io.py`, `ui_data/`, `main.py` and root artefacts deleted (`hysteretic_sign` moves into `regime_v2/regimes.py`). Route `states.lazyeconomist.com`; port 8505 assumed.
- 2026-09-04 — First live refresh on the Mini (vintage 2026-08) exposed a malformed FRED-MD file, not a data revision: the header line lost `S&P div yield` while the data kept the column, so every series from column 75 on was mislabelled by one (the column named CPIAUCSL held PPICMM, PPICMM held OILPRICEx, the last real column sat under an unnamed header). The engine ran on the shifted panel and the §8 gate caught it (`inflation_2021_22_high_hmm` 0.68 < 0.90), so the 2026-07 outputs stayed published as §12 requires. Fixes: `load_fredmd` now repairs a header that lost exactly one name against the pinned vintage (re-aligning the t-code row, which in that file stayed aligned with the short header, by name) and refuses with `VintageError` any vintage whose block series fail a level-correlation check (≥ 0.98 on the post-1990 overlap) or whose t-codes differ from the pinned vintage; a trailing blank column is ignored; the 10×IQR rule skips a column whose IQR is zero on the estimation rows (undefined there). The FRED-MD download works from the Mini; the pinned-vintage results are unchanged by these fixes. The same refresh also showed yfinance's threaded downloader returning one ticker (EEM) empty inside the container (sqlite "database is locked"), which emptied the common history and overwrote the returns cache with nothing: `load_returns` now downloads single-threaded, treats an all-NaN ticker or fewer than 120 months of common history as a failed download (falling back to the cache on a refresh), and writes the cache only after validation.
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
- **Published surface:** the dashboard (`app.py`) reading `output/`, plus the seven engine figures, the four asset-stage figures (`fig8_regime_returns.png`, `fig9_mixture_6040.png`, `fig10_backtest_wealth.png`, `fig11_pit_weights.png`), `regime_labels.csv`, and the four asset-stage CSVs (`regime_returns.csv`, `regime_corr_<Regime>.csv`, `backtest_returns.csv`, `portfolio_weights.csv`) as downloadable files. Nothing on the site is produced by the notebook.
- **Secrets:** none required. FRED-MD is a public CSV; `FRED_API_KEY` is no longer needed once the notebook is retired.
- **Pinned vintage:** `data/fredmd_2026-07.csv` stays in git so the site can rebuild without network access and the tests are reproducible on the Mini.

Serving mechanics (decided 2026-09-04, from the IPS and MacMiniHosting runbooks): one directory per app at `~/apps/states/` on the Mini, a root `Dockerfile` (python:3.12-slim, `pip install -r requirements.txt`, `streamlit run app.py --server.port=8505 --server.address=0.0.0.0 --server.headless=true`), `docker-compose.yml` with `restart: unless-stopped` and named volumes for `regime_v2/output`, `regime_v2/figs` and the return cache so a rebuild keeps the last published outputs, `.dockerignore`. Exposure is the dashboard-managed Cloudflare Tunnel route `states.lazyeconomist.com` -> HTTP `localhost:8505` (already created by the user). Deploy: `deploy states` from the Windows PowerShell profile (ssh `jameswalsh@JHCW-mini.local`, `git pull`, `docker compose up -d --build`); first deploy clones into `~/apps/states`. Refresh: the dashboard's Refresh button runs `python regime_v2/run.py --vintage YYYY-MM` under a lock file; the driver's staged publish keeps the last good outputs on failure. No cron, launchd or GitHub Action; the image ships the pinned vintage so the first start publishes offline. Port table in MacMiniHosting to be updated by the user.

Deployment checklist (Stage 7, to run on the Mini once per new app; later deploys are `deploy states`):
1. `mkdir -p ~/apps && cd ~/apps && git clone https://github.com/jhcwalsh/StateAnalysis.git states && cd states`
2. `docker compose up -d --build` — the entrypoint publishes the pinned vintage on the first start (≈3 min), then serves on 8505. The tunnel returns 502 until the pinned run finishes (~3 min).
3. In Cloudflare Zero Trust, confirm the published application route `states.lazyeconomist.com` → HTTP `localhost:8505`.
4. Open the site; the status header must show the current regime and "all acceptance tests passed" (or the declared known failure).
5. Press Refresh with the current vintage; confirm the run completes and the caption's vintage changes.
6. Add `8505 = states` to the port table in MacMiniHosting's runbook.
Logs: `docker logs -f states`. Rebuild after a push: `deploy states`. The volume `states_var` holds the published outputs; `docker volume rm states_var` resets the site to the pinned vintage on next start.
