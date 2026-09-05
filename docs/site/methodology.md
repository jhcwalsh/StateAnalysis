# The States engine: methodology

*The Lazy Economist · {{run.date}} · engine regime_v2 · data through {{run.asof_long}}*

## Abstract

States classifies each month of US macroeconomic history into one of four growth–inflation regimes —
`Contraction`, `Goldilocks`, `Overheating`, `Stagflation` — using only information a reader could
have had in that month. Two factors, one for real activity and one for prices, are extracted from a
FRED-MD vintage, converted into one-sided gaps against a trailing trend, and passed to a four-state
hidden Markov model whose emission means are fixed symmetrically about the axes, so the model adds
persistence to a sign rule rather than replacing it. The whole pipeline is re-estimated every month
on data up to that month, and the filtered posterior for the last month is the published label. On
the current run the label for {{current.month_long}} is {{current.regime}}, {{acc.n_passed}} of
{{acc.n_tests}} acceptance thresholds pass, and a point-in-time portfolio built on these labels earns
a Sharpe ratio of {{bt.pit}} against {{bt.sharpe_Static_6040}} for static 60/40. The regime signal is
real; the trading edge is not.

## Motivation and the question

Most published regime classifications are fitted once, on the whole sample, and then displayed
across history as though they had been available all along. Such a display is honest about the
estimator and silent about the information set, and it is the information set that matters. A label
for March 2009 estimated on data through 2026 says how the past looks now; a label for March 2009
estimated on data through March 2009 says what could have been known then. The difference is the
whole quantity of interest for anyone who wants to act on a regime call.

That gap is well documented. Orphanides and van Norden (2002) showed that output-gap estimates
revise so heavily that the real-time series is nearly uninformative about the final one; Croushore
and Stark (2001) built the data infrastructure that made such comparisons routine; Chauvet and Piger
(2008) and Hamilton (2011) evaluated business-cycle dating rules on what they would actually have
said at the time. States takes the same position for growth–inflation regimes: the object of interest
is the walk-forward label, and every full-sample quantity here is marked as an ex-post comparator.

"Regime" here has a deliberately narrow meaning. It is a sign pair: whether the growth gap is above
or below its trailing trend, and whether the inflation gap is above or below its. That is the
growth–inflation quadrant taxonomy familiar from the Investment Clock (Greetham and Hartnett, 2004),
made operational. It is not a recession indicator, not a forecast, and not a claim about causal
mechanism. `Contraction` means below-trend growth *and* below-trend inflation, which is a different
event from an NBER recession — a distinction that costs the engine one acceptance test, discussed in
§ Validation.

![The pipeline from FRED-MD vintage to labels and portfolios](fig:doc_pipeline)

## Data

The input is a single monthly FRED-MD vintage (McCracken and Ng, 2016): a wide panel of US
macroeconomic series with a header row of transformation codes. The published run uses vintage
`{{run.vintage}}`. The FRED-MD panel begins in 1959; the first month with a complete pair of gaps is
{{sample.start}}, after the trend window and the standardization burn-in. Each series is transformed
by its t-code — levels, first and second differences, logs, log differences, second log differences —
so that every column is stationary in the sense the FRED-MD authors intend.

Outliers are removed with the FRED-MD rule: a cell is set to missing when
$|x_t - \mathrm{median}(x)| > {{params.k_outlier}} \times \mathrm{IQR}(x)$. The median and the
interquartile range are computed on the rows outside the COVID mask described next, and in the
walk-forward path on data up to the month being labeled, so the threshold that flags a month never
uses that month's future. A column whose interquartile range is zero on those rows is skipped: the
rule is undefined there.

Two kinds of month are excluded from estimation but still transformed, classified, and reported. The
first is the COVID window {{params.mask}} (decision D9). The 2020 collapse and rebound are real, but
treating them as draws from the same distribution as the rest of the sample inflates the pooled
variance by roughly half and produces a multi-standard-deviation whipsaw in the growth gap. The
second is the thin-month rule (D3), applied after outlier removal: a month in which fewer than half of
a block's series survive the rule cannot support a factor score estimated on the whole cross-section.
Coverage is taken as the minimum across the two blocks, so a month thin in either block is excluded
for both.

The panel is split into two blocks, listed explicitly in `data.py`. The growth block holds
{{panel.n_growth}} real-activity series — industrial production and capacity utilization, payrolls by
sector, unemployment, claims, retail sales and consumption, real income, housing starts and permits.
The inflation block holds {{panel.n_inflation}} price series — CPI variants, PCE deflators, producer
prices, and average hourly earnings. Counter-cyclical series such as the unemployment rate are *not*
sign-flipped: principal components are invariant to column sign, and the anchoring step below fixes
orientation unambiguously.

Vintages drift, and not only through revision. A refresh on a later vintage found a file whose header
line had lost one series name while the data kept the column, so every series from that point on was
labeled with its neighbor's name; the engine ran on the shifted panel and the acceptance gate caught
it. The rule adopted in response (§10, 2026-09-04) is repair-then-refuse: `load_fredmd` reconstructs
a header that lost exactly one name against the pinned vintage, re-aligning the t-code row by name,
then refuses to publish — raising `VintageError` — any vintage whose block series fail a
level-correlation check against the pinned vintage on the post-1990 overlap, or whose t-codes have
changed. Revisions barely move a level correlation; a misaligned header destroys it. A refusal leaves
the previously published outputs in place.

## Factors

Each block is reduced to one factor. Columns are standardized using means and standard deviations
computed on estimation rows only, and the first principal component is taken from the singular value
decomposition of those rows. Missing cells — from the outlier rule and from ragged series histories —
are handled by EM imputation (Dempster, Laird, and Rubin, 1977): missing entries are filled with the
current rank-one reconstruction, the loadings are recomputed, and the loop repeats. Convergence is
judged on the loading vector after aligning its sign to the previous iteration, because the SVD's
sign is arbitrary and an unaligned comparison never converges.

The converged score for a month is not read off the imputed matrix but computed in closed form from
that month's observed cells:

$f_t = \dfrac{\sum_{i \in O_t} l_i z_{it}}{\sum_{i \in O_t} l_i^2}$

where $O_t$ is the set of series observed in month $t$, $l_i$ the converged loading, and $z_{it}$ the
standardized value. This is the fixed point the imputation loop converges to, so a thin month's score
does not depend on how many iterations were run; on complete rows it coincides exactly with the usual
projection. The construction is the diffusion-index estimator of Stock and Watson (2002a, 2002b),
with one factor per block by design rather than by an information criterion — Bai and Ng (2002) is
the reference when the number of factors is itself the question.

Orientation is fixed by anchoring: the sign is chosen so the score correlates positively with the
transformed `INDPRO` for growth and `CPIAUCSL` for inflation, over estimation rows.

The score is **not demeaned**. Instead a drift term

$d = \sum_j \dfrac{l_j \mu_j}{s_j}$

is added to it, where $\mu_j$ and $s_j$ are the estimation-row mean and standard deviation of series
$j$, and the sum is then scaled by its own estimation-row standard deviation. The reason is specific
to what happens next. The inflation factor is a rate, and the object the
gap is built on is its cumulative sum — a partial-sum diffusion index, an inflation *level*.
Demeaning on the estimation sample would turn that sample's mean into a drift in the level, which the
trailing-mean trend then converts into a constant offset in the gap. Because the estimation sample
grows month by month in the walk-forward path, that offset would be sample-dependent: it was traced
as a shift of roughly a third of a standard deviation between full-sample and truncated runs, and
removing it is what lifted end-to-end truncation agreement from failing to passing. With the drift
restored, the cumulative sum of the factor is the loading-weighted cumulated raw series up to a
constant. The growth factor is used as-is, since it is already a growth rate.

![Growth and inflation factor loadings by series](fig:doc_loadings)

![Factor levels, trailing trends, and the resulting gaps](fig:fig1_factors_gaps)

Loadings behave as the series list suggests: industrial production, capacity utilization and payrolls
dominate the growth factor, with unemployment and claims entering negatively; CPI variants, the PCE
deflators and producer prices dominate the inflation factor, with average hourly earnings near zero.
The app's Factor levels and Factor gaps tabs show both.

## Trend gaps

A regime is defined against a trend, and the choice of trend filter is where most real-time claims
quietly fail. Two-sided filters — the Hodrick–Prescott filter (Hodrick and Prescott, 1997), a
centered moving average, any forward-backward smoother — use future observations to locate the trend
at $t$. They are excluded from the labeling path entirely (decision D5). Hamilton (2018) sets out the
case against the HP filter specifically; Orphanides and van Norden (2002) show the more general
result that endpoint estimates from two-sided filters revise enough to reverse their own sign.

The gap used here is one-sided by construction. For a series $y$ the engine takes a three-month
moving average and subtracts its trailing mean over a window of {{params.window}} months:

$\mathrm{gap}_t = \bar y_t - \dfrac{1}{W}\sum_{k=0}^{W-1} \bar y_{t-k}, \qquad W = {{params.window}}$

The trailing mean requires only sixty observations rather than a full window, so the earliest trends
average as few as sixty months. The gap is then divided by an expanding-window standard deviation,
accumulated over estimation rows only so masked months do not move it, though every month is scored
with the last available value. No mean is subtracted at the standardization step — the trend
subtraction has done that — so the gap is a scale-free deviation from the trailing trend, nothing
more.

The window was chosen on revision statistics, not on whether the current label looked right. The
default trend is `smoothed_trailing` — the three-month moving average, then the trailing mean — and
the robustness grid compares it against trailing means and medians at several window lengths and
against a recursive Hamilton regression, which has no window at all, scoring each on noise-to-signal
ratio and sign agreement versus a two-sided comparator. The app reports that grid when the run
includes it. The alternatives live in `trend.py` for that table alone and never enter the labeling
path.

![Real-time gap versus the two-sided ex-post comparator](fig:fig5_revisions)

The revision evidence is the direct test of whether a one-sided gap is worth having. The figure plots
the quasi-real-time growth gap against a two-sided, full-sample comparator, with correlation,
noise-to-signal ratio and sign agreement in the title. The end-to-end version of the question is
stronger, because it re-runs the entire pipeline on truncated data. Labels from a sample truncated at
the end of 2015 agree with the full-sample labels on the overlap at a rate of
{{acc.trunc_2015_agreement_hmm}}; truncated at the end of 2007, {{acc.trunc_2007_agreement_hmm}}.
Those agreements are end-to-end — re-estimated loadings, re-estimated outlier thresholds, a refit
HMM. Only the trend step is exactly real-time, and that exactness is tested separately.

## Regimes

### Quadrants with causal hysteresis

The transparent reference classifier is the sign rule: `Contraction` when both gaps are negative,
`Goldilocks` when growth is above trend and inflation below, `Overheating` when both are above,
`Stagflation` when growth is below and inflation above. A memoryless sign rule flickers whenever a
gap sits near zero, so the rule is given memory in a way that cannot see forward. Each gap passes
through a Schmitt trigger (Schmitt, 1938): the sign state flips from positive to negative only when
the gap falls below $-\theta$, and back only when it rises above $+\theta$, with
$\theta = {{params.theta}}$ (decision D10). The state at $t$ depends on history up to $t$ and nothing
else. There is no post-hoc merging of short runs and no minimum-duration rule anywhere in the engine;
both would import future information into a past label.

![The growth–inflation plane, the four quadrants, the hysteresis band, and the current month](fig:doc_quadrants)

### The constrained hidden Markov model

The primary classifier is a four-state Gaussian hidden Markov model on the pair of gaps (Hamilton,
1989; Kim and Nelson, 1999; Rabiner, 1989), constrained so that it is a persistence device over the
quadrant rule rather than an independent clustering of the data (decision D6). Its emission means
are fixed and symmetric about the axes,

$\mu_k = (\pm c_g,\ \pm c_\pi), \qquad c_g = \mathrm{mean}\,|g_t|,\ \ c_\pi = \mathrm{mean}\,|\pi_t|$

with the means taken over estimation months. One pooled within-quadrant covariance is computed from
residuals about those fixed means and stored **diagonal**, so that with symmetric means the
emission-only decision boundaries are exactly the axes. Only the initial-state and transition
probabilities are estimated. The transition matrix carries a Dirichlet prior $1 + \varepsilon +
\kappa I$ with $\varepsilon = {{params.eps}}$ off the diagonal and $\kappa = {{params.persistence}}$
on it: $\kappa$ supplies persistence, and $\varepsilon$ exists so that no transition can be estimated
as exactly zero, which an unregularized fit did, leaving the portfolio layer with an impossible move.
Freezing the covariances is a workaround for a tied-covariance M-step that divides by zero when means
are frozen, not a modeling preference.

The model is fit on the contiguous estimation segments only, so masked months contribute nothing to
the transition counts. Labels come from a forward filter written directly rather than taken from the
library, so that the causal object is unambiguous:

$P(s_t = k \mid x_1 \ldots x_t) \propto \phi_k(x_t) \sum_j P(s_{t-1} = j \mid x_1 \ldots x_{t-1})\,A_{jk}$

The forward–backward posterior is also computed, but only as an ex-post comparator, always captioned
as smoothed. Where a full-sample output is returned beside its causal counterpart it carries an
`_expost` suffix — `labels_smoothed_expost`, `probs_smoothed_expost` — so the two cannot be confused.

![Filtered regime probabilities](fig:fig4_hmm_probabilities)

![The estimated transition matrix and expected durations](fig:doc_transition)

{{hmm.transition_table}}

The published transition matrix is the one from the full-sample fit. Expected durations follow from
its diagonal as $1/(1 - A_{kk})$: {{hmm.expected_duration_Contraction}} for `Contraction`,
{{hmm.expected_duration_Goldilocks}} for `Goldilocks`, {{hmm.expected_duration_Overheating}} for
`Overheating`, {{hmm.expected_duration_Stagflation}} for `Stagflation`. Mean realized run lengths are
a separate measurement, counted on the walk-forward labels: {{hmm.mean_run_Contraction}},
{{hmm.mean_run_Goldilocks}}, {{hmm.mean_run_Overheating}} and {{hmm.mean_run_Stagflation}}. The two
are not directly comparable — one is a property of a chain estimated once on the whole sample, the
other a count over labels from a model refit every month. The share of walk-forward months in each
regime is {{hmm.share_Contraction}}, {{hmm.share_Goldilocks}}, {{hmm.share_Overheating}} and
{{hmm.share_Stagflation}}. Probabilities move: only {{hmm.max_prob_share}} of months have a maximum
filtered probability above 0.95.

### Challengers

Two data-driven classifiers are fit on the same inputs as challengers, never as the primary label: a
Gaussian mixture with four full-covariance components, and an unconstrained HMM with free means and
full covariances (decision D7). Both recover a crisis-versus-normal structure rather than four sign
quadrants, reported as a finding rather than hidden by renaming. They appear under descriptive
centroid names from their own fitted means, never under quadrant names, and are bridged to the
quadrant taxonomy by marginalization,

$P(q \mid t) = \sum_k P(\text{cluster } k \mid t)\; P(q \mid \text{cluster } k)$

with $P(q \mid \text{cluster } k)$ the empirical profile of the memoryless ($\theta = 0$) quadrants
within each cluster, and a uniform row for any cluster that never wins an argmax so probability mass is
conserved. No one-to-one cluster-to-quadrant map is used; it was non-injective every time it was
tried.

![The four classifiers in the same state space](fig:fig6_classifier_comparison)

## The walk-forward label

The published label is not a full-sample fit read backwards. For each month $t$ from
{{sample.wf_start}} onward, the engine re-runs the entire pipeline with `asof = t`: it truncates the
raw vintage at $t$, recomputes outlier thresholds, re-estimates the block standardizations and the
PCA loadings, rebuilds the trend and the expanding standardization, refits the HMM, and keeps the
**filtered** posterior for $t$ alone. Nothing after $t$ is visible at any step. That is
{{sample.wf_n}} separate estimations of the whole model.

Filtered rather than smoothed is not a detail. Within a single full-sample fit, the filtered and
smoothed labels agree on {{hmm.filtered_vs_smoothed}} of months; the remainder is the cost of
reading history forward instead of backward, and it is a cost the paper pays rather than hides.

One further piece of timing is needed before a label is usable. A FRED-MD vintage containing month
$t$ is published during $t+1$, so a month-$t$ label is not available at $t$. Every published row
carries an `available_at` column, set to the first day of the month after the label date
(publication lag {{params.lag}} month, decision D11), and every downstream join uses it. The asset
layer joins on `available_at` and never on the label date.

![Timing of one month: data, label, availability, and the return it may be paired with](fig:doc_timing)

![Walk-forward labels against the full-sample smoothed comparator](fig:fig7_walkforward)

![The regime timeline across all four classifiers](fig:fig2_regime_timeline)

## Validation

Acceptance tests are a publishing gate, not a report card. The history tests — recession windows, the
inflation window, the false-alarm rate, probability dispersion — run on walk-forward labels; the
model-form, truncation and seed tests run on the full-sample fit, which is what they are about.
`run.py` refuses to publish if any thresholded test fails other than a declared known failure: the
previous outputs stay live and the run exits non-zero. On the published run, {{acc.n_passed}} of
{{acc.n_tests}} thresholded tests pass.

{{acc.table}}

The declared known failure is {{acc.known_failures}}: the share of non-NBER months labeled
`Contraction` exceeds its threshold. The mechanism is understood. `Contraction` means below-trend
growth *and* below-trend inflation, not an NBER recession; the excess months cluster in the early
1990s, the mid-1980s and the recent period, all stretches of below-trend growth with below-trend
inflation. The threshold was calibrated against a version of the pipeline whose demeaned diffusion
index carried a sample-dependent inflation offset. Removing that offset was the right fix and made
this test worse. The decision (§9 Q8) was to keep the threshold, keep the metric as defined, and
declare the failure in every table rather than redefine either.

Timing against the business cycle is measured directly against the NBER Business Cycle Dating
Committee's own dates (NBER, n.d.). Of {{nber.n_peaks}} NBER peaks, {{nber.n_in_window}} fall inside
the walk-forward window; those give a mean lag of {{nber.mean_lag}} months from peak to the first
real-time low-growth label and a median of {{nber.median_lag}} ({{nber.n_censored}} left-censored and
excluded from those averages).
Peaks before the window opened are dropped, not averaged in.

Two more checks guard against artifacts of the fitting machinery. Seed invariance requires identical
labels across three random seeds. Emission-only agreement requires that the model's emission argmax —
the classification before any persistence is applied — matches the memoryless quadrant rule almost
exactly, which is the test that the HMM smooths the sign rule rather than replacing it.

Significance is assessed with run-preserving placebos rather than i.i.d. shuffles. Reshuffling labels
independently across months would destroy the persistence that defines a regime and make any
persistent signal look significant. Instead the series is split into its observed runs and the runs
permuted, so the run-length distribution is preserved exactly and only the alignment between regimes
and outcomes is destroyed. Confidence intervals on regime-conditional statistics come from a
bootstrap over fixed-length twelve-month blocks of the aligned panel, in the tradition of Künsch
(1989); see Politis and Romano (1994) for the random-length variant. Resampling in calendar time
means regime membership is resampled with the returns, which is the honest way to price in the fact
that regimes are few and long.

<!-- if:assets -->

## Assets and the achievable backtest

The asset layer answers one question honestly: if you had these labels in real time, what could you
have done with them? The universe is eleven exchange-traded funds spanning global equity, duration,
credit, real assets and inflation protection — {{assets.universe}} — measured as monthly simple total
returns from adjusted closes. Common history binds at the shortest fund's start, giving a window of
{{assets.window_start}} to {{assets.window_end}}, {{assets.n_months}} months.

Timing is the load-bearing part. Every join runs through `available_at`. In the descriptive tables the
return of month $r$ is paired with the most recent label whose `available_at` is at or before $r$ —
in practice the label for $r-1$, published at the start of $r$. In the backtest the join is strict: a
weight earning month $r$ may use only labels whose `available_at` falls strictly before $r$, so the
decision at the end of month $d = r-1$ acts on the label for $d-1$ and earns the return of $d+1$. No
return is ever paired with a label dated in that month or later, and a test enforces it.

![Regime-conditional annualized returns with block-bootstrap intervals](fig:fig8_regime_returns)

Regime-conditional tables report, per asset and regime, the count, annualized return and volatility,
Sharpe ratio, within-regime drawdown, hit rate, and block-bootstrap standard errors on the return and
Sharpe; the Correlations tab applies the same conditioning to the correlation matrix. Given
per-regime moments and the walk-forward probabilities, the mixture mean and covariance follow from
the law of total variance,

$\mu = \sum_k p_k \mu_k, \qquad \Sigma = \sum_k p_k \left[\Sigma_k + (\mu_k - \mu)(\mu_k - \mu)'\right]$

which gives an expected return and volatility path for a fixed-weight 60/40 portfolio. That path is
descriptive, not achievable — it uses full-sample regime moments — and its figure says so on its face
and indexes time by `available_at`.

![Probability-weighted 60/40 expected return and volatility](fig:fig9_mixture_6040)

How much of a 60/40 portfolio's monthly variation do the gaps explain at all? Regressing the 60/40
return on both gaps and splitting the $R^2$ between them by the LMG (Shapley) decomposition
(Grömping, 2006) attributes {{assets.growth_share}} of the explained variance to growth. That number
must be read next to the $R^2$ it is a share of, {{assets.r2}}. At that magnitude the split is
uninformative: it is a share of almost nothing, and stating it without the $R^2$ — as an earlier
version of this work did — makes a rounding error look like a result.

Six strategies are run, plus one ex-post comparator that exists only to measure look-ahead.
`PIT_MaxSharpe` and `PIT_MinVar` use point-in-time labels and expanding-window regime moments;
`ProbWeighted_MaxSharpe` uses the mixture moments above with the walk-forward probabilities;
`Static_6040` and `EqualWeight` are the passive benchmarks; `Oracle_MaxSharpe` uses full-sample
smoothed labels with expanding moments, isolating label look-ahead; and `InSample_MaxSharpe_expost`
uses full-sample moments *and* smoothed labels. The last is never presented as achievable — it is the
instrument that measures how much of a backtest is hindsight.

Weights are unconstrained long-short mean-variance (Markowitz, 1952; Sharpe, 1994), $w \propto
\Sigma^{+}(\mu - r_f)$ for maximum Sharpe and $w \propto \Sigma^{+}\mathbf{1}$ for minimum variance,
computed with a pseudo-inverse so a singular regime covariance degrades rather than throws. Weights
are normalized to net exposure of $\pm 1$: the sign of the raw solution is preserved rather than
flipped, and a negative net is flagged `negsum`. When gross exposure exceeds a cap of three, the long
and short books are rescaled separately so gross hits the cap while net is preserved — a uniform
scale-down cannot do this, since it shrinks net and gross together. Rank deficiency is flagged the
same way. When the current regime has fewer than {{bt.min_obs}} paired months of history the strategy
falls back to 60/40. Every flag and fallback is published, `negsum` included: {{bt.counters}}. The
backtest starts {{bt.start}}; costs are applied per unit of turnover and reported at zero and ten
basis points.

{{bt.perf0}}

{{bt.perf10}}

![Where the Sharpe ratio of a regime backtest comes from](fig:doc_lookahead)

The look-ahead decomposition is the central result of this section. In-sample — full-sample moments,
smoothed labels — the max-Sharpe regime strategy earns {{bt.insample}}. Replacing full-sample moments
with expanding ones, while keeping the smoothed labels, gives the oracle at {{bt.oracle}}: a moment
look-ahead of {{bt.moment_lookahead}}. Replacing smoothed labels with real-time walk-forward labels
gives the achievable point-in-time result, {{bt.pit}}: a label look-ahead of {{bt.label_lookahead}}.
When that term is negative the real-time labels beat the smoothed ones, and the whole apparent edge is
knowledge of the moments. Either way the achievable number must be read against the passive benchmark
in the same table: {{bt.pit}} for the point-in-time regime strategy against
{{bt.sharpe_Static_6040}} for the static portfolio, over the same window at zero cost. Reporting only
the in-sample figure would hide that comparison entirely, which is the failure mode Bailey, Borwein,
López de Prado, and Zhu (2014) and Harvey, Liu, and Zhu (2016) document.

![Placebo distributions for the Sharpe spread and the backtest](fig:doc_placebo)

The placebos are read by direction, not by magnitude. The real point-in-time Sharpe sits at the
{{bt.placebo_ord}} percentile of {{bt.placebo_n}} run-preserving label shuffles,
{{bt.placebo_direction}} the median of the null. The max-minus-min Sharpe spread of the 60/40
portfolio across regimes sits at the {{assets.spread_ord}} percentile of {{assets.spread_n}}
shuffles, {{assets.spread_direction}} its own null median. {{bt.placebo_sentence}} Neither placebo
licenses a claim that these regimes are tradeable in this universe with this optimizer.

<!-- endif -->

## Limitations, and what would change the conclusion

The walk-forward is real-time in estimation but not in data. Each month's model sees only data up to
that month, but it sees the *current* vintage of that data, not the vintage published then. Real
revisions to industrial production and payrolls would make the real-time gaps worse, not better;
reconstructing the pipeline over ALFRED vintages is the single change most likely to move the
headline agreement numbers, and it is out of scope here.

One factor per block is a modeling choice, not a test result. A second growth factor might separate
manufacturing from services, and a second price factor goods from services inflation — either could
change quadrant assignments where the two diverge, as in 2021–22.

The optimizer is unconstrained long-short with a gross cap, kept for continuity with the earlier
look-ahead decomposition. It is the most fragile component: a long-only or risk-parity version would
almost certainly show smaller look-ahead and less dispersion between strategies. Every backtest
number here is a statement about this optimizer on this universe, not about regime conditioning in
general.

The {{params.window}}-month trend window was selected on revision statistics over a small grid, so it
carries some selection risk. The hysteresis band {{params.theta}} was carried over rather than tuned.
The false-alarm threshold that {{acc.known_failures}} fails was kept by decision rather than adjusted to
fit: anyone who thinks `Contraction` should mean "NBER recession" should read that row as a
disagreement about definitions, not a bug.

## Reproducibility

Everything on this site is produced by one command against one public input file, plus a cached ETF
price series for the asset stage. The engine is `regime_v2` in `github.com/jhcwalsh/StateAnalysis`;
`python regime_v2/run.py --vintage YYYY-MM` downloads a FRED-MD vintage, rebuilds the factors, gaps,
labels, figures and asset tables, re-runs the acceptance gate, and publishes only if the gate passes.
A pinned vintage ships with the repository so the site can rebuild without network access and so the
test suite is reproducible. The published run used `{{run.vintage}}`, labels through {{run.asof}},
label source `{{run.label_source}}`. The Acceptance and Figures tabs in the app show the same table
and images as this paper.

## References

1. Bai, J., and Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191–221.
2. Bailey, D. H., Borwein, J. M., López de Prado, M., and Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society*, 61(5), 458–471.
3. Chauvet, M., and Piger, J. (2008). A comparison of the real-time performance of business cycle dating methods. *Journal of Business & Economic Statistics*, 26(1), 42–49.
4. Croushore, D., and Stark, T. (2001). A real-time data set for macroeconomists. *Journal of Econometrics*, 105(1), 111–130.
5. Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*, 39(1), 1–38.
6. Greetham, T., and Hartnett, M. (2004). *The Investment Clock: Making Money from Macro*. Merrill Lynch Global Investment Strategy research note.
7. Grömping, U. (2006). Relative importance for linear regression in R: the package relaimpo. *Journal of Statistical Software*, 17(1), 1–27.
8. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
9. Hamilton, J. D. (2011). Calling recessions in real time. *International Journal of Forecasting*, 27(4), 1006–1026.
10. Hamilton, J. D. (2018). Why you should never use the Hodrick–Prescott filter. *Review of Economics and Statistics*, 100(5), 831–843.
11. Harvey, C. R., Liu, Y., and Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
12. Hodrick, R. J., and Prescott, E. C. (1997). Postwar U.S. business cycles: an empirical investigation. *Journal of Money, Credit and Banking*, 29(1), 1–16.
13. Kim, C.-J., and Nelson, C. R. (1999). *State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications*. MIT Press.
14. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217–1241.
15. Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
16. McCracken, M. W., and Ng, S. (2016). FRED-MD: a monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
17. National Bureau of Economic Research, Business Cycle Dating Committee (n.d.). US business cycle expansions and contractions. https://www.nber.org/research/business-cycle-dating
18. Orphanides, A., and van Norden, S. (2002). The unreliability of output-gap estimates in real time. *Review of Economics and Statistics*, 84(4), 569–583.
19. Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
20. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
21. Schmitt, O. H. (1938). A thermionic trigger. *Journal of Scientific Instruments*, 15(1), 24–26.
22. Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49–58.
23. Stock, J. H., and Watson, M. W. (2002a). Macroeconomic forecasting using diffusion indexes. *Journal of Business & Economic Statistics*, 20(2), 147–162.
24. Stock, J. H., and Watson, M. W. (2002b). Forecasting using principal components from a large number of predictors. *Journal of the American Statistical Association*, 97(460), 1167–1179.
