# States: an introduction

*The Lazy Economist · {{run.date}} · data through {{run.asof_long}}*

This site asks two questions and answers both with the same machinery. Which macro regime is
the United States economy in this month? And is knowing that worth anything to a portfolio?

As of {{current.month_long}}, the model puts the economy in **{{current.regime}}**. It is not
certain: the four probabilities are Contraction {{current.prob_Contraction}}, Goldilocks
{{current.prob_Goldilocks}}, Overheating {{current.prob_Overheating}} and Stagflation
{{current.prob_Stagflation}}. Once you strip out everything the model could not have known at the
time, what is left is measured against a plain 60/40 portfolio, before and after trading costs,
and the site sets that comparison out below rather than in a footnote.

## The four regimes

Every month the model asks two yes-or-no questions. Is growth running above or below its own
trend? Is inflation running above or below its trend? Two questions with two answers each give
four states, and those four states are the whole taxonomy:

- **Goldilocks** — growth above trend, inflation below trend.
- **Overheating** — growth above trend, inflation above trend.
- **Stagflation** — growth below trend, inflation above trend.
- **Contraction** — growth below trend, inflation below trend.

The names are shorthand for the quadrant, not claims about the business cycle. Contraction here
means below-trend growth alongside below-trend inflation. That includes recessions, and also quiet
disinflationary stretches nobody would call one.

![The four regimes as quadrants of the two gaps](fig:doc_quadrants)

The **State space** tab shows every month in the sample as a dot in that plane, the current month
circled. There is a dead band around each axis: when a gap is small, the month keeps the label it
already had. That stops the label flickering when a gap hovers a hair either side of trend.

## How a month gets its label

The raw material is FRED-MD, a monthly panel of official United States statistics maintained by
the St. Louis Fed. Two blocks are pulled from it: {{panel.n_growth}} real-activity series
(industrial production, payrolls, claims, retail sales, housing and so on) and
{{panel.n_inflation}} price series (CPI and PCE variants, producer prices, average hourly
earnings). Each series is transformed the standard way, and readings absurdly far from the rest
of their own history are dropped before anything is estimated.

From each block the model extracts one common factor — the single summary line that best
explains the co-movement of that block. That gives a growth factor and an inflation factor. Each
is then turned into a *gap*: the current reading minus a one-sided trend fitted over the trailing
{{params.window}} months and scaled by its own past variability. The trend uses only months already
past, so nothing from the future leaks in. The gap is what the four regimes are defined on.

The gaps are then fed to a hidden Markov model with four states, one per regime. Its centers are
pinned to the four quadrants, so the model cannot quietly invent its own definition of
Goldilocks. What it estimates is the noise around those centers and the persistence: how likely each regime
is to continue next month, and how likely it is to hand over to the others. That buys two things a bare
sign rule does not have: probabilities instead of a single hard call, and stickiness, which real
regimes have.

![From the FRED-MD vintage to labels, probabilities and portfolios](fig:doc_pipeline)

The last step is the one that matters most for honesty. The whole chain — outlier rules, factor
loadings, trend, scaling, transition probabilities — is re-estimated from scratch for every
month from {{sample.wf_start}} onward, using only data up to that month. The label shown for
2011 is the label the model would have produced in 2011, not a better one written afterwards. That
is the walk-forward rule, and every headline number on this site is computed on those labels.

![Regime probabilities month by month](fig:fig4_hmm_probabilities)

The **Probabilities** tab shows this history as a stacked chart. Wide bands of mixed color are
months the model found genuinely ambiguous; they are common.

## Why "available at" matters

A month's economic data is not published during that month. The FRED-MD file covering month *t*
arrives in month *t* + {{params.lag}}, so the label for month *t* is available on the
first day of the following month. Nobody could have traded on it before then.

![What is known when, and what can be traded on it](fig:doc_timing)

Every table on this site joins on that availability date, never on the data month. A return is
never paired with a label describing the same month; the backtest goes further still, so the
decision made at the end of a month uses the most recent label already published and the
resulting weights earn the month after that. Much of the gap between the flattering numbers
regime research usually reports and the ones below comes from this rule.

<!-- if:assets -->
## Does it make money?

The asset side tests the obvious strategy. Take eleven exchange-traded funds spanning equities,
bonds, real estate, gold, commodities and inflation protection. Each month, estimate how each fund
has behaved in the current regime from returns already observed, and hold the mix with the best
expected risk-adjusted return.

Done that way, the strategy earns a Sharpe ratio of {{bt.pit}}. A static 60/40 portfolio of US
equities and aggregate bonds over exactly the same window earns {{bt.sharpe_Static_6040}}.
Charging 10 basis points of trading costs per unit of turnover takes the timed portfolio down to
{{bt.sharpe10_PIT_MaxSharpe}}. It trades far more than 60/40 does, so costs bite harder.
A long-only version of the same strategy earns {{bt.lo_pit}} before costs and
{{bt.sharpe10_PIT_LongOnly_MaxSharpe}} after 10 basis points, against {{bt.sharpe_Static_6040}} and
{{bt.sharpe10_Static_6040}} for 60/40; a risk-parity version that ignores expected returns
altogether earns {{bt.sharpe_PIT_RiskParity}}.

The version of this backtest that regime papers tend to show — regime returns measured over the
whole sample, labels assigned with hindsight — scores {{bt.insample}}. That number is not
achievable and is never shown as one. Knowing the return statistics in advance accounts for
{{bt.moment_lookahead}} of that Sharpe; knowing the labels in advance accounts for
{{bt.label_lookahead}}.

![Where the in-sample Sharpe goes once look-ahead is removed](fig:doc_lookahead)

Two placebo tests ask the sharper question: would random labels have done as well? Shuffling the
regime labels while preserving how long regimes last puts the real backtest at the
{{bt.placebo_ord}} percentile of the shuffled runs, and the real spread of returns across
regimes at the {{assets.spread_ord}} percentile. {{bt.placebo_sentence}}

![Wealth curves, including the strategies that cheat](fig:fig10_backtest_wealth)

The **Portfolios** and **Backtest** tabs carry the full tables, the cost comparison and the
look-ahead decomposition.
<!-- endif -->

## What refreshes, and what stops it

FRED-MD publishes a new vintage every month; this page reflects vintage {{run.vintage_month}}.
A refresh re-downloads the vintage and the fund prices, re-runs the walk-forward, rebuilds every
figure and table, and only then replaces what the site serves.

Before it publishes, the run has to clear a set of acceptance tests — the global financial crisis
has to come out as Contraction, 2021–22 as high inflation, recession months as low growth, the
probabilities have to move, and truncating the sample by a decade must not rewrite history. On
the current run {{acc.n_passed}} of {{acc.n_tests}} pass, with known failures:
{{acc.known_failures}}. A known failure is one the site declares, explains
and reports every time rather than tuning away. Any other failure stops publication, and the
previous run stays on screen. The **Acceptance** tab lists every test with its threshold and
reason.

## Where to go next

The **Methodology** page sets out the full method, the design decisions behind it and the
sources. The **Figures** tab has every chart at full size, with the underlying labels, returns
and weights as downloadable CSV files. The code is at `github.com/jhcwalsh/StateAnalysis`; the
engine is `regime_v2`, and one command rebuilds the site from a fresh vintage.
