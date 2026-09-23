# States — status

*2026-09-23. Live at https://states.lazyeconomist.com; source github.com/jhcwalsh/StateAnalysis.*

## Done

- **Two documents on the States app.** An Introduction and a Methodology paper with references,
  rendered as pages from Markdown (`docs/site/`) with every number substituted live from the
  published run and seven engine-drawn figures. Each page downloads as a self-contained HTML that
  prints to PDF. Two independent reviews caught real errors before merge (full-sample quantities
  labelled as walk-forward, typed placebo directions, mangled math in the export).
- **Landing page.** The States card links to both pages, and its blurb states the comparison
  rather than a verdict.
- **Scheduled refresh with alerts.** A LaunchAgent on the Mini runs `scripts/refresh_states.sh`;
  any failure pushes to a private ntfy topic with the log tail. Tested end to end, and the test
  exposed an engine bug: the Fed's site returns an HTML page for a missing vintage, which now
  fails cleanly instead of as a parse error. Superseded in part by the daily job below.
- **Long-only and risk-parity strategies.** Added with long-only oracle and in-sample comparators,
  so the look-ahead decomposition is measured under the constraint. Result on the live vintage:
  long-only point-in-time Sharpe 1.05 equals static 60/40 before costs (1.00 vs 1.05 at 10 bp),
  with moment look-ahead 0.34 and label look-ahead near zero; risk parity 0.93. The abstract,
  introduction, README and card were rewritten to present that comparison from placeholders
  instead of asserting "no edge".

- **Opening screen rebuilt around the latest assessment.** The θ explorer is now the first Results
  tab and the walk-forward probability chart took its place, over a trace of the numbers the
  published label was read from: each gap against the ±θ trigger and the side a held sign carries
  forward, the factor levels behind the gaps, the vintage and publication lag, this month's and
  last month's filtered probabilities and the persistence of the current state. It states that the
  gaps shown are the published full-sample-loading series while the labels come from the
  unpublished walk-forward gaps, and says so explicitly when the two disagree.
- **The masthead link back to lazyeconomist.com now works.** It had never been clickable on any
  page: Streamlit's transparent header is a full-width bar over the top 60px and swallowed the
  click. The header and its toolbar pass pointer events through now, and the anchor is `_self`
  because Streamlit rewrites a target-less anchor to `_blank`.

- **The refresh now asks FRED-MD what exists, daily, instead of guessing from the calendar.**
  Chasing "why is there no August state" established that nothing had failed: 2026-08 is the
  newest vintage published and it ends 2026-07. Two rules came out of the probing — a vintage
  `YYYY-MM` ends the *previous* month, and it is not posted during its own month (2026-09 was
  still absent on 13 Sep 2026; archived FRED-MD pages from 17 Oct 2025, 16 Jan 2026 and 6 Jun 2026
  each list the previous month as newest). So the newest state available in any month is two months
  back, and the release day drifts. The old rule (07:00 on the 10th, `date -v-1m`) re-ran an
  already-published vintage in a good month and would miss a month whenever the vintage landed
  after the 10th. `run.py --vintage latest --if-newer` now probes back from the current month for
  the first vintage that downloads and exits 3 in about two seconds unless it is strictly newer;
  4 means nothing downloaded at all. The script routes those: a published month pushes a note
  naming vintage, month and state; nothing new is logged only; an unreachable source pushes after
  three consecutive days and then weekly. `tests/test_refresh_script.py` drives that routing
  against stub docker and curl rather than grepping the script.
- **A freshness line under the Latest assessment.** A daily job publishes on one day in thirty, so
  the run timestamp alone made a working site read as abandoned. Every resolved run now writes
  `last_check.json` beside `output/` — beside, because `publish()` swaps that directory by rename
  — and the page shows when the engine last ran, when the source was last asked, and the clause
  that answers the original question: which vintage is newest, what month the run reads through,
  and that this is what sets the month shown, not the schedule.
- **The engine takes the Fed's revised vintage file.** `FREDMD_URLS` tries `YYYY-rev-MM-md.csv`
  first, which is what the site's own `current.csv` link points at; the plain `YYYY-MM-md.csv` is
  the file that shipped the malformed 2026-08 header. The two differ only in the header and t-code
  rows, so no published number changes and `load_fredmd`'s repair stays as the fallback.

- **Long-only backtest placebo (2026-09-22, deployed and published 2026-09-23).** The placebo now scores
  `PIT_LongOnly_MaxSharpe` on the same 200 shuffles as the unconstrained strategy (paired nulls; the
  unconstrained null is unchanged draw for draw, 25.5th percentile as published). Local run on vintage
  2026-08: real long-only 1.05 at the 86.5th percentile, shuffled-label median 0.87, 60/40 1.05; 14% of
  shuffles reach 1.05. Documents state that comparison from placeholders inside an `if:lo_placebo`
  guard, so older and `--skip-placebo` runs drop the paragraph. Adds about five minutes to a refresh.
- **Real-time vintages scoped.** FRED-MD publishes every vintage 1999-08→2026-08 in two zips
  (1999–2014 reconstructed from Haver archives), so a real-time walk-forward is possible from 1999-07,
  far cheaper than ALFRED. Needs a D8 spec change plus rename/missing-series handling in the loader.

Everything above is merged, deployed and live on the Mini (the live run reproduces the local placebo numbers exactly); engine suite 201 passed 1 skipped, root
suite 29. The daily LaunchAgent was reinstalled and kickstarted as a test: it resolved 2026-08 off
the live site, exited "nothing to do" in five seconds and wrote the heartbeat, with no push.

## Next steps, in order

1. **Real-time vintage label from 1999-07.** In design: proposed as a comparator column
   (`hmm_realtime_vintage`) beside the unchanged published label, with a backtest on it; D8 amended,
   not overturned. Awaiting approval of that role before the design sections and spec.
2. **Small items:** doc-figure regeneration in the container entrypoint; figure restyle.

## Decided

- **Non-NBER Contraction stays a documented known failure** (2026-09-23). Contraction is a sign
  pair, not a recession call; changing the model to hit an NBER target would tune to the test. The
  0.10 threshold and its known-failure note stand as decided in spec §9 Q8.
- **The `dreamy-pare` worktree is gone** (2026-09-23). Its April notebook edits, exports and an
  untracked `pe_optimiser.py` are archived on the local branch `claude/dreamy-pare` (`1d955ae`), not merged.

## Watch for

The 2026-09 vintage — the first with an August observation — was still absent on 22 Sep and should appear during October on the
pattern above. The daily job will publish it within a day and push a note naming the month and
state; that push is also the first live test of the success path, which so far has only been
exercised by its tests.
