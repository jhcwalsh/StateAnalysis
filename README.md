# StateAnalysis — Macro Regime Engine

The site classifies the US macro economy into four regimes — Contraction, Goldilocks, Overheating,
Stagflation — from walk-forward filtered HMM probabilities published with a one-month lag, and shows what
that labeling is actually worth as a portfolio-timing signal. Every regime-timing number is shown beside a
static 60/40 benchmark over the same window, before and after 10 bp of trading costs, so the comparison is
unavoidable rather than buried in a caption. The in-sample Sharpe is decomposed into the part that comes
from knowing the return moments in advance and the part that comes from knowing the labels, and both an
unconstrained long-short optimiser and a long-only one are carried through that decomposition; two
run-preserving placebo tests put the real labels against random relabelings. The live figures are in the
documents the engine renders (`docs/site/introduction.md`, `docs/site/methodology.md`) and on the site
itself, never typed into this README.

## Run locally

    .venv/Scripts/python.exe -m pip install -r requirements.txt
    cd regime_v2 && ../.venv/Scripts/python.exe run.py data/fredmd_2026-07.csv   # ~3 min; add --no-assets to skip the ETF stage
    .venv/Scripts/python.exe -m streamlit run app.py                            # from the repo root

## Refresh

The dashboard has a Refresh button at the bottom of the page. It reruns the engine for a given vintage — the
input defaults to the previous month — and republishes `regime_v2/output/`. If the run fails any acceptance
check, the last published outputs are kept and the dashboard keeps serving them.

On the Mini the same refresh runs on a schedule: `scripts/refresh_states.sh`, launched by the
LaunchAgent in `deploy/com.lazyeconomist.states.refresh.plist` at 07:00 local **every day**. It runs
`run.py --vintage latest --if-newer`, which probes backwards from the current month for the newest
FRED-MD vintage that actually downloads and does nothing unless it is newer than the published one.
Daily rather than monthly because FRED-MD is not on a fixed release day and does not post a vintage
during its own month, so no calendar rule can name the right vintage in advance — the job asks
instead, and a day with nothing new costs about two seconds.

What each outcome does: a new month published pushes a note naming the vintage, month and state; no
new vintage is logged and never pushed; a failed run (vintage rejected, acceptance gate failed,
container down) pushes an alert with the tail of `~/apps/states/logs/refresh.log`. A source that is
unreachable — no vintage downloads at all — pushes only after three consecutive days and then
weekly, so an outage does not alert every morning. The topic is named in `~/apps/states/.refresh.env`
(`NTFY_TOPIC=...`, untracked). `VINTAGE=2026-08 scripts/refresh_states.sh` pins one vintage by hand
and skips the newer-than check; the Refresh button remains the manual fallback.

Every check records itself in `last_check.json` beside `output/` (not inside it — `publish()` swaps
that directory by rename), and the dashboard prints both dates under the Latest assessment: when the
engine last ran and when the source was last asked. The two differ by design, since most days check
and publish nothing.

Note on timing: a vintage named `YYYY-MM` ends the *previous* month and is posted during the
*following* one, so the newest state available in any month is usually two months back. That is the
data, not the schedule — the job publishes each vintage within a day of it appearing.

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
