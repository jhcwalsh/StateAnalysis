# StateAnalysis

The States site, live at `https://states.lazyeconomist.com`: a macro regime engine that labels the US
economy Contraction, Goldilocks, Overheating or Stagflation from walk-forward filtered HMM probabilities
on FRED-MD, and shows what that labelling is worth as a portfolio-timing signal against a static 60/40.
A Streamlit dashboard plus two rendered documents (Introduction, Methodology). Deployed and refreshing
daily on the Mac mini; `status.md` is the state of play and next steps, `docs/SPEC.md` the design.

## Run and test

Python 3.12, one `requirements.txt` for engine and dashboard. README commands use the Windows venv path
`.venv/Scripts/python.exe`; on Mac/Linux it is `.venv/bin/python`.

    python -m pip install -r requirements.txt
    cd regime_v2 && python run.py data/fredmd_2026-07.csv     # ~3 min; --no-assets skips the ETF stage
    python -m streamlit run app.py                            # from the repo root
    python -m pytest tests -q                                 # dashboard and pages, ~1 min
    cd regime_v2 && python -m pytest -q                       # engine, ~4 min
    cd regime_v2 && RUN_SLOW=1 python -m pytest -q tests/test_acceptance.py   # full walk-forward acceptance

Both suites run offline: each conftest does one real engine run on the pinned vintage and blocks
yfinance. Checked 4 Oct 2026 on Python 3.12: engine 263 passed, 1 skipped (11 min on that box);
dashboard 37 passed, but only on Streamlit 1.59-1.60. `requirements.txt` allows `>=1.50,<2`: before
1.59 `AppTest` has no `download_button`, and from 1.61 `AppTest.from_file("app.py")` resolves against
the test file's folder, so 22 tests fail with FileNotFoundError. The app itself runs on 1.65 (the
suite passes there with the paths made absolute), so it is the tests, not the site. SPEC §11: after any change to `regime_v2/`, run the engine on the pinned vintage and
then its tests.

## Layout

    app.py                  the dashboard; reads regime_v2/output/ only through regime_v2.publish
    pages/                  Introduction and Methodology pages; their body is site_pages.py
    docs/site/              the documents' Markdown; CONTRACT.md lists every {{placeholder}} and fig: marker
    site_theme.py           lazyeconomist.com palette, CSS, masthead, page nav
    regime_v2/run.py        engine driver: download/load a vintage, run, acceptance gate, publish
    regime_v2/regime_v2/    engine modules (data, factors, trend, regimes, walkforward, assets, portfolio,
                            placebo, rtvintage, acceptance, publish, sitedocs, figures)
    regime_v2/data/         pinned vintage fredmd_2026-07.csv and returns_fixture.parquet (the only tracked data)
    scripts/                refresh_states.sh (the Mini's daily job), build_vintage_archive.py
    deploy/                 the LaunchAgent plist for the daily refresh
    docker/entrypoint.sh    publishes the pinned vintage on first start, then serves Streamlit
    docs/SPEC.md            source of truth for the engine; §10 decision log, §12 deployment
    docs/superpowers/       design specs and implementation plans
    tests/                  dashboard, page and refresh-script tests; engine tests are in regime_v2/tests/

## Conventions and gotchas

- No look-ahead in the labelling path. A full-sample quantity goes in a function ending `_expost` and is
  only for comparison figures; HMM smoothing counts as look-ahead. Every estimation step takes the
  estimation mask (SPEC §11).
- Regime names only, never `Q1..Q4`, repo-wide. Grep before committing.
- Nothing Windows-specific in code: it must run unchanged on the Mini. Windows paths only in README setup.
- Numbers are never typed into the README or documents: `docs/site/*.md` use `{{placeholders}}` filled from
  the published `summary.json`. Unknown keys render `[missing: key]`, which the tests catch.
- Publishing is atomic: `publish()` swaps `output/` by rename, and a run that fails an acceptance check
  exits non-zero and leaves the last good output. `last_check.json` sits beside `output/`, not in it.
- A vintage `YYYY-MM` ends the previous month and is posted during the following one, so the newest state is
  usually two months back. That is the data, not a broken schedule.
- `run.py` exit codes: 0 published, 1 acceptance failed, 3 nothing newer, 4 no vintage downloaded.
- Generated and not in git: `regime_v2/output*`, `regime_v2/figs*`, downloaded vintages, the ~305 MB
  vintage archive (`scripts/build_vintage_archive.py --out DIR`, passed as `--vintage-archive DIR`).
- Paths are overridable by env: `REGIME_OUTPUT_DIR`, `REGIME_FIGS_DIR`, `REGIME_RETURNS_CACHE`,
  `REGIME_VINTAGE_ARCHIVE` (the Dockerfile points them at the `/app/var` volume). No API keys needed.
- SPEC §11 asks commit messages to name the stage and task (`stage4: walk-forward HMM refit`); recent
  history uses a prefix such as `docs:`.
- The tests' dashboard fixture mirrors `regime_v2/tests/conftest.py`; keep the two in step.

## Deploy

Docker on the Mac mini in `~/apps/states`, port **8505** bound to `127.0.0.1` (set in `docker-compose.yml`),
named volume `states_var` holding published output, figures, return cache and the
vintage archive. Cloudflare tunnel route `states.lazyeconomist.com` -> `localhost:8505`. Ship with
`deploy states` (git pull + `docker compose up -d --build`); logs with `docker logs -f states`. The daily
refresh is the LaunchAgent in `deploy/` running `scripts/refresh_states.sh` at 07:00 local; it pushes to
an ntfy topic named in the untracked `~/apps/states/.refresh.env` (`NTFY_TOPIC`). The dashboard's
Refresh button is the manual fallback. Full checklist in SPEC §12. Nothing in the repo deploys on push;
the documented path is `deploy states`. An unpinned rebuild picks up the newest Streamlit under 2.

<!-- BEGIN shared: James's house rules. Identical in every repo; change it everywhere, not here alone. -->
## How James works

This section and the two after it are the same in all of James's repos. **Where this repo's
own rules above are stricter or more specific - about branches, pushing, deploying, editing,
or spending on model calls - this repo's rules win.**

**Talking to James**
- Plain English, and short. Lead with the outcome. If something failed or was not checked, say that first.
- Don't take yourself too seriously, and make suggestions: if there is a better way or a next step worth doing, say so in a line.
- Numbers go in a small table, parallel items in a list. No walls of text.

**Just do it, or ask first**
- Just do anything reversible that the request implies: reading, editing, running tests, committing.
- Ask first before deploying; touching live data, money, or other people's details; deleting
  files or branches you did not create; adding a heavy dependency; or widening the job beyond
  what was asked.
- If a request is ambiguous and the readings lead to different work, ask one short question.
  Otherwise pick the sensible reading, say which one, and carry on.

**Done means verified**
- Run the project's tests before saying done, and say what passed and what did not. Never
  skip, xfail or delete a failing test to get green; report it.
- For anything with a UI, see it working in a browser, not just building.
- Regenerate generated files rather than hand-editing them.
- Secrets stay out of git: `.env` is local, `.env.example` shows its shape.

## Git, PRs and deploys

- Commit message: a plain sentence saying what changed, then a short body saying why. One topic per commit.
- The default branch is `main` in some repos and `master` in others; check before assuming.
- Local sessions on James's machine: committing to the default branch is normal for work he
  asked for, unless this repo's rules say otherwise. Push only when he says so.
- Cloud sessions: work on the session's `claude/...` branch, open a draft PR, never push to the
  default branch.
- **In some repos a push to the default branch IS a deploy** (the Mini polls and rebuilds). Read
  this repo's Deploy section before any push to it.
- Apps run on the Mac mini (`JHCW-mini.local`) behind a Cloudflare tunnel. Where an app is
  deployed by hand it is `deploy <app>` from James's PowerShell. Never deploy unless asked. A
  cloud session cannot reach the Mini: say so and give him the command.
- One host port per app on the Mini: 8501 test, 8502 terrarium, 8503 regimes, 8504 glidepath,
  8505 states, 8506 brief, 8507 lampmold. A new app takes the next free port, checked on the box.
  `MacMiniHosting/mac-mini-hosting-runbook2.md` on James's machine is the authority for the Mini.

## Session notes

Keep SESSION_NOTES.md in the repo root up to date.

**When:** Before every git commit, update it and include it in
the same commit. Also update it when I say "wrap up".

**What:** Newest entry at the top, dated. Cover:
- Done: what changed, and which files
- Why: key decisions and reasoning
- Broken / unsure: known bugs, hacks, untested bits
- Next: the 1–3 most sensible next steps

**How much:** A new entry for meaningful work, or a one-line
append to today's entry for small fixes. Each entry under ~20
lines, plain English, no code dumps.

**Housekeeping:** Keep the last 10 entries in full and fold
older ones into a short "History" section at the bottom.

**Enforced:** `.claude/hooks/require-session-notes.sh` runs before every Bash command
(`.claude/settings.json`) and refuses a commit in this repo unless SESSION_NOTES.md is staged,
or staged by the same command. Merge commits pass, and so does a command that only mentions a
commit. If it blocks you, write the entry; do not work around the hook.
<!-- END shared -->
