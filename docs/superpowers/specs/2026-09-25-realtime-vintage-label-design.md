# Real-time-vintage label — design

*2026-09-25. Approved in conversation (option A: a comparator beside the published label). Amends
spec D8; the published label, the current-state banner and every existing number are unchanged.*

## Purpose

The walk-forward label (D8) refits each month on data up to that month but reads the *final*
vintage, so it ignores data revisions. The Methodology paper names this as its biggest limitation.
FRED-MD publishes every monthly vintage from 1999-08, so the limitation can be measured instead
of stated: label each month *t* from the vintage a reader actually had, and report how far that
label, its gaps, its NBER calls and its backtest sit from the final-vintage walk-forward.

Success: a `hmm_rt_vintage` label for every month from 1999-07 whose vintage is usable, published
beside `hmm_walkforward` with the comparison metrics below, reported with no verdict.

Out of scope: months before 1999-07 (no vintage exists); replacing the primary label (option B);
ALFRED; changing any model parameter.

## Evidence this rests on (spike, 2026-09-25)

A throwaway run of the unchanged pipeline on vintage *t+1* cut at *t* covered 324 months in 9 s
and reproduced the published 2026-07 probabilities to four decimals from vintage 2026-08. Against
the final-vintage walk-forward: label agreement 0.787 (0.784 on 1999–2014 vintages, 0.791 on
2015+); gap correlation 0.97 growth, 0.95 inflation; identical first low-growth calls for the 2001,
2008 and 2020 peaks; backtest Sharpe 0.77 → 1.18 unconstrained and 1.05 → 0.84 long-only (60/40
1.05) with 45 of 199 months labelled differently. These numbers motivate the design; the engine
recomputes them on every run and they are not hard-coded anywhere.

## 1. The vintage archive

- `scripts/build_vintage_archive.py` (one-off, idempotent) downloads from
  `https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/`:
  `historical_fred-md.zip` (vintages 1999-08..2014-12), `historical-vintages-of-fred-md-2015-01-to-2025-12.zip`
  (2015-01..2025-12, three filename styles: `YYYY-MM.csv`, `FRED-MD-YYYYmMM.csv`,
  `FRED-MD_YYYYmMM.csv`), and `monthly/{YYYY}-rev-{MM}-md.csv` falling back to `monthly/{YYYY-MM}-md.csv`
  for every month from 2026-01 through the newest (the engine's `fredmd_urls`, so the revised file is
  preferred — the plain 2026-08 file has the shifted header).
- Output: `<archive>/fredmd_YYYY-MM.csv`, the file bytes unchanged. Default `<archive>` is
  `regime_v2/data/vintages/` (gitignored, ~305 MB); on the Mini `/app/var/vintages` on the existing
  `states_var` volume. The script skips files already present and prints a coverage summary
  (first, last, missing months).
- Provenance is derived, not stored: vintages ≤ 2014-12 are `reconstructed` (rebuilt by the St. Louis
  Fed from archived Haver data), later ones `published`.
- The daily refresh keeps the archive current: when `run.py` downloads a vintage and an archive
  directory is configured, it copies the downloaded file to `<archive>/fredmd_YYYY-MM.csv` if absent.

## 2. Loading an archive vintage — `data.load_archive_vintage`

A new function beside `load_fredmd`; the live refresh path is untouched.

```
load_archive_vintage(path, neighbour=None) -> (levels, tcodes, report)
```

- **Renames** (applied to the header): `PPIFGS → WPSFD49207`, `PPIFCG → WPSFD49502`.
- **Not substituted:** `CUUR0000SA0L2` (the not-seasonally-adjusted CPI stand-in in the 2015-01..2017-03
  vintages) is dropped; the seasonally adjusted `CUSR0000SA0L2` is then missing for those vintages.
- **Missing series** are allowed; `build_blocks` already drops absent block series.
- **Per-series check against the neighbouring vintage** (the previous archive vintage; the next one for
  the first): `check_vintage` with the existing 0.98 level-correlation rule on the post-1990 overlap,
  after renames. A failing series is dropped for that vintage and recorded; the vintage is kept.
  The t-code check compares against the neighbour the same way (a changed t-code drops the series).
- **Refused** (`VintageError`, recorded, the month gets no label) only when an anchor (`INDPRO`,
  `CPIAUCSL`) is missing or fails, or when either block keeps fewer than half its series.
- `report` records `renamed`, `dropped_check`, `dropped_tcode`, `missing`, `refused` per vintage.
- `run_pipeline` / `build_blocks` gain a `loader` argument (default `load_fredmd`) so an archive run
  passes `load_archive_vintage`; nothing else in the pipeline changes.

## 3. The real-time-vintage walk-forward — `walkforward.fit_hmm4_rt_vintage`

```
fit_hmm4_rt_vintage(archive_dir, start="1999-07", end=None, **kw) -> RtVintageResult
```

- For each month *t* from `start`: the vintage is *t*+1 (a vintage `YYYY-MM` ends the previous
  month, matching D11's one-month lag); run the unchanged pipeline on it with `asof` = end of *t*,
  take the filtered probabilities, label and gaps at *t*.
- **Gap months:** a vintage that does not contain *t* (the 2025-11 vintage has no 2025-10 row after
  the shutdown), a missing vintage file, or a refused vintage gives no label for *t*; the month is
  listed in `gaps` with its reason. Nothing is carried forward in the published column.
- Result: `labels` (`hmm_rt_vintage`), `probs` (four columns), `growth_gap`, `inflation_gap`,
  `gaps` (month → reason), `reports` (vintage → loader report), `provenance` (month → reconstructed/published).
- No caching: the full run is about 324 pipeline fits (≈1 min single-threaded).

## 4. What is published

**`regime_labels.csv`** gains `hmm_rt_vintage`, `p_rt_<Regime>` (four), `growth_gap_rt_vintage`,
`inflation_gap_rt_vintage`; empty before 1999-07 and at gap months.

**`summary.json["rt_vintage"]`** (absent or `{"skipped": reason}` when no archive is configured or the
stage fails — a failure never changes the engine's exit code, like the asset stage):

- `window` (first, last, n_months), `gaps` (month → reason), `n_reconstructed`, `n_published`;
- `agreement` with `hmm_walkforward`: overall, reconstructed, published; `crosstab` (final rows × real-time columns);
- `shares` and `mean_run_length` per regime under both labels; `switches` under both;
- `gap_revision` per axis: mean and sd of (real-time − final), correlation, sign agreement;
- `nber_lags`: `figures.nber_lags` under both labels for peaks inside the window;
- `loader`: counts of vintages with each rename, drop and refusal, and the series most often dropped.

**Asset stage** (inside the existing `run_assets`, same all-or-nothing staging): `PIT_MaxSharpe` and
`PIT_LongOnly_MaxSharpe` rerun with `hmm_rt_vintage` (and its probabilities) in place of the
walk-forward label, at 0 and 10 bp, published as `summary["assets"]["rt_vintage_backtest"]` with its
counters. For the backtest input only, a gap month holds the previous month's label (what a reader
would still have held); the count of held months is reported.

**Acceptance:** report-only rows `rt_vintage_agreement`, `rt_vintage_agreement_reconstructed`,
`rt_vintage_agreement_published`, `rt_growth_gap_corr`, `rt_inflation_gap_corr`,
`rt_pit_sharpe`, `rt_pit_longonly_sharpe`, each with a rationale naming its comparator. No thresholds.

**Figure** `fig12_rt_vintage`: two label strips 1999-07→ (final-vintage walk-forward above, real-time
vintage below), disagreement months ticked between them, the reconstructed span shaded and labelled,
NBER recessions shaded as in fig 7.

## 5. Command line and operations

- `run.py --vintage-archive DIR` (default: unset → stage skipped with reason "no vintage archive
  configured"). `scripts/refresh_states.sh` passes `/app/var/vintages`.
- A one-off on the Mini after deploy: `docker exec states python scripts/build_vintage_archive.py
  --out /app/var/vintages`, then a manual refresh so the site shows the block.

## 6. Documents and dashboard

- Placeholders (listed in `docs/site/CONTRACT.md`): `rt.window_start`, `rt.n_months`,
  `rt.agreement`, `rt.agreement_reconstructed`, `rt.agreement_published`, `rt.growth_corr`,
  `rt.inflation_corr`, `rt.nber_sentence` (one sentence comparing the calls, worded for match or
  mismatch), `rt.pit`, `rt.pit_longonly`, `rt.n_gaps`, and `skipped.rt_vintage` driving an
  `<!-- if:rt_vintage -->…<!-- endif:rt_vintage -->` guard (its own closing tag, like `if:lo_placebo`).
  Missing block → every key `n/a` and the guarded text dropped.
- Methodology § Limitations: the paragraph asserting that revisions "would make the real-time gaps
  worse" and that ALFRED reconstruction is out of scope is replaced by the measurement, from
  placeholders, stating that 1999–2014 vintages are reconstructed. The Introduction gains one
  sentence. Both place each backtest number beside 60/40 and state no verdict.
- Dashboard: fig 12 and a one-paragraph caption in the walk-forward results area, shown only when
  the block exists.
- `docs/SPEC.md`: D8 amended by a decision-log entry (the published label stays the final-vintage
  walk-forward; the real-time-vintage label is a reported comparator from 1999-07); §5/§6 get the new
  function contracts; §7 lists fig 12.

## 7. Testing

On a synthetic archive built in `tmp_path` from the pinned vintage (truncated to successive end
months), no network:

- month *t* reads only vintage *t*+1: a revision planted in vintage *t*+2 and later leaves *t*'s
  probabilities unchanged, and one planted in *t*+1 changes them;
- the last month from the pinned-equivalent vintage equals `run_pipeline(pinned, asof=t)` exactly;
- renames applied; `CUUR0000SA0L2` dropped; a series redefined in one vintage is dropped there and
  recorded, the vintage kept; an anchor failure and a half-empty block refuse the vintage;
- a vintage without its final month, and a missing file, become `gaps` with reasons;
- `run.py` without `--vintage-archive` publishes `{"skipped": ...}` and exits as before; a stage
  failure does not change the exit code; the archive copy-on-download writes once;
- sitedocs: every new placeholder is in the contract, `n/a` without the block, guarded text dropped;
  docfigs/app render without the block.

## Risks

- The 1999–2014 vintages are reconstructions, not what was on screens then; every surface says
  "reconstructed" for them.
- Block composition changes across vintages (e.g. HWI absent before 2015), so the factors are not
  defined on an identical series set; the loader report makes this visible.
- The Fed's archive layout (zip names, filename styles) can change; the build script fails loudly
  with the URL and a coverage summary rather than producing a partial archive silently.
