"""End-to-end driver (spec §5 outputs, §8 acceptance, §12 staged publish).

Usage:
  python run.py data/fredmd_2026-07.csv
  python run.py --vintage 2026-08            # downloads data/fredmd_2026-08.csv first
  python run.py --vintage latest --if-newer  # the scheduled job: publish only a vintage not yet published
Options: --trend-window 240 --theta 0.5 --no-walkforward --wf-step 1 --wf-min-obs 240
         --skip-robustness --skip-expanding --out-dir output --figs-dir figs --data-sheet data/README.md
Exit codes: 0 published; 1 no publish (a §8 threshold failed, staged results stay in <out-dir>.staging/);
3 nothing to do (--if-newer, the newest vintage is already published); 4 no vintage could be downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from regime_v2 import acceptance, assets, docfigs, figures, portfolio, regimes as R, rtvintage
from regime_v2.data import GROWTH_BLOCK, INFLATION_BLOCK
from regime_v2.factors import pca_factor_expanding
from regime_v2.data import VintageError
from regime_v2.pipeline import DEFAULTS, labels_frame, run_pipeline
from regime_v2.publish import load_published, write_last_check
from regime_v2.trend import centred_trend_expost, revision_stats
from regime_v2.walkforward import fit_hmm4_walkforward

HERE = Path(__file__).resolve().parent
# Spec §6 Stage 1: verified 2026-09-04 against the St. Louis Fed FRED-MD page —
# monthly vintages are named YYYY-MM-md.csv under /-/media/...; the older
# files.stlouisfed.org pattern is kept as a fallback. The revised file is tried
# first (added 2026-09-13): the Fed's own current.csv and YYYY-MM.csv links both
# point at YYYY-rev-MM-md.csv, while the plain YYYY-MM-md.csv is the file that
# shipped the malformed 2026-08 header (the dropped `S&P div yield` name). The
# two files' data rows are byte-identical, so preferring the revised one only
# spares load_fredmd the repair, which stays as the fallback.
FREDMD_URLS = [
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{year}-rev-{month}-md.csv",
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{vintage}-md.csv",
    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{vintage}.csv",
]
FREDMD_URL = FREDMD_URLS[1]   # kept for callers/tests that format a single {vintage} pattern
FIG_NAMES = ["fig1_factors_gaps", "fig2_regime_timeline", "fig3_state_space", "fig4_hmm_probabilities",
             "fig5_revisions", "fig6_classifier_comparison", "fig7_walkforward", "fig12_rt_vintage"]
# Everything the asset stage publishes, so a run can clear the previous run's asset
# artefacts before it starts and only put its own in place once the stage succeeded.
ASSET_CSVS = ["regime_returns.csv", "backtest_returns.csv", "portfolio_weights.csv"]
ASSET_CSV_GLOBS = ["regime_corr_*.csv"]
ASSET_FIGS = ["fig8_regime_returns", "fig9_mixture_6040", "fig10_backtest_wealth", "fig11_pit_weights"]
ASSETS_STAGING = ".assets_staging"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fredmd_urls(vintage: str) -> list[str]:
    """Every known download location for one vintage, in the order they are tried."""
    year, month = vintage.split("-")
    return [t.format(vintage=vintage, year=year, month=month) for t in FREDMD_URLS]


def looks_like_fredmd(data: bytes) -> bool:
    """A FRED-MD vintage starts with the `sasdate` header row. The St. Louis Fed site answers a
    request for a vintage that does not exist yet with an HTML page and status 200, which would
    otherwise be saved and fail much later as a CSV parse error."""
    return data.lstrip()[:7].lower() == b"sasdate"


def try_download(vintage: str, dest_dir: Path, fetch=urllib.request.urlopen) -> tuple[Path | None, list[str]]:
    """Download one FRED-MD vintage, trying each known URL in turn. Returns (path, errors); the
    path is None when no URL yielded a FRED-MD file, which is the normal answer for a month the
    Fed has not posted yet. A response that is not a FRED-MD file counts as a failure for that
    URL; nothing is written until one checks out."""
    dest_dir = Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"fredmd_{vintage}.csv"
    errors = []
    for url in fredmd_urls(vintage):
        try:
            with fetch(url, timeout=60) as r:
                data = r.read()
        except Exception as e:  # urllib.error.HTTPError / URLError / OSError
            errors.append(f"{url}: {e}")
            continue
        if not looks_like_fredmd(data):
            errors.append(f"{url}: response is not a FRED-MD file (no `sasdate` header; "
                          f"{len(data)} bytes) — the vintage is probably not published yet")
            continue
        dest.write_bytes(data)
        return dest, errors
    return None, errors


def download_vintage(vintage: str, dest_dir: Path, fetch=urllib.request.urlopen) -> Path:
    """`try_download` for a vintage the caller named explicitly: not finding it is fatal."""
    dest, errors = try_download(vintage, dest_dir, fetch=fetch)
    if dest is None:
        raise SystemExit("FRED-MD download failed for vintage " + vintage + ":\n  " + "\n  ".join(errors)
                         + "\nVerify the download location on the St. Louis Fed FRED-MD page (spec §6 Stage 1).")
    return dest


def latest_vintage(today: date, dest_dir: Path, fetch=urllib.request.urlopen,
                   max_back: int = 3) -> tuple[str | None, Path | None]:
    """The newest FRED-MD vintage that actually downloads, searching back from `today`'s month.

    FRED-MD does not post a vintage during its own month and its release day within the following
    month drifts, so the schedule cannot name the vintage in advance: it has to ask. Verified
    2026-09-13 — 2026-09 answers with the site's HTML 404 while 2026-08 is a real file — and in
    archived snapshots of the FRED-MD page on 2025-10-17, 2026-01-16 and 2026-06-06, where the
    newest vintage listed was in each case the previous month's.

    Returns (None, None) when nothing in the window downloads, which means the source is
    unreachable rather than merely unchanged: the vintage already published lives in that window.
    """
    y, m = today.year, today.month
    for _ in range(max_back + 1):
        vintage = f"{y:04d}-{m:02d}"
        dest, _errors = try_download(vintage, dest_dir, fetch=fetch)
        if dest is not None:
            return vintage, dest
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return None, None


def published_vintage(out_dir: Path) -> str | None:
    """The vintage of the run currently published in `out_dir` ("2026-08"), or None if there is
    none yet or it cannot be read — a fresh volume must not be mistaken for an up-to-date one."""
    try:
        name = json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))["run"]["vintage"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    m = re.fullmatch(r"fredmd_(\d{4}-\d{2})\.csv", str(name))
    return m.group(1) if m else None


def write_data_sheet(blocks: dict, path: Path) -> None:
    cells = blocks["outliers"].groupby("series").size()
    rows = ["# FRED-MD data sheet (generated by run.py)", "",
            "| series | block | t-code | first | last | outlier cells removed |", "|---|---|---|---|---|---|"]
    for block_name, cols in [("growth", GROWTH_BLOCK), ("inflation", INFLATION_BLOCK)]:
        df = blocks[block_name]
        for c in cols:
            if c not in df:
                continue
            s = df[c].dropna()
            rows.append(f"| {c} | {block_name} | {int(blocks['tcodes'][c])} | {s.index[0]:%Y-%m} | {s.index[-1]:%Y-%m} | {int(cells.get(c, 0))} |")
    rows += ["", f"Missing from vintage: {blocks['missing_series'] or 'none'}",
             f"Estimation mask excludes {int((~blocks['estimation_mask']).sum())} months."]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def robustness_table(path: str, **kw) -> dict:
    """Spec §6 Stage 2: labels under alternative trends; agreement matrix, revision stats, GFC share."""
    variants = {"120_mean": dict(window=120), "180_mean": dict(window=180), "240_mean": dict(window=240),
                "120_median": dict(window=120, method="trailing_median"),
                "180_median": dict(window=180, method="trailing_median"),
                "hamilton": dict(method="hamilton")}
    runs = {k: run_pipeline(path, **{**kw, **v}) for k, v in variants.items()}
    names = list(runs)
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            la, lb = runs[a].hmm.labels_filtered, runs[b].hmm.labels_filtered
            idx = la.index.intersection(lb.index)
            agree.loc[a, b] = float((la.loc[idx] == lb.loc[idx]).mean())
    per = {}
    for k, r in runs.items():
        expost = centred_trend_expost(r.growth_factor["factor"], smooth=r.params["smooth"], window=r.params["window"])
        per[k] = {**revision_stats(r.G, expost),
                  "gfc_contraction_hmm": acceptance.share(r.hmm.labels_filtered, "2008-09", "2009-06", ["Contraction"]),
                  "counts_2024_26": r.hmm.labels_filtered.loc["2024-01":"2026-12"].value_counts().to_dict(),
                  "mean_growth_gap_2024_26": float(r.G.loc["2024-01":"2026-12"].mean())}
    return {"label_agreement": agree.round(3).to_dict(orient="index"), "per_variant": per}


def build_summary(res, free, gmm, wf, hist_labels, hist_probs, label_source, table, rev, lags, extra) -> dict:
    hmm = res.hmm
    return {
        "n_months": int(len(hist_labels)),
        "sample": [str(hist_labels.index[0].date()), str(hist_labels.index[-1].date())],
        "label_source": label_source,
        "quadrant_source": "walk-forward gaps" if wf is not None else "full-sample",
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in res.params.items()},
        "regime_counts": hist_labels.value_counts().to_dict(),
        "regime_counts_quadrants": res.quadrant.value_counts().to_dict(),
        "agreement_with_quadrants": float((hist_labels == res.quadrant.reindex(hist_labels.index)).mean()),
        "emission_only_agreement": float((hmm.emission_labels == res.quadrant0).mean()),
        "filtered_vs_smoothed_agreement": float((hmm.labels_filtered == hmm.labels_smoothed_expost).mean()),
        "share_max_prob_gt_095": {"primary": float((hist_probs.max(axis=1) > 0.95).mean()),
                                  "smoothed_expost": float((hmm.probs_smoothed_expost.max(axis=1) > 0.95).mean())},
        "expected_duration_months": R.expected_duration(hmm.transmat).round(2).to_dict(),
        "mean_run_length_months": R.run_lengths(hist_labels).round(1).to_dict(),
        "transition_matrix": hmm.transmat.round(4).to_dict(orient="index"),
        "min_transition_prob": float(hmm.transmat.to_numpy().min()),
        "emission_means": {r: [round(float(x), 3) for x in hmm.means[i]] for i, r in enumerate(R.REGIMES)},
        "free_hmm": {"state_names": list(free.state_map.values()),
                     "quadrant_profile": free.quadrant_profile.round(3).to_dict(orient="index"),
                     "agreement_marginalised_vs_quadrants": float((free.quadrant_probs_filtered.idxmax(axis=1) == res.quadrant0).mean())},
        "gmm": {"cluster_names": gmm.cluster_names,
                "quadrant_profile": gmm.quadrant_profile.round(3).to_dict(orient="index"),
                "agreement_marginalised_vs_quadrants": float((gmm.quadrant_probs.idxmax(axis=1) == res.quadrant0).mean())},
        "stagflation_1973_75": acceptance.share(hist_labels, "1973-11", "1975-03", ["Stagflation"]),
        "stagflation_1980_82": acceptance.share(hist_labels, "1980-01", "1982-11", ["Stagflation"]),
        "growth_gap_revision": rev,
        "nber_lags_rt": lags.to_dict(orient="records") if lags is not None else None,
        "walkforward": None if wf is None else {"n_months": int(len(wf.labels_rt)),
                                                "start": str(wf.labels_rt.index[0].date())},
        "loadings": {"growth": res.growth_loadings.round(3).to_dict(), "inflation": res.inflation_loadings.round(3).to_dict()},
        "acceptance_tests": table.reset_index().to_dict(orient="records"),
        "acceptance_all_passed": bool(acceptance.all_passed(table)),
        "acceptance_known_failures": {name: acceptance.KNOWN_FAILURES[name] for name in table.index
                                       if table.loc[name, "known_failure"] and not table.loc[name, "passed"]},
        "acceptance_blocking_failures": acceptance.blocking_failures(table),
        **extra,
    }


def publish(staging: Path, out_dir: Path, figs_dir: Path) -> None:
    """Move staged results into place with directory renames.

    The new tree is copied to <dst>.new first, then the old tree is renamed to
    <dst>.old and the new one renamed into place, so an interruption leaves
    either the previous outputs or the new ones, never a half-copied mix.
    """
    for src, dst in [(staging / "output", out_dir), (staging / "figs", figs_dir)]:
        new, old = dst.with_name(dst.name + ".new"), dst.with_name(dst.name + ".old")
        for leftover in (new, old):
            if leftover.exists():
                shutil.rmtree(leftover)
        shutil.copytree(src, new)
        if dst.exists():
            dst.rename(old)
        new.rename(dst)
        if old.exists():
            shutil.rmtree(old)
    shutil.rmtree(staging)


def clear_asset_artefacts(out_dir: Path, figs_dir: Path) -> None:
    """Remove every asset artefact a previous run published (plus any staging left behind)."""
    for name in ASSET_CSVS:
        (out_dir / name).unlink(missing_ok=True)
    for pattern in ASSET_CSV_GLOBS:
        for p in out_dir.glob(pattern):
            p.unlink()
    for name in ASSET_FIGS:
        (figs_dir / f"{name}.png").unlink(missing_ok=True)
    for d in (out_dir / ASSETS_STAGING, figs_dir / ASSETS_STAGING):
        if d.exists():
            shutil.rmtree(d)


PLACEBO_STRATEGIES = ("PIT_MaxSharpe", "PIT_LongOnly_MaxSharpe")


def _placebo_block(plcs: dict | None, strategy: str) -> dict | None:
    """summary['assets'] entry for one strategy's backtest placebo; None when it was skipped."""
    if plcs is None:
        return None
    p = plcs[strategy]
    return {"real": p["real"], "percentile": p["percentile"], "n": p["n"],
            "null": np.asarray(p["null"], dtype=float).tolist()}


RT_LABEL_COLUMNS = ["hmm_rt_vintage", "growth_gap_rt_vintage", "inflation_gap_rt_vintage"]
# The comparison is published only over at least this many months shared with the walk-forward
# label; fewer (e.g. an archive holding just the vintages the daily refresh copied in) would
# publish agreement as 0% or 100% and the correlations as n/a.
RT_MIN_OVERLAP_MONTHS = 24


def archive_downloaded_vintage(path, archive_dir) -> bool:
    """Keep the archive current: copy a downloaded vintage into it under its own name, once.

    Only into an archive directory that already exists; it is never created here. The archive is
    built deliberately (scripts/build_vintage_archive.py): creating it on download would leave a
    one-file archive that publishes a one-month comparison."""
    if not archive_dir or not Path(archive_dir).is_dir():
        return False
    src = Path(path)
    dest = Path(archive_dir) / src.name
    if not src.name.startswith("fredmd_") or dest.exists():
        return False
    shutil.copyfile(src, dest)
    return True


def run_rt_vintage(archive_dir, labels_df: pd.DataFrame, staging_figs: Path, wf=None,
                   min_overlap: int = RT_MIN_OVERLAP_MONTHS, **kw):
    """The real-time-vintage stage (spec 2026-09-25). Returns (result, summary block); a failure
    anywhere becomes {"skipped": reason} and never changes the engine's exit code.

    `wf` is the final-vintage walk-forward result; its gaps are what the rt gaps are compared
    with. No month past the run's last walk-forward-labelled month is labelled, and the stage is
    skipped when fewer than `min_overlap` months overlap the walk-forward label."""
    if not archive_dir:
        return None, {"skipped": "no vintage archive configured"}
    if not labels_df["hmm_walkforward"].notna().any():
        return None, {"skipped": "walk-forward disabled: nothing to compare the real-time-vintage label with"}
    try:
        end = labels_df["hmm_walkforward"].last_valid_index().strftime("%Y-%m")
        rtv = rtvintage.fit_hmm4_rt_vintage(archive_dir, end=end, progress=lambda i, n, t: print(f"\rreal-time vintage {i}/{n} {t:%Y-%m}", end=""), **kw)
        print()
        if len(rtv.labels) == 0:
            first = next(iter(rtv.gaps.values()), "no month in range")
            return None, {"skipped": f"no month could be labelled ({len(rtv.gaps)} gaps; first: {first})"}
        n = len(rtv.labels.index.intersection(labels_df.index[labels_df["hmm_walkforward"].notna()]))
        if n < min_overlap:
            reason = f"only {n} months overlap the walk-forward label; at least {min_overlap} needed"
            print(f"real-time-vintage stage skipped: {reason}", file=sys.stderr)
            return None, {"skipped": reason}
        block = rtvintage.summary_block(rtv, labels_df,
                                        growth_gap_final=None if wf is None else wf.growth_gap_rt,
                                        inflation_gap_final=None if wf is None else wf.inflation_gap_rt)
        figures.fig12_rt_vintage(labels_df, rtv, str(staging_figs / "fig12_rt_vintage.png"))
        return rtv, block
    except Exception as e:
        print(f"real-time-vintage stage skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return None, {"skipped": f"{type(e).__name__}: {e}"}


def rt_label_columns(rtv, index: pd.DatetimeIndex) -> list[pd.Series | pd.DataFrame]:
    """The published columns for the real-time-vintage label; empty when the stage did not run."""
    if rtv is None:
        empty = [pd.Series(pd.NA, index=index, name=c) for c in RT_LABEL_COLUMNS]
        return empty + [pd.DataFrame(pd.NA, index=index, columns=[f"p_rt_{r}" for r in R.REGIMES])]
    return [rtv.labels, rtv.growth_gap, rtv.inflation_gap, rtv.probs.add_prefix("p_rt_")]


RT_BACKTEST_STRATEGIES = ["PIT_MaxSharpe", "PIT_LongOnly_MaxSharpe", "Static_6040"]


def rt_backtest_inputs(labels_df: pd.DataFrame, rtv) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Labels and probabilities for the backtest on the real-time-vintage label. Inside the rt
    window a gap month holds the previous month's label — what a reader would still have held —
    and the count of held months is reported; outside the window there is no label."""
    lf = labels_df.copy()
    win = (lf.index >= rtv.labels.index[0]) & (lf.index <= rtv.labels.index[-1])
    lab = rtv.labels.reindex(lf.index)
    held = int((lab.isna() & win).sum())
    lab[win] = lab[win].ffill()
    lf["hmm_walkforward"] = lab
    probs = rtv.probs.reindex(lf.index)
    probs[win] = probs[win].ffill()
    return lf, probs, held


def run_assets(labels_df: pd.DataFrame, probs_rt: pd.DataFrame, out_dir: Path, figs_dir: Path,
               returns_cache: Path, refresh: bool, placebo_n: int, skip_placebo: bool, rtv=None) -> dict:
    """Stage 5-6 after the engine has published. Returns the summary['assets'] block.

    The whole stage runs inside one try/except: publish() has already swapped
    output/ into place by the time this is called, so a failure anywhere here
    (network, cache, a computation, or a figure write) must not change the
    engine's exit code -- it is reported via the "skipped" key instead.

    Publication is all-or-nothing. The previous run's asset artefacts are
    removed up front, everything is written to <dir>/.assets_staging, and the
    files are moved into place only once the whole stage has succeeded, so a
    mid-stage failure can never leave this run's early files sitting beside the
    previous run's late ones.
    """
    if not labels_df["hmm_walkforward"].notna().any():
        return {"skipped": "walk-forward disabled: the asset stage needs real-time labels"}
    stage_out, stage_figs = out_dir / ASSETS_STAGING, figs_dir / ASSETS_STAGING
    try:
        clear_asset_artefacts(out_dir, figs_dir)
        stage_out.mkdir(parents=True); stage_figs.mkdir(parents=True)
        rets = assets.load_returns(cache=returns_cache, refresh=refresh)
        table = assets.regime_conditional_table(rets, labels_df, "hmm_walkforward")
        table.to_csv(stage_out / "regime_returns.csv")
        corr = assets.conditional_corr(rets, labels_df, "hmm_walkforward")
        for reg, c in corr.items():
            c.to_csv(stage_out / f"regime_corr_{reg}.csv")
        mu, cov = assets.regime_moments(rets, labels_df, "hmm_walkforward")
        aligned = assets.align_to_available(rets, labels_df, "hmm_walkforward")
        r6040 = assets.portfolio_returns(aligned.drop(columns=["label", "label_date"]), assets.W6040)
        gaps = pd.concat([assets.align_to_available(rets, labels_df, "growth_gap")["label"].rename("growth_gap"),
                          assets.align_to_available(rets, labels_df, "inflation_gap")["label"].rename("inflation_gap")], axis=1)
        growth = assets.growth_share_6040(pd.concat([r6040.rename("r6040"), gaps], axis=1))
        spread = assets.sharpe_spread_placebo(pd.DataFrame({"label": aligned["label"], "r6040": r6040}), n=1000)
        probs_avail = probs_rt.reindex(labels_df.index).dropna(how="all")
        path_df = assets.mixture_path(mu, cov, probs_avail.loc[probs_avail.index >= rets.index[0]], pd.Series(assets.W6040))
        # fig9 is descriptive, but its x-axis must still be honest: index the path by the
        # month each number could first have been known (label date + publication lag).
        path_df.index = pd.DatetimeIndex(labels_df["available_at"].reindex(path_df.index), name="available_at")
        bts = {f"cost_bp_{c}": portfolio.backtest(rets, labels_df, probs_rt, cost_bp=float(c)) for c in (0, 10)}
        rt_bt = None
        if rtv is not None:
            # Its own try: a failure of the comparator backtest must not take the existing asset
            # numbers down with it (the stage's outer try would degrade the whole block).
            try:
                lf_rt, probs_rt_v, held = rt_backtest_inputs(labels_df, rtv)
                rt_runs = {f"cost_bp_{c}": portfolio.backtest(rets, lf_rt, probs_rt_v, strategies=RT_BACKTEST_STRATEGIES,
                                                               include_expost=False, cost_bp=float(c)) for c in (0, 10)}
                rt_bt = {"perf": {k: v.perf.round(4).to_dict(orient="index") for k, v in rt_runs.items()},
                         "counters": rt_runs["cost_bp_0"].counters, "held_months": held,
                         # window is the traded span this backtest actually reports on (data-derived
                         # from its own returns, never the portfolio.backtest default start param);
                         # label_window is the rt-vintage label's own coverage, reported separately
                         # since a reader could only start trading once a label was first published.
                         "window": {"start": str(rt_runs["cost_bp_0"].returns.index[0].date()),
                                    "end": str(rt_runs["cost_bp_0"].returns.index[-1].date())},
                         "label_window": {"start": str(rtv.labels.index[0].date()),
                                           "end": str(rtv.labels.index[-1].date())}}
            except Exception as e:
                rt_bt = {"skipped": f"{type(e).__name__}: {e}"}
                print(f"real-time-vintage backtest skipped: {rt_bt['skipped']}", file=sys.stderr)
        bt0 = bts["cost_bp_0"]
        bt0.returns.to_csv(stage_out / "backtest_returns.csv")
        weight_blocks = ("PIT_MaxSharpe", "ProbWeighted_MaxSharpe", "PIT_LongOnly_MaxSharpe", "PIT_RiskParity")
        pd.concat({s: bt0.weights[s] for s in weight_blocks}, axis=1).to_csv(stage_out / "portfolio_weights.csv")
        look = portfolio.lookahead_decomposition(bt0.perf)
        look_lo = portfolio.lookahead_decomposition(bt0.perf, family="longonly")
        # One set of shuffles scores both optimisers: the long-only null asks whether matching
        # 60/40 comes from the labels or from long-only mean-variance on this universe.
        plcs = None if skip_placebo else portfolio.backtest_placebos(
            rets, labels_df, probs_rt, strategies=PLACEBO_STRATEGIES, n=placebo_n)
        figures.fig8_regime_returns(table, str(stage_figs / "fig8_regime_returns.png"))
        figures.fig9_mixture_6040(path_df, str(stage_figs / "fig9_mixture_6040.png"))
        figures.fig10_backtest_wealth(bt0.returns, str(stage_figs / "fig10_backtest_wealth.png"))
        figures.fig11_pit_weights(bt0.weights["PIT_MaxSharpe"], str(stage_figs / "fig11_pit_weights.png"))
        n_per = table.xs(rets.columns[0], level="asset")["n"].astype(int).to_dict()
        block = {
            "skipped": None, "universe": assets.UNIVERSE,
            "window": {"start": str(aligned.index[0].date()), "end": str(aligned.index[-1].date()), "n_months": int(len(aligned))},
            "n_per_regime": n_per,
            "growth_share_6040": growth,
            "sharpe_spread_placebo": {"real": spread["real"], "percentile": spread["percentile"],
                                     "null": np.asarray(spread["null"]).tolist()},
            "backtest": {k: {"perf": v.perf.round(4).to_dict(orient="index"), "counters": v.counters, "params": v.params}
                         for k, v in bts.items()},
            "lookahead": look,
            "lookahead_longonly": look_lo,
            "backtest_placebo": _placebo_block(plcs, "PIT_MaxSharpe"),
            "backtest_placebo_longonly": _placebo_block(plcs, "PIT_LongOnly_MaxSharpe"),
            "rt_vintage_backtest": rt_bt,
        }
        for staged, dest in ((stage_out, out_dir), (stage_figs, figs_dir)):
            for src in sorted(staged.iterdir()):
                shutil.move(str(src), str(dest / src.name))
            staged.rmdir()
        return block
    except Exception as e:  # asset-stage failure anywhere must not change the engine's exit code
        for d in (stage_out, stage_figs):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        return {"skipped": f"{type(e).__name__}: {e}"}


def main(argv=None, today: date | None = None, fetch=urllib.request.urlopen) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--vintage", help="YYYY-MM, or `latest` for the newest vintage the Fed has posted")
    ap.add_argument("--if-newer", action="store_true",
                    help="exit 3 without running the engine unless the vintage is newer than the published one")
    ap.add_argument("--trend-window", type=int, default=DEFAULTS["window"])
    ap.add_argument("--theta", type=float, default=DEFAULTS["theta"])
    ap.add_argument("--no-walkforward", action="store_true")
    ap.add_argument("--wf-step", type=int, default=1)
    ap.add_argument("--wf-min-obs", type=int, default=240)
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-expanding", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "output"))
    ap.add_argument("--figs-dir", default=str(HERE / "figs"))
    ap.add_argument("--data-sheet", default=str(HERE / "data" / "README.md"))
    ap.add_argument("--no-assets", action="store_true")
    ap.add_argument("--returns-cache", default=str(HERE / "data" / "returns_yfinance.parquet"))
    ap.add_argument("--refresh-returns", action="store_true")
    ap.add_argument("--placebo-n", type=int, default=200)
    ap.add_argument("--skip-placebo", action="store_true")
    ap.add_argument("--vintage-archive", default=None,
                    help="directory of fredmd_YYYY-MM.csv vintages; enables the real-time-vintage comparator")
    ap.add_argument("--rt-min-overlap", type=int, default=RT_MIN_OVERLAP_MONTHS,
                    help="months the real-time-vintage label must share with the walk-forward label "
                         "before the comparison is published")
    a = ap.parse_args(argv)
    if a.if_newer and not a.vintage:
        ap.error("--if-newer needs --vintage (it compares the requested vintage with the published one)")
    out_dir, figs_dir = Path(a.out_dir), Path(a.figs_dir)

    vintage = a.vintage
    if a.vintage == "latest":
        vintage, resolved = latest_vintage(today or date.today(), HERE / "data", fetch=fetch)
        # Every ask counts as a check, including the ones that publish nothing: it is the only
        # evidence the dashboard has that a quiet site is still being looked after.
        write_last_check(out_dir, utc_stamp(), vintage)
        if vintage is None:
            # Distinct from "nothing new": every vintage in the window failed, including the one
            # already published, so this is the source being unreachable. The scheduled job
            # debounces this code rather than alerting on it daily.
            print("no FRED-MD vintage could be downloaded; the source is unreachable", file=sys.stderr)
            return 4
        path = str(resolved)
    elif a.vintage:
        path = str(download_vintage(a.vintage, HERE / "data", fetch=fetch))
        write_last_check(out_dir, utc_stamp(), a.vintage)      # the Refresh button is a check too
    else:
        path = a.path
    if not path:
        ap.error("give a FRED-MD csv path or --vintage YYYY-MM")
    if a.vintage:
        archive_downloaded_vintage(path, a.vintage_archive)
    if a.if_newer:
        published = published_vintage(out_dir)
        if published is not None and vintage <= published:
            print(f"vintage {vintage} is not newer than the published {published}; nothing to do")
            return 3
    kw = dict(window=a.trend_window, theta=a.theta)
    staging = out_dir.parent / (out_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "output").mkdir(parents=True); (staging / "figs").mkdir(parents=True)

    try:

        res = run_pipeline(path, **kw)

    except VintageError as e:

        print(f"VINTAGE REJECTED: {e} — {out_dir} untouched", file=sys.stderr)

        return 1
    print(f"sample {res.hmm.labels_filtered.index[0]:%Y-%m}..{res.hmm.labels_filtered.index[-1]:%Y-%m}; "
          f"outliers removed {len(res.blocks['outliers'])}; masked months {int((~res.est_mask).sum())}")
    free = R.fit_free_hmm4(res.G, res.P, res.est_mask, persistence=res.params["persistence"], eps=res.params["eps"])
    gmm = R.fit_gmm4(res.G, res.P, res.est_mask)

    wf = None
    if not a.no_walkforward:
        wf = fit_hmm4_walkforward(path, min_obs=a.wf_min_obs, step=a.wf_step,
                                  progress=lambda i, n, t: print(f"\rwalk-forward {i}/{n} {t:%Y-%m}", end=""), **kw)
        print()
    if wf is not None:
        hist_labels, hist_probs, label_source = wf.labels_rt, wf.probs_rt, "walk-forward filtered"
    else:
        hist_labels, hist_probs, label_source = res.hmm.labels_filtered, res.hmm.probs_filtered, "full-sample filtered (walk-forward disabled)"
    quad_hist = (R.quadrant_labels(wf.growth_gap_rt, wf.inflation_gap_rt, a.theta) if wf is not None
                 else res.quadrant)

    vals = {}
    vals.update(acceptance.history_metrics(hist_labels, quad_hist.reindex(hist_labels.index), hist_probs))
    vals.update(acceptance.model_metrics(res))
    vals.update(acceptance.truncation_metrics(path, **kw))
    vals["seed_invariance_disagreements"] = acceptance.seed_metric(path, **kw)
    table = acceptance.evaluate(vals)
    print(table.to_string())

    fig = lambda n: str(staging / "figs" / f"{n}.png")
    figures.fig1_factors_gaps(res, fig(FIG_NAMES[0]))
    figures.fig2_regime_timeline(res, wf, free, gmm, fig(FIG_NAMES[1]))
    figures.fig3_state_space(res, wf, fig(FIG_NAMES[2]))
    figures.fig4_hmm_probabilities(res, wf, fig(FIG_NAMES[3]))
    rev = figures.fig5_revisions(res, fig(FIG_NAMES[4]))
    figures.fig6_classifier_comparison(res, free, gmm, fig(FIG_NAMES[5]))
    lags = figures.fig7_walkforward(res, wf, fig(FIG_NAMES[6])) if wf is not None else None

    extra = {}
    if not a.skip_robustness:
        extra["robustness"] = robustness_table(path, theta=a.theta)
    if not a.skip_expanding:
        ep, loads = pca_factor_expanding(res.blocks["growth"], "INDPRO", res.est_mask)
        extra["expanding_loadings"] = {"indpro_loading_range": [float(loads["INDPRO"].min()), float(loads["INDPRO"].max())],
                                       "endpoint_vs_full_factor_corr": float(pd.concat([ep["factor"], res.growth_factor["factor"]], axis=1).dropna().corr().iloc[0, 1])}
    summary = build_summary(res, free, gmm, wf, hist_labels, hist_probs, label_source, table, rev, lags, extra)

    parts = [res.growth_factor["factor"].rename("growth_factor"),
             res.inflation_factor["factor"].rename("inflation_factor"),
             pd.Series(pd.NA, index=res.hmm.labels_filtered.index, name="hmm_walkforward") if wf is None else wf.labels_rt,
             quad_hist.rename("quadrant_walkforward") if wf is not None
             else pd.Series(pd.NA, index=res.hmm.labels_filtered.index, name="quadrant_walkforward"),
             free.labels_filtered.rename("hmm_free"), gmm.labels.rename("gmm"),
             (res.hmm.probs_filtered if wf is None else wf.probs_rt).add_prefix("p_")]
    labels_df = labels_frame(res, extra=parts)
    hist_col = "hmm_walkforward" if wf is not None else "hmm_filtered"
    quad_col = "quadrant_walkforward" if wf is not None else "quadrant"
    # A coarse --wf-step leaves the most recent months unlabelled (labels_frame left-joins
    # the sparse walk-forward grid with no forward-fill), so "current" must be read at the
    # last month hist_col actually has a label for, not the sample's last month; the full
    # monthly walk-forward (--wf-step 1) always labels the last month, so this is a no-op there.
    last = labels_df[hist_col].last_valid_index()
    summary["run"] = {"timestamp": utc_stamp(),
                      "vintage": os.path.basename(str(path)), "asof": str(last.date()),
                      "engine": "regime_v2", "label_source": label_source}
    summary["current"] = {"month": last.strftime("%Y-%m"),
                          "regime": str(labels_df.loc[last, hist_col]),
                          "quadrant": str(labels_df.loc[last, quad_col]),
                          "growth_gap": float(labels_df.loc[last, "growth_gap"]),
                          "inflation_gap": float(labels_df.loc[last, "inflation_gap"]),
                          "probs": {r: float(labels_df.loc[last, f"p_{r}"]) for r in R.REGIMES}}

    rtv, rt_block = run_rt_vintage(a.vintage_archive, labels_df, staging / "figs", wf=wf,
                                   min_overlap=a.rt_min_overlap, **kw)
    for col in rt_label_columns(rtv, labels_df.index):
        labels_df = labels_df.join(col, how="left")
    summary["rt_vintage"] = rt_block
    if rt_block.get("skipped") is None:
        vals.update({"rt_vintage_agreement": rt_block["agreement"]["overall"],
                     "rt_vintage_agreement_reconstructed": rt_block["agreement"]["reconstructed"],
                     "rt_vintage_agreement_published": rt_block["agreement"]["published"],
                     "rt_growth_gap_corr": rt_block["gap_revision"]["growth"]["corr"],
                     "rt_inflation_gap_corr": rt_block["gap_revision"]["inflation"]["corr"]})
        table = acceptance.evaluate(vals)
        summary["acceptance_tests"] = table.reset_index().to_dict(orient="records")
        summary["acceptance_all_passed"] = bool(acceptance.all_passed(table))
        summary["acceptance_known_failures"] = {name: acceptance.KNOWN_FAILURES[name] for name in table.index
                                                if table.loc[name, "known_failure"] and not table.loc[name, "passed"]}
        summary["acceptance_blocking_failures"] = acceptance.blocking_failures(table)

    labels_df.to_csv(staging / "output" / "regime_labels.csv")
    res.blocks["outliers"].to_csv(staging / "output" / "outliers_removed.csv", index=False)
    table.to_csv(staging / "output" / "acceptance.csv")
    (staging / "output" / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_data_sheet(res.blocks, a.data_sheet)

    if not acceptance.all_passed(table):
        print(f"ACCEPTANCE FAILED ({acceptance.blocking_failures(table)}) — outputs left in {staging}; {out_dir} untouched", file=sys.stderr)
        return 1
    publish(staging, out_dir, figs_dir)
    print(f"published to {out_dir} and {figs_dir}")

    if not a.no_assets:
        probs_src = wf.probs_rt if wf is not None else res.hmm.probs_filtered
        block = run_assets(labels_df, probs_src, out_dir, figs_dir, Path(a.returns_cache), a.refresh_returns,
                           a.placebo_n, a.skip_placebo, rtv=rtv)
        if block.get("skipped") is None:
            # Promoting the metrics into the acceptance table is part of the stage: if it
            # fails, the stage degrades to "skipped" like any other asset-stage failure
            # (and drops its artefacts) rather than leaving a half-updated acceptance.csv.
            try:
                perf0 = block["backtest"]["cost_bp_0"]["perf"]
                perf10 = block["backtest"]["cost_bp_10"]["perf"]
                vals.update({"pit_sharpe": block["lookahead"]["pit_sharpe"], "oracle_sharpe": block["lookahead"]["oracle_sharpe"],
                             "insample_sharpe": block["lookahead"]["insample_sharpe"],
                             "label_lookahead": block["lookahead"]["label_lookahead"],
                             "moment_lookahead": block["lookahead"]["moment_lookahead"],
                             "growth_share_6040": block["growth_share_6040"]["growth_share"],
                             "growth_share_6040_r2": block["growth_share_6040"]["r2"],
                             "static_6040_sharpe": perf0["Static_6040"]["sharpe"],
                             "pit_sharpe_10bp": perf10["PIT_MaxSharpe"]["sharpe"],
                             "backtest_placebo_pct": (block["backtest_placebo"] or {}).get("percentile", float("nan")),
                             "longonly_placebo_pct": (block["backtest_placebo_longonly"] or {}).get("percentile", float("nan")),
                             "pit_longonly_sharpe": perf0["PIT_LongOnly_MaxSharpe"]["sharpe"],
                             "pit_riskparity_sharpe": perf0["PIT_RiskParity"]["sharpe"],
                             "longonly_moment_lookahead": block["lookahead_longonly"]["moment_lookahead"],
                             "longonly_label_lookahead": block["lookahead_longonly"]["label_lookahead"],
                             "rt_pit_sharpe": ((block.get("rt_vintage_backtest") or {}).get("perf", {}).get("cost_bp_0", {}).get("PIT_MaxSharpe", {}).get("sharpe", float("nan"))),
                             "rt_pit_longonly_sharpe": ((block.get("rt_vintage_backtest") or {}).get("perf", {}).get("cost_bp_0", {}).get("PIT_LongOnly_MaxSharpe", {}).get("sharpe", float("nan")))})
                table = acceptance.evaluate(vals)
                table.to_csv(out_dir / "acceptance.csv")
                # Every key build_summary derives from the table has to move with it, or the
                # summary reports a pass/fail set from the pre-promotion evaluation beside
                # asset rows from this one.
                summary["acceptance_tests"] = table.reset_index().to_dict(orient="records")
                summary["acceptance_all_passed"] = bool(acceptance.all_passed(table))
                summary["acceptance_known_failures"] = {name: acceptance.KNOWN_FAILURES[name] for name in table.index
                                                        if table.loc[name, "known_failure"] and not table.loc[name, "passed"]}
                summary["acceptance_blocking_failures"] = acceptance.blocking_failures(table)
            except Exception as e:
                clear_asset_artefacts(out_dir, figs_dir)
                block = {"skipped": f"{type(e).__name__}: {e}"}
        summary["assets"] = block
        if block.get("skipped") is not None:
            print(f"asset stage skipped: {block['skipped']}", file=sys.stderr)
        # Publish the "assets" block first so the doc figures (which read the published
        # run back off disk via load_published, per the site-docs contract) see it.
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print("asset stage " + ("skipped" if block.get("skipped") else "published"))

    # Doc figures (docs/site/CONTRACT.md): drawn on every run, whether the asset stage
    # published, was skipped, or was not requested (the two asset-dependent figures are
    # then simply absent). A failure here must never change the exit code.
    # The re-write of summary.json belongs inside the guard too: output/ and figs/ are already
    # published at this point, and an IO error while adding the doc_figures key must not fail
    # a run that otherwise succeeded.
    try:
        pub = load_published(out_dir, figs_dir)
        doc_paths = docfigs.write_doc_figures(pub, figs_dir)
        # File names, not host paths: the published surface is read by name against figs_dir.
        summary["doc_figures"] = {k: (v.name if v is not None else None) for k, v in doc_paths.items()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"doc figures failed: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
