"""Real-time-vintage label (spec 2026-09-25): month t labelled from the FRED-MD vintage a
reader had at t+1, by the unchanged pipeline. A comparator beside the published walk-forward
label (D8 amended 2026-09-25); never the primary label.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import VintageError, load_archive_vintage
from .figures import nber_lags
from .pipeline import run_pipeline
from .regimes import REGIMES, run_lengths

RECONSTRUCTED_THROUGH = "2014-12"     # vintages up to here were rebuilt by the Fed from archived Haver data
_NAME = re.compile(r"^fredmd_(\d{4}-\d{2})\.csv$")


def list_vintages(archive_dir) -> dict[str, Path]:
    d = Path(archive_dir)
    if not d.is_dir():
        return {}
    out = {m.group(1): p for p in d.iterdir() if (m := _NAME.match(p.name))}
    return dict(sorted(out.items()))


def vintage_for_month(month: str) -> str:
    return (pd.Period(month, "M") + 1).strftime("%Y-%m")


def provenance(vintage: str) -> str:
    return "reconstructed" if vintage <= RECONSTRUCTED_THROUGH else "published"


@dataclass
class RtVintageResult:
    labels: pd.Series
    probs: pd.DataFrame
    growth_gap: pd.Series
    inflation_gap: pd.Series
    gaps: dict = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    provenance: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    start: str = ""


def fit_hmm4_rt_vintage(archive_dir, start: str = "1999-07", end: str | None = None,
                        progress=None, **kw) -> RtVintageResult:
    vintages = list_vintages(archive_dir)
    if not vintages:
        raise FileNotFoundError(f"no fredmd_YYYY-MM.csv files in {archive_dir}")
    names = list(vintages)
    first_month = (pd.Period(names[0], "M") - 1).strftime("%Y-%m")
    last_month = (pd.Period(names[-1], "M") - 1).strftime("%Y-%m")
    start = max(start, first_month)
    end = min(end or last_month, last_month)
    months = pd.period_range(start, end, freq="M")
    probs, gg, pp, prov, gaps, reports = {}, {}, {}, {}, {}, {}
    for i, m in enumerate(months):
        month = str(m)
        v = vintage_for_month(month)
        if v not in vintages:
            gaps[month] = f"vintage {v} not in archive"
            continue
        pos = names.index(v)
        neighbour = vintages[names[pos - 1]] if pos > 0 else (vintages[names[1]] if len(names) > 1 else None)

        def loader(path, _n=neighbour):
            levels, tcodes, report = load_archive_vintage(path, neighbour=None if _n is None else str(_n))
            reports[v] = report
            return levels, tcodes

        t = m.to_timestamp()
        try:
            res = run_pipeline(str(vintages[v]), asof=str(m.to_timestamp(how="end").normalize()), loader=loader, **kw)
        except VintageError as e:
            reports[v] = {"refused": str(e)}
            gaps[month] = f"vintage {v} refused: {e}"
            continue
        except Exception as e:
            # Any other pipeline failure (e.g. too few burn-in months) costs this one
            # month, not the whole multi-decade walk (unattended daily run; §12).
            reports[v] = {"failed": f"{type(e).__name__}: {e}"}
            gaps[month] = f"vintage {v} failed: {type(e).__name__}: {e}"
            continue
        if t not in res.hmm.probs_filtered.index:
            gaps[month] = f"vintage {v} carries no usable row for {month}"
            continue
        probs[t] = res.hmm.probs_filtered.loc[t]
        gg[t], pp[t] = float(res.G.loc[t]), float(res.P.loc[t])
        prov[t] = provenance(v)
        if progress:
            progress(i + 1, len(months), t)
    P = pd.DataFrame(probs).T.reindex(columns=REGIMES)
    P.index = pd.DatetimeIndex(P.index, name="date")
    return RtVintageResult(labels=P.idxmax(axis=1).rename("hmm_rt_vintage"), probs=P,
                           growth_gap=pd.Series(gg, name="growth_gap_rt_vintage", dtype=float),
                           inflation_gap=pd.Series(pp, name="inflation_gap_rt_vintage", dtype=float),
                           gaps=gaps, reports=reports, provenance=pd.Series(prov, dtype=object), start=start)


def _agree(a: pd.Series, b: pd.Series) -> float | None:
    return None if len(a) == 0 else float((a == b).mean())


def _revision(rt: pd.Series, fin: pd.Series) -> dict:
    d = pd.concat([rt.rename("rt"), fin.rename("fin")], axis=1).dropna()
    if len(d) < 3:
        return {"mean": None, "sd": None, "corr": None, "sign_agreement": None}
    diff = d["rt"] - d["fin"]
    return {"mean": float(diff.mean()), "sd": float(diff.std()),
            "corr": float(d["rt"].corr(d["fin"])),
            "sign_agreement": float((np.sign(d["rt"]) == np.sign(d["fin"])).mean())}


def summary_block(rtv: RtVintageResult, labels_df: pd.DataFrame) -> dict:
    """The comparison of the real-time-vintage label with the published walk-forward label,
    computed over their overlap. Everything here is reported, never thresholded."""
    j = rtv.labels.index.intersection(labels_df.index[labels_df["hmm_walkforward"].notna()])
    fin, rt = labels_df.loc[j, "hmm_walkforward"], rtv.labels.loc[j]
    prov = rtv.provenance.reindex(j)
    recon, pub = prov == "reconstructed", prov == "published"
    cross = pd.crosstab(fin, rt).reindex(index=REGIMES, columns=REGIMES, fill_value=0)
    renamed, dropped, missing, refused, failed = Counter(), Counter(), Counter(), {}, {}
    for v, rep in rtv.reports.items():
        if "refused" in rep:
            refused[v] = rep["refused"]
            continue
        if "failed" in rep:
            failed[v] = rep["failed"]
            continue
        renamed.update(rep["renamed"].keys())
        dropped.update(rep["dropped_check"] + rep["dropped_tcode"])
        missing.update(rep["missing"])
    lags_f, lags_r = nber_lags(fin), nber_lags(rt)
    win_start = j[0].strftime("%Y-%m") if len(j) else "9999-99"
    win_end = j[-1].strftime("%Y-%m") if len(j) else "0000-00"
    in_window = (lags_f["peak"] >= win_start) & (lags_f["peak"] <= win_end)
    return {
        "window": {"start": j[0].strftime("%Y-%m") if len(j) else None,
                   "end": j[-1].strftime("%Y-%m") if len(j) else None, "n_months": int(len(j))},
        "start_requested": rtv.start,
        "gaps": dict(rtv.gaps), "n_gaps": len(rtv.gaps),
        "n_reconstructed": int(recon.sum()), "n_published": int(pub.sum()),
        "agreement": {"overall": _agree(fin, rt), "reconstructed": _agree(fin[recon], rt[recon]),
                      "published": _agree(fin[pub], rt[pub])},
        "crosstab": {r: {c: int(cross.loc[r, c]) for c in REGIMES} for r in REGIMES},
        "shares": {"final": fin.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0).round(6).to_dict(),
                   "rt": rt.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0).round(6).to_dict()},
        "mean_run_length": {"final": run_lengths(fin).round(2).to_dict(), "rt": run_lengths(rt).round(2).to_dict()},
        "switches": {"final": int((fin != fin.shift()).sum() - 1) if len(fin) else 0,
                     "rt": int((rt != rt.shift()).sum() - 1) if len(rt) else 0},
        "gap_revision": {"growth": _revision(rtv.growth_gap.reindex(j), labels_df.loc[j, "growth_gap"]),
                         "inflation": _revision(rtv.inflation_gap.reindex(j), labels_df.loc[j, "inflation_gap"])},
        "nber_lags": {"final": lags_f[in_window].to_dict(orient="records"),
                      "rt": lags_r[in_window].to_dict(orient="records")},
        "loader": {"n_vintages": len(rtv.reports), "renamed": dict(renamed), "dropped": dict(dropped),
                   "missing": dict(missing), "refused": refused, "failed": failed},
    }
