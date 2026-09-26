"""Real-time-vintage label (spec 2026-09-25): month t labelled from the FRED-MD vintage a
reader had at t+1, by the unchanged pipeline. A comparator beside the published walk-forward
label (D8 amended 2026-09-25); never the primary label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import VintageError, load_archive_vintage
from .pipeline import run_pipeline
from .regimes import REGIMES

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
