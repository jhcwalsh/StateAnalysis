"""Stage 4 — monthly re-estimation (D8).

For each month t the whole pipeline is re-run on data <= t and the
*filtered* probability for t is kept. Nothing from after t is ever seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import load_fredmd
from .pipeline import run_pipeline
from .regimes import REGIMES


@dataclass
class WalkForwardResult:
    labels_rt: pd.Series
    probs_rt: pd.DataFrame
    growth_gap_rt: pd.Series
    inflation_gap_rt: pd.Series
    transmat_by_month: dict


def fit_hmm4_walkforward(path: str, min_obs: int = 240, step: int = 1, start: str | None = None,
                         end: str | None = None, progress=None, **kw) -> WalkForwardResult:
    levels, _ = load_fredmd(path)
    months = levels.index
    lo = pd.Timestamp(start) if start else months[min_obs - 1]
    hi = pd.Timestamp(end) if end else months[-1]
    targets = [t for t in months if lo <= t <= hi][::step]
    probs, gg, pp, tms = {}, {}, {}, {}
    for i, t in enumerate(targets):
        res = run_pipeline(path, asof=str(t + pd.offsets.MonthEnd(0)), **kw)
        if t not in res.hmm.probs_filtered.index:
            continue
        probs[t] = res.hmm.probs_filtered.loc[t]
        gg[t], pp[t] = res.G.loc[t], res.P.loc[t]
        tms[t] = res.hmm.transmat
        if progress:
            progress(i + 1, len(targets), t)
    P = pd.DataFrame(probs).T.reindex(columns=REGIMES)
    P.index.name = "date"
    return WalkForwardResult(
        labels_rt=P.idxmax(axis=1).rename("hmm_walkforward"),
        probs_rt=P,
        growth_gap_rt=pd.Series(gg, name="growth_gap_rt"),
        inflation_gap_rt=pd.Series(pp, name="inflation_gap_rt"),
        transmat_by_month=tms,
    )
