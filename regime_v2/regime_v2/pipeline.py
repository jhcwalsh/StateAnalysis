"""The one composition of data -> factors -> gaps -> labels (D8).

`run_pipeline(path, asof=t)` is what the walk-forward driver calls each
month; `run_pipeline(path)` is the full-sample run used for model-form
checks and the ex-post comparator.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import COVID_MASK, build_blocks
from .factors import pca_factor_em
from .regimes import HMMResult, fit_hmm4, quadrant_labels
from .trend import make_gap

DEFAULTS = dict(window=240, smooth=3, method="smoothed_trailing", theta=0.5,
                persistence=10.0, eps=0.5, seed=0, mask=COVID_MASK, k_outlier=10.0,
                publication_lag_months=1)


@dataclass
class PipelineResult:
    blocks: dict
    est_mask: pd.Series
    growth_factor: pd.DataFrame
    growth_loadings: pd.Series
    inflation_factor: pd.DataFrame
    inflation_loadings: pd.Series
    g_gap: pd.DataFrame
    p_gap: pd.DataFrame
    G: pd.Series
    P: pd.Series
    quadrant: pd.Series
    quadrant0: pd.Series
    hmm: HMMResult
    params: dict


def trend_kwargs(params: dict) -> dict:
    """Keyword arguments for make_gap under the chosen method (window-less methods get none)."""
    if params["method"] == "smoothed_trailing":
        return dict(window=params["window"], smooth=params["smooth"])
    if params["method"] in ("trailing_mean", "trailing_median"):
        return dict(window=params["window"])
    return {}


def run_pipeline(path: str, asof: str | None = None, **overrides) -> PipelineResult:
    params = {**DEFAULTS, **overrides}
    blocks = build_blocks(path, k_outlier=params["k_outlier"], asof=asof, mask=params["mask"])
    m = blocks["estimation_mask"]
    gf, gl = pca_factor_em(blocks["growth"], "INDPRO", m)
    pf, pl = pca_factor_em(blocks["inflation"], "CPIAUCSL", m)
    trend_kw = trend_kwargs(params)
    g_gap = make_gap(gf["factor"], params["method"], m, **trend_kw)
    p_gap = make_gap(pf["diffusion"], params["method"], m, **trend_kw)
    G, P = g_gap["gap"].rename("growth_gap"), p_gap["gap"].rename("inflation_gap")
    n_ok = int(pd.concat([G, P], axis=1).dropna().shape[0])
    if n_ok < 60:
        raise ValueError(f"only {n_ok} months with both gaps after burn-in (asof={asof}); "
                         "need at least 60 — extend the sample or lower the trend window")
    hmm = fit_hmm4(G, P, m, persistence=params["persistence"], eps=params["eps"], seed=params["seed"])
    idx = hmm.labels_filtered.index
    quad = quadrant_labels(G.reindex(idx), P.reindex(idx), theta=params["theta"])
    quad0 = quadrant_labels(G.reindex(idx), P.reindex(idx), theta=0.0)
    return PipelineResult(blocks=blocks, est_mask=m, growth_factor=gf, growth_loadings=gl,
                         inflation_factor=pf, inflation_loadings=pl, g_gap=g_gap, p_gap=p_gap,
                         G=G, P=P, quadrant=quad, quadrant0=quad0, hmm=hmm, params=params)


def labels_frame(res: PipelineResult, extra: list[pd.DataFrame | pd.Series] | None = None) -> pd.DataFrame:
    idx = res.hmm.labels_filtered.index
    df = pd.DataFrame(index=idx)
    df["available_at"] = idx + pd.DateOffset(months=res.params["publication_lag_months"])
    df["growth_gap"] = res.G.reindex(idx)
    df["inflation_gap"] = res.P.reindex(idx)
    df["quadrant"] = res.quadrant
    df["quadrant_theta"] = res.params["theta"]
    df["hmm_filtered"] = res.hmm.labels_filtered
    df["hmm_smoothed_expost"] = res.hmm.labels_smoothed_expost
    for x in extra or []:
        df = df.join(x, how="left")
    df.index.name = "date"
    return df
