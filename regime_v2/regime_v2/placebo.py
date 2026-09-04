"""Stage 4 — null distributions for regime-conditional statistics.

placebo: reorder the observed regime runs (run-length distribution kept)
and recompute any statistic. block_bootstrap: resample 12-month blocks
with replacement for confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_shuffle(labels: pd.Series, rng: np.random.Generator) -> pd.Series:
    runs = (labels != labels.shift()).cumsum()
    pieces = [grp.to_list() for _, grp in labels.groupby(runs)]
    order = rng.permutation(len(pieces))
    flat = [v for i in order for v in pieces[i]]
    return pd.Series(flat, index=labels.index, name=labels.name)


def placebo(labels: pd.Series, stat_fn, n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    real = float(stat_fn(labels))
    null = np.array([stat_fn(block_shuffle(labels, rng)) for _ in range(n)], dtype=float)
    pct = float((null <= real).mean() * 100.0)
    return {"null": null, "real": real, "percentile": pct}


def block_bootstrap(df: pd.DataFrame, stat_fn, block: int = 12, n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = len(df)
    n_blocks = int(np.ceil(T / block))
    rows = []
    for _ in range(n):
        starts = rng.integers(0, max(T - block, 1), size=n_blocks)
        pos = np.concatenate([np.arange(s, min(s + block, T)) for s in starts])[:T]
        rows.append(stat_fn(df.iloc[pos]))
    return pd.DataFrame(rows).reset_index(drop=True)
