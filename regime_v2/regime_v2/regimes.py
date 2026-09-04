"""Stage 3 — regime classification (D6, D7, D10).

Named regimes only. Two classifier families on the same (growth gap,
inflation gap) input:
  quadrant_labels : deterministic sign rule with causal hysteresis (D10)
  fit_hmm4        : constrained 4-state HMM with symmetric fixed emissions
                    (primary) or free emissions (challenger)
  fit_gmm4        : Gaussian mixture challenger
Challengers report under descriptive names and are bridged to quadrants by
marginalisation (D7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REGIMES = ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
SIGNS = {"Contraction": (-1, -1), "Goldilocks": (1, -1), "Overheating": (1, 1), "Stagflation": (-1, 1)}
COLORS = {"Contraction": "#3b5bdb", "Goldilocks": "#2b8a3e", "Overheating": "#f08c00", "Stagflation": "#e03131"}
_BY_SIGN = {v: k for k, v in SIGNS.items()}


def hysteretic_sign(series: pd.Series, theta: float) -> pd.Series:
    """Schmitt-trigger sign: flips + -> - only below -theta, - -> + only above +theta.

    Causal by construction (moved verbatim from regime_core.hysteretic_sign).
    """
    vals = series.to_numpy()
    if len(vals) == 0:
        return pd.Series([], index=series.index, dtype=int)
    state = 1 if vals[0] >= 0 else -1
    out = np.empty(len(vals), dtype=int)
    for i, v in enumerate(vals):
        if state > 0 and v < -theta:
            state = -1
        elif state < 0 and v > theta:
            state = 1
        out[i] = state
    return pd.Series(out, index=series.index)


def quadrant_labels(g: pd.Series, p: pd.Series, theta: float = 0.0) -> pd.Series:
    df = pd.concat([g, p], axis=1).dropna()
    sg = hysteretic_sign(df.iloc[:, 0], theta)
    sp = hysteretic_sign(df.iloc[:, 1], theta)
    lab = [_BY_SIGN[(int(a), int(b))] for a, b in zip(sg, sp)]
    return pd.Series(lab, index=df.index, name="quadrant")


def run_lengths(labels: pd.Series) -> pd.Series:
    runs = {r: [] for r in REGIMES}
    cur, n = None, 0
    for v in labels.dropna():
        if v == cur:
            n += 1
        else:
            if cur is not None and cur in runs:
                runs[cur].append(n)
            cur, n = v, 1
    if cur is not None and cur in runs:
        runs[cur].append(n)
    return pd.Series({r: (float(np.mean(v)) if v else np.nan) for r, v in runs.items()})


def expected_duration(tm: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / (1.0 - np.diag(tm.to_numpy())), index=tm.index)


def transition_table(transmat: np.ndarray, state_names: dict[int, str]) -> pd.DataFrame:
    """Rows = from-state, columns = to-state. Serialise with orient='index'."""
    names = [state_names[s] for s in range(len(state_names))]
    tm = pd.DataFrame(transmat, index=names, columns=names)
    order = [n for n in REGIMES if n in names] + [n for n in names if n not in REGIMES]
    return tm.loc[order, order]
