"""Shared regime-classification logic.

Imported by BOTH Macro_Regime_Analysis.ipynb (cells 20/51/60) and the
Streamlit app (app.py), so the hysteresis trigger and the paper-convention
constants have exactly one implementation. Function bodies were moved
verbatim from the notebook; do not edit here without re-running the
notebook's bit-identity check (see the 2026-07-19 UI design spec).
"""
import numpy as np
import pandas as pd

QUAD_BY_SIGN = {
    (-1, -1): "Q1_LowG_LowPi",
    ( 1, -1): "Q2_HighG_LowPi",
    ( 1,  1): "Q3_HighG_HighPi",
    (-1,  1): "Q4_LowG_HighPi",
}

# Paper convention: Q1=Goldilocks, Q2=Overheating, Q3=Stagflation, Q4=Recession
CODE_TO_PAPER = {
    "Q1_LowG_LowPi":   "Q4_Recession",
    "Q2_HighG_LowPi":  "Q1_Goldilocks",
    "Q3_HighG_HighPi": "Q2_Overheating",
    "Q4_LowG_HighPi":  "Q3_Stagflation",
    "Neutral":          "Neutral",
}

PAPER_DISPLAY = {
    "Q1_Goldilocks":  "Q1: Goldilocks (High G, Low Infl)",
    "Q2_Overheating": "Q2: Overheating (High G, High Infl)",
    "Q3_Stagflation": "Q3: Stagflation (Low G, High Infl)",
    "Q4_Recession":   "Q4: Recession (Low G, Low Infl)",
}

PAPER_COLORS = {
    "Q1_Goldilocks":  "#2166AC",
    "Q2_Overheating": "#4DAC26",
    "Q3_Stagflation": "#F97B06",
    "Q4_Recession":   "#D7191C",
    "Neutral":        "#888888",
}

PAPER_ORDER = ["Q1_Goldilocks", "Q2_Overheating", "Q3_Stagflation", "Q4_Recession"]


def hysteretic_sign(series, theta):
    """
    Schmitt-trigger sign of a series.

    The state flips from + to - only once the value drops BELOW -theta, and
    from - to + only once it rises ABOVE +theta. Between -theta and +theta the
    previous state persists, so observations hovering near zero stop
    oscillating.

    CAUSAL BY CONSTRUCTION: state at t is a function of observations up to t
    only. No future information enters, unlike a run-length / minimum-duration
    filter applied after the fact.
    """
    vals  = series.values
    state = 1 if vals[0] >= 0 else -1
    out   = np.empty(len(vals), dtype=int)
    for i, v in enumerate(vals):
        if   state > 0 and v < -theta: state = -1
        elif state < 0 and v >  theta: state = 1
        out[i] = state
    return pd.Series(out, index=series.index)


def assign_quadrants(g_gap, pi_gap, theta):
    """Assign a quadrant regime to each observation, with hysteresis."""
    df = pd.concat([g_gap, pi_gap], axis=1).dropna()
    sg = hysteretic_sign(df.iloc[:, 0], theta)
    sp = hysteretic_sign(df.iloc[:, 1], theta)
    lab = [QUAD_BY_SIGN[(int(a), int(b))] for a, b in zip(sg, sp)]
    return pd.Series(lab, index=df.index, name="Quadrant")


def duration_stats(labels):
    """(avg run length in periods, number of regime switches)."""
    runs = (labels != labels.shift()).cumsum()
    lengths = labels.groupby(runs).size()
    return float(lengths.mean()), int(len(lengths) - 1)
