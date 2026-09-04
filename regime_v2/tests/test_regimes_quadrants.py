import numpy as np
import pandas as pd
import pytest

from regime_v2 import regimes as R


def _noisy():
    v = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, -0.30, -0.10, 0.10, -0.10, 0.30, 0.05]
    return pd.Series(v, index=pd.date_range("2020-01-01", periods=len(v), freq="MS"))


def _switches(s):
    return int((s.diff() != 0).sum()) - 1


def test_names_and_colours():
    assert R.REGIMES == ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
    assert R.COLORS == {"Contraction": "#3b5bdb", "Goldilocks": "#2b8a3e",
                        "Overheating": "#f08c00", "Stagflation": "#e03131"}
    assert R.SIGNS == {"Contraction": (-1, -1), "Goldilocks": (1, -1),
                       "Overheating": (1, 1), "Stagflation": (-1, 1)}


def test_theta_zero_is_memoryless_sign():
    s = _noisy()
    assert list(R.hysteretic_sign(s, 0.0)) == [1 if x >= 0 else -1 for x in s]


def test_hysteresis_reduces_switches_and_is_causal():
    s = _noisy()
    assert _switches(R.hysteretic_sign(s, 0.0)) == 8
    assert _switches(R.hysteretic_sign(s, 0.25)) == 2
    full = R.hysteretic_sign(s, 0.25)
    assert list(full.iloc[:7]) == list(R.hysteretic_sign(s.iloc[:7], 0.25))


def test_quadrant_labels_sign_pairs_and_alignment():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    g = pd.Series([-1.0, 1.0, 1.0, -1.0, np.nan], index=idx)
    p = pd.Series([-1.0, -1.0, 1.0, 1.0, 1.0], index=idx)
    out = R.quadrant_labels(g, p, theta=0.5)
    assert list(out) == ["Contraction", "Goldilocks", "Overheating", "Stagflation"]
    assert out.name == "quadrant" and len(out) == 4


def test_run_lengths_and_duration():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    lab = pd.Series(["Goldilocks"] * 3 + ["Contraction"] * 2 + ["Goldilocks"], index=idx)
    rl = R.run_lengths(lab)
    assert rl["Goldilocks"] == 2.0 and rl["Contraction"] == 2.0 and np.isnan(rl["Stagflation"])
    tm = pd.DataFrame(np.array([[0.9, 0.1, 0, 0], [0.05, 0.95, 0, 0], [0, 0, 0.8, 0.2], [0, 0, 0.5, 0.5]]),
                      index=R.REGIMES, columns=R.REGIMES)
    d = R.expected_duration(tm)
    assert d["Contraction"] == pytest.approx(10.0) and d["Goldilocks"] == pytest.approx(20.0)


def test_transition_table_rows_are_from_state():
    A = np.array([[0.7, 0.1, 0.1, 0.1], [0.0, 1.0, 0.0, 0.0], [0.25] * 4, [0.4, 0.2, 0.2, 0.2]])
    names = {0: "Contraction", 1: "Goldilocks", 2: "Overheating", 3: "Stagflation"}
    tm = R.transition_table(A, names)
    assert list(tm.index) == R.REGIMES and list(tm.columns) == R.REGIMES
    assert tm.loc["Contraction", "Goldilocks"] == 0.1 and tm.loc["Goldilocks", "Contraction"] == 0.0
    assert np.allclose(tm.sum(axis=1), 1.0)
