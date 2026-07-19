import pandas as pd
import pytest

from regime_core import (
    hysteretic_sign, assign_quadrants, duration_stats,
    QUAD_BY_SIGN, CODE_TO_PAPER, PAPER_ORDER, PAPER_COLORS, PAPER_DISPLAY,
)


def _noisy_series():
    # Verified fixture: hovers around zero, one decisive down-move, one up-move.
    v = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05,
         -0.30, -0.10, 0.10, -0.10, 0.30, 0.05]
    idx = pd.date_range("2020-01-01", periods=len(v), freq="ME")
    return pd.Series(v, index=idx)


def _switches(s):
    return int((s.diff() != 0).sum()) - 1


def test_theta_zero_is_memoryless_sign():
    s = _noisy_series()
    out = hysteretic_sign(s, 0.0)
    expected = [1 if x >= 0 else -1 for x in s]
    # theta=0: flips whenever the value crosses 0 (v<0 -> -1, v>0 -> +1);
    # exact zeros hold the previous state, none in this fixture.
    assert list(out) == expected


def test_hysteresis_reduces_switches():
    s = _noisy_series()
    assert _switches(hysteretic_sign(s, 0.0)) == 8
    assert _switches(hysteretic_sign(s, 0.25)) == 2


def test_trigger_is_causal_prefix_property():
    s = _noisy_series()
    full = hysteretic_sign(s, 0.25)
    prefix = hysteretic_sign(s.iloc[:7], 0.25)
    assert list(full.iloc[:7]) == list(prefix)


def test_assign_quadrants_maps_sign_pairs():
    idx = pd.date_range("2020-01-01", periods=4, freq="ME")
    g = pd.Series([-1.0, 1.0, 1.0, -1.0], index=idx)
    p = pd.Series([-1.0, -1.0, 1.0, 1.0], index=idx)
    out = assign_quadrants(g, p, theta=0.5)
    assert list(out) == ["Q1_LowG_LowPi", "Q2_HighG_LowPi",
                         "Q3_HighG_HighPi", "Q4_LowG_HighPi"]
    assert out.name == "Quadrant"


def test_duration_stats():
    idx = pd.date_range("2020-01-01", periods=5, freq="ME")
    labels = pd.Series(["A", "A", "B", "B", "B"], index=idx)
    avg, switches = duration_stats(labels)
    assert avg == pytest.approx(2.5)
    assert switches == 1


def test_paper_constants_consistent():
    assert set(QUAD_BY_SIGN.values()) == set(CODE_TO_PAPER) - {"Neutral"}
    assert set(PAPER_ORDER) <= set(CODE_TO_PAPER.values())
    for k in PAPER_ORDER:
        assert k in PAPER_COLORS and k in PAPER_DISPLAY
