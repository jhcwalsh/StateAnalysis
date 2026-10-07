from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from regime_v2 import theme

ROOT = Path(__file__).resolve().parents[2]


def test_tokens_match_the_streamlit_theme_file():
    cfg = theme.tokens_from_streamlit_config(ROOT / ".streamlit" / "config.toml")
    assert cfg["primaryColor"] == theme.ACCENT
    assert cfg["backgroundColor"] == theme.BG
    assert cfg["secondaryBackgroundColor"] == theme.BG_SOFT
    assert cfg["textColor"] == theme.INK
    assert cfg["borderColor"] == theme.RULE


def test_rc_dicts_differ_only_in_the_ground():
    assert theme.RC["figure.facecolor"] == theme.BG
    assert theme.RC["savefig.facecolor"] == theme.BG
    assert theme.RC_INLINE["figure.facecolor"] == "none"
    assert theme.RC_INLINE["savefig.facecolor"] == "none"
    shared = {k: v for k, v in theme.RC.items() if k not in ("figure.facecolor", "savefig.facecolor")}
    assert shared == {k: v for k, v in theme.RC_INLINE.items() if k not in ("figure.facecolor", "savefig.facecolor")}
    assert theme.RC["axes.edgecolor"] == theme.RULE and theme.RC["text.color"] == theme.INK


def test_themed_applies_inside_and_restores_after():
    before = plt.rcParams["figure.facecolor"]

    @theme.themed
    def inside():
        return plt.rcParams["figure.facecolor"], plt.rcParams["axes.edgecolor"]

    assert inside() == (theme.BG, theme.RULE)
    assert plt.rcParams["figure.facecolor"] == before


def test_themed_restores_rcparams_after_an_exception():
    before = plt.rcParams["figure.facecolor"]

    @theme.themed
    def boom():
        raise RuntimeError("mid-draw failure")

    with pytest.raises(RuntimeError):
        boom()
    assert plt.rcParams["figure.facecolor"] == before


def test_themed_keeps_the_wrapped_name():
    @theme.themed
    def fig99_example(path):
        return path

    assert fig99_example.__name__ == "fig99_example"


def test_cmap_runs_from_ground_to_accent():
    from matplotlib.colors import to_hex
    assert theme.CMAP.name == "lazyeconomist"
    assert to_hex(theme.CMAP(0.0)) == theme.BG and to_hex(theme.CMAP(1.0)) == theme.ACCENT
