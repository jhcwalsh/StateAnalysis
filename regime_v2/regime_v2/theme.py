"""The lazyeconomist.com palette for every matplotlib figure the engine saves.

One source for the site's tokens (the Streamlit side reads the same values from
.streamlit/config.toml; site_theme.py re-exports these for the app's inline charts).
Nothing here imports Streamlit, so the engine and its tests stay free of it.
Fonts are left at matplotlib's DejaVu Sans: the site's web fonts are loaded by the
browser and are not installed where the PNGs are drawn.
"""
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BG, BG_SOFT = "#fbfaf7", "#f4f2ec"
INK, INK_SOFT, INK_FAINT, RULE = "#1a1a1a", "#4a4a4a", "#8a8780", "#e8e4dc"
ACCENT, ACCENT_SOFT = "#b8410e", "#f5e6dd"
GROWTH_C, INFL_C = "#2C7FB8", "#D95F0E"
SHADE_ALPHA = 0.15          # NBER recession bands: INK_FAINT at this alpha

# A sequential map from the page ground to the accent, for the transition-matrix heat map.
CMAP = LinearSegmentedColormap.from_list("lazyeconomist", [BG, ACCENT])

_RC_SHARED = {
    "axes.facecolor": "none", "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": INK_SOFT, "axes.titlesize": 10, "axes.edgecolor": RULE, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK_FAINT, "ytick.color": INK_FAINT, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": RULE, "grid.alpha": 0.9, "grid.linewidth": 0.6, "savefig.edgecolor": "none",
}
# Saved PNGs: an explicit cream ground, because the files are also downloaded and viewed alone.
RC = dict(_RC_SHARED, **{"figure.facecolor": BG, "savefig.facecolor": BG})
# The app's inline charts: transparent, so they sit on the page's own ground.
RC_INLINE = dict(_RC_SHARED, **{"figure.facecolor": "none", "savefig.facecolor": "none"})


def themed(fn):
    """Run a figure function under the saved-PNG rcParams and restore the previous ones after,
    whether it returns or raises (matplotlib.rc_context does the restoring)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return fn(*args, **kwargs)
    return wrapper


def tokens_from_streamlit_config(path) -> dict[str, str]:
    """The `key = "value"` pairs of the [theme] table in a Streamlit config.toml, read without
    a TOML dependency: enough for the test that pins these constants to that file."""
    out: dict[str, str] = {}
    in_theme = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):      # full-line comments only: the values are "#rrggbb"
            continue
        if line.startswith("["):
            in_theme = line == "[theme]"
            continue
        if in_theme and "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = v.strip('"')
    return out
