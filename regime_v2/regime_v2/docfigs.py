"""Documentation figures for the two site pages (docs/site/*.md via sitedocs.py).

Seven figures, drawn on every run from the `Published` run (labels frame +
summary dict) so the site never needs a separate build step. Same visual
language as figures.py: regime colours from `regimes.COLORS`, everything else
a light neutral palette, 130 dpi, Agg. A figure whose inputs are missing (the
asset stage skipped, or `--skip-placebo` dropped the null arrays) is skipped
and its slot comes back None -- write_doc_figures never raises.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from .regimes import COLORS, REGIMES, SIGNS  # noqa: E402

DPI = 130
DOC_FIGURES = ["doc_pipeline", "doc_quadrants", "doc_timing", "doc_lookahead", "doc_placebo",
               "doc_loadings", "doc_transition"]

INK = "#212529"
LINE = "#495057"
NEUTRAL = "#adb5bd"
NEUTRAL_LIGHT = "#eef2f6"
BLUE = "#1c7ed6"
ORANGE = "#e8590c"


# ---------------------------------------------------------------- box-and-arrow helpers

def _wrap(s: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(s, width=width)) if s else s


def _ordinal(x: float) -> str:
    n = int(round(x))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _stage_box(ax, x, y, w, h, title, detail=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=LINE, facecolor=NEUTRAL_LIGHT, zorder=2))
    if detail:
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.32, _wrap(detail, 22), ha="center", va="center", fontsize=7.0,
               color=LINE, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=INK, zorder=3)


def _hconnect(ax, a, b, w, h, color=LINE):
    """Arrow between two boxes at the same height; direction follows x order."""
    ax0, ay0 = a; bx0, by0 = b
    yc0, yc1 = ay0 + h / 2, by0 + h / 2
    p0, p1 = ((ax0 + w, yc0), (bx0, yc1)) if bx0 >= ax0 else ((ax0, yc0), (bx0 + w, yc1))
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3, zorder=1))


def _vconnect(ax, a, b, w, h, color=LINE):
    """Vertical drop from the bottom-centre of box a to the top-centre of box b."""
    ax0, ay0 = a; bx0, by0 = b
    p0, p1 = (ax0 + w / 2, ay0), (bx0 + w / 2, by0 + h)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.3, zorder=1))


def _gap_arrow(ax, x0, x1, y, text, above=True, color=LINE, fontsize=7.6, label_gap=0.3):
    """A short straight arrow spanning a gap between two boxes, with a label beside it."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3, zorder=1))
    ty = y + label_gap if above else y - label_gap
    ax.text((x0 + x1) / 2, ty, text, ha="center", va="center", fontsize=fontsize, color=color)


# --------------------------------------------------------------------- doc_pipeline

def _num(v, fallback):
    """A run parameter as a short string, falling back to the schematic's default."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(fallback)


def _draw_pipeline(pub, path) -> bool:
    # Every number in the schematic comes from the run it describes; the literals are only
    # fallbacks for a summary that predates the key (a --trend-window 180 run must not draw "240").
    S = pub.summary
    params = S.get("params") or {}
    loadings = S.get("loadings") or {}
    n_growth = len(loadings["growth"]) if isinstance(loadings.get("growth"), dict) else 22
    n_infl = len(loadings["inflation"]) if isinstance(loadings.get("inflation"), dict) else 13
    window = _num(params.get("window"), 240)
    lag = _num(params.get("publication_lag_months"), 1)
    universe = (S.get("assets") or {}).get("universe") or {}
    n_etf = len(universe) if universe else 11
    k_out = _num(params.get("k_outlier"), 10)
    stages = [
        ("FRED-MD vintage", "current vintage, McCracken-Ng"),
        ("Transforms & outliers", f"t-codes, {k_out}x IQR on estimation rows"),
        ("Growth & inflation blocks", f"{n_growth} growth series, {n_infl} inflation series"),
        ("One-factor PCA (EM)", "sign-anchored to INDPRO / CPIAUCSL"),
        ("Cumulated diffusion index", "inflation factor -> level"),
        ("One-sided trend gap", f"{window}-month trailing mean, expanding SD"),
        ("Constrained 4-state HMM", "filtered (real-time), symmetric emissions"),
        ("Labels + available_at", f"publication lag = {lag} month"),
        ("Asset tables & backtest", f"{n_etf}-ETF universe, PIT weights"),
    ]
    w, h, step = 1.55, 1.05, 1.8
    positions = {i: (i * step, 1.45) for i in range(5)}
    for k, i in enumerate(range(5, 9)):
        positions[i] = ((4 - k) * step, 0.0)

    fig, ax = plt.subplots(figsize=(14.5, 4.8))
    ax.set_xlim(-0.3, 4 * step + w + 0.3)
    ax.set_ylim(-0.3, 1.45 + h + 0.55)
    ax.axis("off")
    for i, (title, detail) in enumerate(stages):
        _stage_box(ax, *positions[i], w, h, title, detail)
    for i in range(4):
        _hconnect(ax, positions[i], positions[i + 1], w, h)
    _vconnect(ax, positions[4], positions[5], w, h)
    for i in range(5, 8):
        _hconnect(ax, positions[i], positions[i + 1], w, h)
    run = S.get("run", {})
    sub = f"vintage {run.get('vintage', '?')}, as of {run.get('asof', '?')}" if run else ""
    ax.set_title("regime_v2 pipeline: FRED-MD vintage to labels and backtest (spec §6)" +
                (f"\n{sub}" if sub else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_quadrants

def _draw_quadrants(pub, path) -> bool:
    current = pub.summary["current"]
    theta = float(pub.summary.get("params", {}).get("theta", 0.0))
    g = pub.labels["growth_gap"].dropna()
    p = pub.labels["inflation_gap"].dropna()
    cg, cp, month = float(current["growth_gap"]), float(current["inflation_gap"]), current["month"]
    lim = float(np.nanmax([g.abs().quantile(0.98) if len(g) else 1.0,
                           p.abs().quantile(0.98) if len(p) else 1.0,
                           abs(cg), abs(cp), 1.0])) * 1.2

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    for name, (sg, sp) in SIGNS.items():
        x0, x1 = (0.0, lim) if sg > 0 else (-lim, 0.0)
        y0, y1 = (0.0, lim) if sp > 0 else (-lim, 0.0)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=COLORS[name], alpha=0.14, lw=0, zorder=0))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center", fontsize=12,
               color=COLORS[name], fontweight="bold", alpha=0.9, zorder=1)
    if theta > 0:
        ax.axvspan(-theta, theta, color=NEUTRAL, alpha=0.25, lw=0, zorder=2)
        ax.axhspan(-theta, theta, color=NEUTRAL, alpha=0.25, lw=0, zorder=2)
    ax.axhline(0, color=LINE, lw=0.9, zorder=3); ax.axvline(0, color=LINE, lw=0.9, zorder=3)
    ax.scatter([cg], [cp], s=170, color="black", marker="*", zorder=5, edgecolor="white", linewidth=0.9)
    ax.annotate(month, (cg, cp), textcoords="offset points", xytext=(12, 12), fontsize=9.5, fontweight="bold",
               zorder=6, bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LINE, alpha=0.92))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Growth gap (SD)"); ax.set_ylabel("Inflation gap (SD)")
    ax.set_title(f"Growth-inflation plane: hysteresis quadrants (θ={theta:g}), current month marked", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_timing

def _draw_timing(pub, path) -> bool:
    lag = int(pub.summary.get("params", {}).get("publication_lag_months", 1))
    w, h, step = 1.3, 0.8, 2.1
    row1_y, row2_y = 2.6, 0.0
    row1 = {i: (i * step, row1_y) for i in range(4)}
    row2 = {i: (i * step, row2_y) for i in range(3)}

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(-1.5, 3 * step + w + 0.4)
    ax.set_ylim(-0.5, row1_y + h + 1.0)
    ax.axis("off")

    for i, t in zip(range(4), ["t−1", "t", "t+1", "t+2"]):
        _stage_box(ax, *row1[i], w, h, t)
    for i, d in zip(range(3), ["d−1", "d", "d+1"]):
        _stage_box(ax, *row2[i], w, h, d)

    cx_t = row1[1][0] + w / 2
    ax.annotate("data for t", xy=(cx_t, row1_y + h), xytext=(cx_t, row1_y + h + 0.45), ha="center",
               fontsize=8.3, color=LINE, arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2))

    gx0, gx1 = row1[1][0] + w, row1[2][0]
    _gap_arrow(ax, gx0, gx1, row1_y + h + 0.18, "label t published\n(available_at = 1st of t+1)", above=True)
    _gap_arrow(ax, gx1, gx0, row1_y - 0.18, "return of t+1 pairs\nwith label t", above=False)

    g2x0, g2x1 = row2[0][0] + w, row2[1][0]
    _gap_arrow(ax, g2x0, g2x1, row2_y + h + 0.18, f"decision at end of d\nuses label d−{lag}", above=True)
    g3x0, g3x1 = row2[1][0] + w, row2[2][0]
    _gap_arrow(ax, g3x0, g3x1, row2_y + h + 0.18, "weights earn d+1", above=True)

    ax.text(-0.35, row1_y + h / 2, "Descriptive\ntables", ha="right", va="center", fontsize=8.5, color=LINE, style="italic")
    ax.text(-0.35, row2_y + h / 2, "Backtest", ha="right", va="center", fontsize=8.5, color=LINE, style="italic")
    ax.set_title(f"Publication timing (D11): a month-t label is available at t+{lag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_lookahead

def _draw_lookahead(pub, path) -> bool:
    a = pub.summary["assets"]
    look = a["lookahead"]
    static = a["backtest"]["cost_bp_0"]["perf"]["Static_6040"]["sharpe"]
    ins, orc, pit = look["insample_sharpe"], look["oracle_sharpe"], look["pit_sharpe"]
    moment_la, label_la = look["moment_lookahead"], look["label_lookahead"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    labels = ["In-sample\n(ex-post)", "− moment\nlook-ahead", "Oracle\n(label look-ahead only)",
             "− label\nlook-ahead", "PIT\n(achievable)"]

    ax.bar(0, ins, color=BLUE, width=0.6, zorder=3)
    ax.bar(1, moment_la, bottom=orc, color=NEUTRAL, width=0.6, zorder=3, edgecolor=LINE, linewidth=0.8)
    ax.bar(2, orc, color=BLUE, width=0.6, zorder=3)
    ax.bar(3, label_la, bottom=pit, color=NEUTRAL, width=0.6, zorder=3, edgecolor=LINE, linewidth=0.8)
    ax.bar(4, pit, color=BLUE, width=0.6, zorder=3)

    for x, v in [(0, ins), (2, orc), (4, pit)]:
        ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(1, orc + moment_la / 2, f"{-moment_la:+.2f}", ha="center", va="center", fontsize=8.5)
    ax.text(3, pit + label_la / 2, f"{-label_la:+.2f}", ha="center", va="center", fontsize=8.5)
    for x0, x1, y in [(0.3, 0.7, ins), (1.3, 1.7, orc), (2.3, 2.7, orc), (3.3, 3.7, pit)]:
        ax.plot([x0, x1], [y, y], color=NEUTRAL, lw=0.8, ls=":")

    ax.axhline(static, color=LINE, ls="--", lw=1.3, zorder=2)
    ax.text(4.35, static, f"Static 60/40 (0 bp): {static:.2f}", va="center", ha="left", fontsize=8.5, color=LINE,
           zorder=4, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("annualised Sharpe (0 bp)")
    ax.set_title("Look-ahead decomposition: in-sample Sharpe is not achievable")
    ax.set_xlim(-0.6, 5.4)
    top = max(ins, static) * 1.15
    ax.set_ylim(min(0, pit, static) * 1.15, top)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_placebo

def _draw_placebo(pub, path) -> bool:
    a = pub.summary["assets"]
    ssp = a["sharpe_spread_placebo"]
    btp = a["backtest_placebo"]
    if btp is None:
        raise KeyError("backtest_placebo was skipped: no null draws to plot")
    null1 = np.asarray(ssp["null"], dtype=float)
    null2 = np.asarray(btp["null"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    specs = [(axes[0], null1, ssp["real"], ssp["percentile"], "Sharpe-spread placebo\n(max−min regime Sharpe, 60/40)"),
             (axes[1], null2, btp["real"], btp["percentile"], "Backtest placebo\n(PIT max-Sharpe)")]
    for ax, null, real, pct, title in specs:
        ax.hist(null, bins=30, color=NEUTRAL, edgecolor="white")
        ax.axvline(real, color=ORANGE, lw=2.2, label=f"real = {real:.2f}")
        ax.set_title(f"{title}\nreal value at the {_ordinal(pct)} percentile of {len(null)} shuffles", fontsize=9.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("statistic"); ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_loadings

def _draw_loadings(pub, path) -> bool:
    loadings = pub.summary["loadings"]
    growth = pd.Series(loadings["growth"]).sort_values()
    infl = pd.Series(loadings["inflation"]).sort_values()

    height = max(4.0, 0.32 * max(len(growth), len(infl)) + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(13, height))
    for ax, s, title in zip(axes, [growth, infl], ["Growth factor loadings", "Inflation factor loadings"]):
        colors = [BLUE if v >= 0 else ORANGE for v in s.to_numpy()]
        ax.barh(s.index, s.to_numpy(), color=colors)
        ax.axvline(0, color=LINE, lw=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlabel("loading")
    fig.suptitle("Factor loadings by series (sign-anchored to INDPRO / CPIAUCSL)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- doc_transition

def _draw_transition(pub, path) -> bool:
    tm = pd.DataFrame(pub.summary["transition_matrix"]).T.reindex(index=REGIMES, columns=REGIMES)
    dur = pub.summary["expected_duration_months"]
    if tm.isna().any().any():
        raise KeyError("transition matrix missing a regime row/column")

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(tm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_aspect("auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(REGIMES, rotation=25, ha="right", fontsize=9.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(REGIMES, fontsize=9.5)
    for tick, r in zip(ax.get_xticklabels(), REGIMES):
        tick.set_color(COLORS[r])
    for tick, r in zip(ax.get_yticklabels(), REGIMES):
        tick.set_color(COLORS[r])
    for i in range(4):
        for j in range(4):
            v = float(tm.iloc[i, j])
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                   color="white" if v > 0.5 else INK)
    for i, r in enumerate(REGIMES):
        d = float(dur.get(r, float("nan")))
        ax.text(4.05, i, f"E[dur] {d:.1f} mo", ha="left", va="center", fontsize=8.3, color=COLORS[r])
    ax.set_xlim(-0.5, 6.3)
    ax.set_ylabel("from"); ax.set_xlabel("to")
    ax.set_title("HMM transition matrix (full-sample fit) and expected regime duration", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="transition probability")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


_DRAWERS = {
    "doc_pipeline": _draw_pipeline,
    "doc_quadrants": _draw_quadrants,
    "doc_timing": _draw_timing,
    "doc_lookahead": _draw_lookahead,
    "doc_placebo": _draw_placebo,
    "doc_loadings": _draw_loadings,
    "doc_transition": _draw_transition,
}


def write_doc_figures(pub, figs_dir) -> dict[str, "Path | None"]:
    """Draw the seven documentation figures from a `Published` run.

    `pub` needs only `.labels` (the regime_labels frame) and `.summary` (the
    parsed summary.json dict) -- it is not required to be a `publish.Published`
    instance, just to duck-type it. A figure whose inputs are missing (the
    asset stage skipped, the placebo null arrays absent, etc.) is skipped: its
    slot in the returned dict is None and no exception escapes this function.
    """
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path | None] = {}
    for name in DOC_FIGURES:
        path = figs_dir / f"{name}.png"
        path.unlink(missing_ok=True)
        try:
            ok = _DRAWERS[name](pub, path)
        except Exception:
            ok = False
            plt.close("all")
        out[name] = path if ok else None
    return out
