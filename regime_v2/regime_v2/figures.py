"""Figures 1-7 (spec §7). 130 dpi, Agg, NBER shading, regime colours from regimes.COLORS.

Challenger strips use their own names with a tab10 palette, never the
regime palette. Anything smoothed is captioned "smoothed (ex-post)".
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .nber import NBER  # noqa: E402
from .regimes import COLORS, REGIMES  # noqa: E402
from .trend import centred_trend_expost, revision_stats  # noqa: E402

DPI = 130
LOW_GROWTH = ["Contraction", "Stagflation"]


def _shade(ax):
    for a, b in NBER:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), color="grey", alpha=0.18, lw=0)


def _palette(names):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names)}


def _strip(ax, labels, colors, title):
    labels = labels.dropna()
    for name, col in colors.items():
        m = (labels == name).to_numpy()
        if m.any():
            ax.fill_between(labels.index, 0, 1, where=m, color=col, step="mid", lw=0, label=name)
    _shade(ax); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    ax.legend(ncol=len(colors), fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.0))


def _primary(res, wf):
    if wf is not None:
        return wf.labels_rt, wf.probs_rt, "walk-forward (real-time)"
    return res.hmm.labels_filtered, res.hmm.probs_filtered, "full-sample fit, filtered (NOT real-time)"


def fig1_factors_gaps(res, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, gp, name in zip(axes, [res.g_gap, res.p_gap], ["Growth", "Inflation"]):
        ax.plot(gp.index, gp["level"], lw=1, label=f"{name} index", color="#1c7ed6")
        ax.plot(gp.index, gp["trend"], lw=1.6, label="One-sided trend", color="#e8590c")
        ax2 = ax.twinx()
        ax2.plot(gp.index, gp["gap"], lw=1, color="#2b8a3e", label="Gap (SD, real-time)")
        ax2.axhline(0, color="grey", ls="--", lw=0.8); ax2.set_ylabel("gap (SD)")
        _shade(ax); ax.set_title(f"{name}: level, one-sided trend, quasi-real-time gap")
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig2_regime_timeline(res, wf, free, gmm, path):
    lab, _, how = _primary(res, wf)
    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    _strip(axes[0], res.quadrant, COLORS, f"Quadrants with hysteresis (θ={res.params['theta']})")
    _strip(axes[1], lab, COLORS, f"Constrained HMM, {how} (primary)")
    _strip(axes[2], free.labels_filtered, _palette(free.labels_filtered.unique()), "Free HMM (challenger, own state names)")
    _strip(axes[3], gmm.labels, _palette(gmm.cluster_names), "GMM (challenger, own cluster names)")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig3_state_space(res, wf, path):
    lab, _, how = _primary(res, wf)
    df = pd.concat([res.G, res.P, lab.rename("r")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for r in REGIMES:
        d = df[df["r"] == r]
        ax.scatter(d["growth_gap"], d["inflation_gap"], s=10, color=COLORS[r], label=f"{r} (n={len(d)})", alpha=0.7)
    ax.axhline(0, color="grey", ls="--", lw=0.8); ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Growth gap (SD, real-time)"); ax.set_ylabel("Inflation gap (SD, real-time)")
    ax.set_title(f"Growth–inflation state space, HMM labels: {how}"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig4_hmm_probabilities(res, wf, path):
    _, probs, how = _primary(res, wf)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, pr, title in zip(axes, [probs, res.hmm.probs_smoothed_expost],
                             [f"Filtered probabilities, {how}", "Smoothed (ex-post) probabilities — comparison only"]):
        ax.stackplot(pr.index, [pr[r] for r in REGIMES], colors=[COLORS[r] for r in REGIMES], labels=REGIMES, lw=0)
        _shade(ax); ax.set_ylim(0, 1); ax.set_title(title, fontsize=9)
    axes[0].legend(ncol=4, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def fig5_revisions(res, path) -> dict:
    expost = centred_trend_expost(res.growth_factor["factor"], smooth=res.params["smooth"], window=res.params["window"])
    rev = revision_stats(res.G, expost)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(res.G.index, res.G, lw=0.9, label="Quasi-real-time (one-sided)")
    ax.plot(expost.index, expost, lw=0.9, alpha=0.8, label="Ex-post (two-sided trend, full-sample std)")
    ax.axhline(0, color="grey", ls="--", lw=0.8); _shade(ax)
    ax.set_title(f"Growth gap revisions — corr {rev['corr_first_final']:.2f}, N/S {rev['noise_to_signal_rmse']:.2f}, sign agreement {rev['sign_agreement']:.0%}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
    return rev


def fig6_classifier_comparison(res, free, gmm, path):
    panels = [("Quadrants (θ=0)", res.quadrant0, COLORS),
              ("Constrained HMM, filtered", res.hmm.labels_filtered, COLORS),
              ("Free HMM (own names)", free.labels_filtered, _palette(free.labels_filtered.unique())),
              ("GMM (own names)", gmm.labels, _palette(gmm.cluster_names))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, (title, lab, cols) in zip(axes.ravel(), panels):
        df = pd.concat([res.G, res.P, lab.rename("r")], axis=1).dropna()
        for name, col in cols.items():
            d = df[df["r"] == name]
            if len(d):
                ax.scatter(d["growth_gap"], d["inflation_gap"], s=8, color=col, alpha=0.7, label=f"{name} ({len(d)})")
        ax.axhline(0, color="grey", ls="--", lw=0.8); ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)


def nber_lags(labels_rt: pd.Series) -> pd.DataFrame:
    """First real-time low-growth call after each NBER peak.

    `censored` is True when the label series starts after the peak, so the
    lag is only a lower bound (the regime may have been called before the
    window opened).
    """
    rows = []
    start = labels_rt.index[0] if len(labels_rt) else pd.NaT
    for peak, trough in NBER:
        pk = pd.Timestamp(peak)
        after = labels_rt[(labels_rt.index >= pk) & (labels_rt.index <= pd.Timestamp(trough) + pd.DateOffset(months=12))]
        hit = after[after.isin(LOW_GROWTH)]
        first = hit.index[0] if len(hit) else pd.NaT
        lag = (first.year - pk.year) * 12 + (first.month - pk.month) if first is not pd.NaT else np.nan
        censored = bool(start is not pd.NaT and start > pk and first is not pd.NaT)
        rows.append(dict(peak=peak, first_low_growth_rt=None if first is pd.NaT else first.strftime("%Y-%m"),
                         lag_months=lag, censored=censored))
    return pd.DataFrame(rows)


def fig7_walkforward(res, wf, path) -> pd.DataFrame:
    lags = nber_lags(wf.labels_rt)
    agree = (wf.labels_rt == res.hmm.labels_smoothed_expost.reindex(wf.labels_rt.index)).mean()
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    _strip(axes[0], wf.labels_rt, COLORS, "Walk-forward filtered label (real-time)")
    _strip(axes[1], res.hmm.labels_smoothed_expost.reindex(wf.labels_rt.index), COLORS, "Full-sample smoothed label (ex-post)")
    for _, r in lags.dropna(subset=["lag_months"]).iterrows():
        tag = f"{'≥' if r['censored'] else ''}+{int(r['lag_months'])}m"
        axes[0].annotate(tag, (pd.Timestamp(r["first_low_growth_rt"]), 1.02), fontsize=7, ha="center")
    fig.suptitle(f"Real-time vs ex-post labels — month-level agreement {agree:.0%}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
    return lags
