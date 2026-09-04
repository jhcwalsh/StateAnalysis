"""Macro regime dashboard — a thin Streamlit viewer over regime_v2/output/ (spec §12).

Reads only through regime_v2.publish. Regime names and colours come from
regime_v2.regimes. The Refresh button runs the engine; a failed run leaves the
last published outputs on screen because the engine publishes atomically.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "regime_v2"
sys.path.insert(0, str(ENGINE))
from regime_v2 import publish, regimes as R   # noqa: E402  (path set above)


def _dir(env, default):
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


OUT_DIR = _dir("REGIME_OUTPUT_DIR", "regime_v2/output")
FIGS_DIR = _dir("REGIME_FIGS_DIR", "regime_v2/figs")
RETURNS_CACHE = _dir("REGIME_RETURNS_CACHE", "regime_v2/data/returns_yfinance.parquet")
PYTHON = os.environ.get("REGIME_PYTHON", sys.executable)
LOCK = OUT_DIR.parent / ".refresh.lock"

st.set_page_config(page_title="Macro Regime Dashboard", layout="wide")

# ---- chart style: theme-matched ink, transparent surfaces --------------------
FIG_W = 12
try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
INK = "#FAFAFA" if DARK else "#31333F"
INK_MUTED = "#A3A8B4" if DARK else "#808495"
SURFACE = "#0E1117" if DARK else "#FFFFFF"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": INK, "axes.titlesize": 10, "axes.edgecolor": INK_MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": INK_MUTED, "grid.alpha": 0.15 if DARK else 0.25, "grid.linewidth": 0.5,
})
GROWTH_C, INFL_C = "#2C7FB8", "#D95F0E"
CLIP_NOTE = " Extreme COVID-2020 observations lie beyond the visible range."


def _fig(height, width=FIG_W):
    return plt.subplots(figsize=(width, height))


@st.cache_data
def _load(mtime, out_dir, figs_dir):
    return publish.load_published(out_dir, figs_dir)


def _refresh_ui(button_label):
    vintage = st.text_input("FRED-MD vintage (YYYY-MM)", publish.default_vintage(),
                            help="The vintage file to download from the St. Louis Fed; previous month by default.")
    if st.button(button_label):
        cmd = publish.refresh_command(PYTHON, ENGINE / "run.py", vintage, OUT_DIR, FIGS_DIR, RETURNS_CACHE)
        with st.spinner("Running the engine (≈3 min: download, walk-forward, asset stage)…"):
            ok, tail = publish.run_refresh(cmd, str(ENGINE), LOCK)
        if ok:
            st.success("Refresh complete."); st.cache_data.clear(); st.rerun()
        st.error("Refresh failed — the previously published run is still shown. Log tail:")
        st.code(tail)


try:
    pub = _load(publish.published_mtime(OUT_DIR), str(OUT_DIR), str(FIGS_DIR))
except publish.PublishedMissing:
    st.title("Macro Regime Dashboard")
    st.markdown("**No published run found.** Run the engine once to publish `output/` and `figs/`.")
    _refresh_ui("Run the engine now")
    st.stop()

S, cur, run = pub.summary, pub.summary["current"], pub.summary["run"]
lab = pub.labels
theta_run = float(S["params"]["theta"])
hist_col = "hmm_walkforward" if lab["hmm_walkforward"].notna().any() else "hmm_filtered"

# ---------------- Zone 1: status header ----------------
st.title("Macro Regime Dashboard")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"<div style='background:{R.COLORS[cur['regime']]};color:white;padding:1.2em;border-radius:8px;"
        f"font-size:1.5em;font-weight:bold'>{cur['regime']} · {cur['month']}</div>", unsafe_allow_html=True)
    st.markdown(f"Growth gap **{cur['growth_gap']:+.2f}** · Inflation gap **{cur['inflation_gap']:+.2f}** "
                f"· quadrant rule says **{cur['quadrant']}** (θ = {theta_run:.2f})")
with c2:
    probs = pd.Series(cur["probs"]).reindex(R.REGIMES).fillna(0.0)
    fig, ax = _fig(1.6, width=7)
    lefts = probs.cumsum().shift(fill_value=0).values
    ax.barh([0] * len(R.REGIMES), probs.values, left=lefts, color=[R.COLORS[k] for k in R.REGIMES], height=0.5,
            edgecolor=SURFACE, linewidth=1.5)
    for k, p, l in zip(R.REGIMES, probs.values, lefts):
        if p > 0.08:
            ax.text(l + p / 2, 0, f"{k} {p:.0%}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1); ax.axis("off")
    st.pyplot(fig, clear_figure=True)

gate = "all acceptance tests passed" if S["acceptance_all_passed"] else "acceptance FAILED"
known = S.get("acceptance_known_failures") or {}
st.caption(f"Labels: {run['label_source']} · vintage {run['vintage']} · data through {run['asof'][:7]} · run {run['timestamp'][:16]} "
           f"· {gate}" + (f" · known failures: {', '.join(known)}" if known else ""))

# ---------------- Zone 2: explore (theta) ----------------
st.header("Explore: hysteresis θ")
theta = st.slider("θ (regime persistence band)", 0.0, 1.0, theta_run, 0.05,
                  help=f"Run value: {theta_run}. Live causal relabelling of the published gaps (spec D10).")
g, p = lab["growth_gap"].dropna(), lab["inflation_gap"].dropna()
live = R.quadrant_labels(g, p, theta)
lengths = live.groupby((live != live.shift()).cumsum()).size()   # one entry per run
m1, m2, m3 = st.columns(3)
m1.metric("Avg regime duration", f"{lengths.mean():.1f} mo")
m2.metric("Regime switches", int(len(lengths) - 1))
m3.metric("Months classified", len(live))
fig, ax = _fig(1.4)
for reg in [r for r in R.REGIMES if (live == r).any()]:
    mask = (live == reg).values
    ax.bar(live.index[mask], 1, width=32, color=R.COLORS[reg], label=reg)
ax.set_yticks([]); ax.margins(x=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4)
st.pyplot(fig, clear_figure=True)
occ = live.value_counts().reindex(R.REGIMES).fillna(0).astype(int).rename_axis("Regime").to_frame("Months")
occ["% Time"] = (occ["Months"] / len(live)).map("{:.1%}".format)
st.dataframe(occ, width="stretch")
st.info(f"The HMM labels, tables and backtest below are pinned to the run's θ = {theta_run}; the slider only relabels the quadrant rule.")

# ---------------- Zone 3: results tabs ----------------
st.header("Results")
(t_gap, t_lev, t_prob, t_ss, t_ret, t_corr, t_port, t_bt, t_acc, t_fig) = st.tabs(
    ["Factor gaps", "Factor levels", "Probabilities", "State space", "Regime returns", "Correlations",
     "Portfolios", "Backtest", "Acceptance", "Figures"])
NO_ASSETS = "The asset stage did not publish (skipped: {}). Re-run the engine with network access to fill this tab."
skipped = (S.get("assets") or {}).get("skipped", "not run")


def _robust_lim(*series, floor=2.0):
    v = pd.concat([s.abs() for s in series])
    lim = 1.2 * max(floor, float(v.quantile(0.98)))
    return lim, bool(v.max() > lim)


def _lines(ax, a, b, la, lb):
    ax.plot(lab.index, lab[a], color=GROWTH_C, linewidth=1.8, label=la)
    ax.plot(lab.index, lab[b], color=INFL_C, linewidth=1.8, label=lb)
    lim, clipped = _robust_lim(lab[a].dropna(), lab[b].dropna())
    ax.set_ylim(-lim, lim); ax.margins(x=0); ax.legend(loc="upper left"); ax.grid(True)
    return clipped


with t_gap:
    fig, ax = _fig(4)
    ax.axhspan(-theta, theta, color=INK, alpha=0.07, zorder=0); ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    clipped = _lines(ax, "growth_gap", "inflation_gap", "Growth gap", "Inflation gap")
    ax.set_title(f"Classification inputs (shaded: ±θ = {theta:.2f} dead band)", loc="left")
    st.pyplot(fig, clear_figure=True)
    st.caption("One-sided trend gaps (spec D4): each point uses data up to that month only." + (CLIP_NOTE if clipped else ""))

with t_lev:
    fig, ax = _fig(4)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    clipped = _lines(ax, "growth_factor", "inflation_factor", "Growth factor", "Inflation factor")
    ax.set_title("Underlying factor levels (cumulated diffusion indices)", loc="left")
    st.pyplot(fig, clear_figure=True)
    st.caption("Composite activity and inflation factors before the gap transformation." + (CLIP_NOTE if clipped else ""))

with t_prob:
    pr = lab[[f"p_{r}" for r in R.REGIMES]].dropna().clip(lower=0)
    pr.columns = R.REGIMES
    fig, ax = _fig(4)
    ax.stackplot(pr.index, [pr[k].values for k in R.REGIMES], colors=[R.COLORS[k] for k in R.REGIMES],
                 labels=R.REGIMES, edgecolor=SURFACE, linewidth=0.8)
    ax.set_ylim(0, 1); ax.margins(x=0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4)
    st.pyplot(fig, clear_figure=True)
    st.caption(f"Constrained 4-state HMM probabilities, {run['label_source']}: causal at every month, never revised by later data.")
    with st.expander("View data"):
        st.dataframe(pr.style.format("{:.1%}"), width="stretch")

with t_ss:
    ss = pd.concat([g, p], axis=1).dropna(); ss.columns = ["growth_gap", "inflation_gap"]
    fig, ax = _fig(6, width=6.5)
    ax.axvspan(-theta, theta, color=INK, alpha=0.06, zorder=0); ax.axhspan(-theta, theta, color=INK, alpha=0.06, zorder=0)
    ax.axhline(0, color=INK_MUTED, linewidth=0.8); ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    for reg in [r for r in R.REGIMES if (live == r).any()]:
        m = (live.reindex(ss.index) == reg).values
        ax.scatter(ss["growth_gap"][m], ss["inflation_gap"][m], s=16, color=R.COLORS[reg], edgecolors=SURFACE,
                   linewidths=0.5, alpha=0.9, label=reg)
    last = ss.iloc[-1]
    ax.scatter([last["growth_gap"]], [last["inflation_gap"]], s=130, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=5)
    ax.annotate(ss.index[-1].strftime("%Y-%m"), (last["growth_gap"], last["inflation_gap"]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Growth gap"); ax.set_ylabel("Inflation gap")
    lim, clipped = _robust_lim(ss["growth_gap"], ss["inflation_gap"])
    lim = max(lim, 1.15 * abs(last["growth_gap"]), 1.15 * abs(last["inflation_gap"]))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.11), ncol=2)
    st.pyplot(fig, clear_figure=True, width="content")
    st.caption("Each point is one month, coloured by the live θ quadrant labels from the slider. Points inside the band keep "
               "their previous regime (hysteresis)." + (CLIP_NOTE if clipped else ""))
    if pub.figures["fig3_state_space"]:
        with st.expander("Published run version (fig 3, HMM labels)"):
            st.image(str(pub.figures["fig3_state_space"]))

with t_ret:
    if pub.regime_returns is None:
        st.info(NO_ASSETS.format(skipped))
    else:
        tbl = pub.regime_returns.copy()
        st.dataframe(tbl.style.format({"ann_ret": "{:+.1%}", "ann_vol": "{:.1%}", "sharpe": "{:+.2f}", "maxdd": "{:.1%}",
                                       "hit": "{:.0%}", "se_ann_ret": "{:.1%}", "se_sharpe": "{:.2f}", "n": "{:.0f}"}), width="stretch")
        st.caption("Monthly returns of month r paired with the regime label available before r (spec D11). Standard errors "
                   "from a 12-month block bootstrap of the whole aligned panel. maxdd chains within-regime months.")
        if pub.figures["fig8_regime_returns"]:
            st.image(str(pub.figures["fig8_regime_returns"]))

with t_corr:
    if not pub.corr:
        st.info(NO_ASSETS.format(skipped))
    for reg in [r for r in R.REGIMES if r in pub.corr]:
        st.subheader(reg)
        st.dataframe(pub.corr[reg].style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")

with t_port:
    a = S.get("assets") or {}
    if a.get("skipped") is not None or "backtest" not in a:
        st.info(NO_ASSETS.format(skipped))
    else:
        for key, title in [("cost_bp_0", "No transaction costs"), ("cost_bp_10", "10 bp per unit turnover")]:
            st.subheader(title)
            perf = pd.DataFrame(a["backtest"][key]["perf"]).T
            st.dataframe(perf.style.format({"ann_ret": "{:+.1%}", "ann_vol": "{:.1%}", "sharpe": "{:+.2f}",
                                            "maxdd": "{:.1%}", "turnover": "{:.2f}"}), width="stretch")
        st.caption("PIT_* use only labels available at the decision date; Oracle uses the ex-post smoothed labels; "
                   "InSample_MaxSharpe_expost also uses full-sample moments and is not achievable. Static_6040 is the benchmark.")
        st.caption(f"Fallbacks and guards: {a['backtest']['cost_bp_0']['counters']}")
        if pub.portfolio_weights is not None and pub.figures["fig11_pit_weights"]:
            st.image(str(pub.figures["fig11_pit_weights"]))

with t_bt:
    a = S.get("assets") or {}
    if pub.backtest_returns is None or "lookahead" not in a:
        st.info(NO_ASSETS.format(skipped))
    else:
        wealth = (1 + pub.backtest_returns).cumprod()
        fig, ax = _fig(5)
        for col in wealth.columns:
            style = dict(linewidth=1.0, linestyle="--", alpha=0.7) if col.endswith("_expost") else dict(linewidth=2.0 if col.startswith("PIT") else 1.2)
            ax.plot(wealth.index, wealth[col], label=col, **style)
        ax.set_yscale("log"); ax.margins(x=0); ax.legend(); ax.grid(True)
        st.pyplot(fig, clear_figure=True)
        L, bp = a["lookahead"], a["backtest_placebo"]
        perf0 = pd.DataFrame(a["backtest"]["cost_bp_0"]["perf"]).T
        st.markdown(
            f"**Look-ahead decomposition** — in-sample Sharpe {L['insample_sharpe']:+.2f} → oracle {L['oracle_sharpe']:+.2f} "
            f"→ achievable (PIT) {L['pit_sharpe']:+.2f}; moment look-ahead {L['moment_lookahead']:+.2f}, label look-ahead "
            f"{L['label_lookahead']:+.2f}. Static_6040 over the same window: {perf0.loc['Static_6040', 'sharpe']:+.2f}.")
        if bp:
            st.markdown(f"**Backtest placebo** — the real PIT Sharpe sits at the {bp['percentile']:.0f}th percentile of {bp['n']} "
                        "run-preserving label shuffles; below 50 means the real labels underperform the median shuffle.")
        sp = a.get("sharpe_spread_placebo")
        if sp:
            st.markdown(f"**Sharpe-spread placebo** — max-minus-min regime Sharpe of 60/40 at the {sp['percentile']:.0f}th percentile.")
        gs = a.get("growth_share_6040")
        if gs:
            st.caption(f"Growth share of the 60/40 regime regression: {gs['growth_share']:.0%} of an R² of {gs['r2']:.3f} "
                       f"(n = {gs['n']}); with an R² this small the split is not informative.")

with t_acc:
    acc = pub.acceptance.copy()
    acc["status"] = acc.apply(lambda r: "report" if r["op"] == "report" else ("known failure" if r["known_failure"] else ("pass" if r["passed"] else "FAIL")), axis=1)
    st.dataframe(acc[["name", "value", "op", "threshold", "status", "rationale"]], width="stretch", hide_index=True)
    st.caption("Spec §8. Known failures are declared in the code with their mechanism and never block publishing; "
               "report rows have no threshold.")

with t_fig:
    for name in publish.FIGURES:
        path = pub.figures[name]
        if path is None:
            st.info(f"{name}.png not published.")
            continue
        st.subheader(name)
        st.image(str(path))
        st.download_button(f"Download {name}.png", data=path.read_bytes(), file_name=f"{name}.png", mime="image/png", key=f"dl_{name}")
    st.subheader("Data downloads")
    for key in ["labels", "acceptance", "regime_returns", "backtest_returns", "portfolio_weights"]:
        f = pub.out_dir / publish.FILES[key]
        if f.exists():
            st.download_button(f"Download {f.name}", data=f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{key}")

# ---------------- Zone 4: refresh ----------------
st.header("Refresh")
st.caption("Downloads the FRED-MD vintage and the ETF returns, reruns the engine and the walk-forward, re-evaluates the "
           "acceptance tests and publishes. If any blocking test fails the previous outputs stay live.")
_refresh_ui("Refresh data (≈3 min)")
