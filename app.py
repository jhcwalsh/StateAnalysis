"""Regime dashboard — thin Streamlit viewer over ui_data/ (see design spec)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import ui_io
from regime_core import (
    assign_quadrants, duration_stats,
    CODE_TO_PAPER, PAPER_DISPLAY, PAPER_COLORS, PAPER_ORDER,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.environ.get("UI_DATA_DIR", os.path.join(PROJECT_DIR, "ui_data"))

st.set_page_config(page_title="Macro Regime Dashboard", layout="wide")


@st.cache_data
def _bundle(mtime):
    return ui_io.load_bundle(UI_DIR)


def _meta_mtime():
    p = os.path.join(UI_DIR, "meta.json")
    return os.path.getmtime(p) if os.path.exists(p) else 0


def _paper(code_label):
    return PAPER_DISPLAY.get(CODE_TO_PAPER.get(code_label, code_label), code_label)


def _paper_color(code_label):
    return PAPER_COLORS.get(CODE_TO_PAPER.get(code_label, code_label), "#888888")


try:
    bundle = _bundle(_meta_mtime())
except ui_io.SchemaError as e:
    st.error(f"ui_data/ is out of date: {e}")
    st.stop()

if bundle is None:
    st.title("Macro Regime Dashboard")
    st.markdown("**No cached run found** — press Refresh to run the notebook (≈5 min).")
    if st.button("Refresh now"):
        with st.spinner("Executing notebook…"):
            ok, tail = ui_io.run_refresh(PROJECT_DIR)
        if ok:
            st.cache_data.clear(); st.rerun()
        st.code(tail)
    st.stop()

meta, cur = bundle["meta"], bundle["meta"]["current"]

# ---------------- Zone 1: status header ----------------
st.title("Macro Regime Dashboard")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(
        f"<div style='background:{_paper_color(cur['quadrant'])};color:white;"
        f"padding:1.2em;border-radius:8px;font-size:1.5em;font-weight:bold'>"
        f"{_paper(cur['quadrant'])}</div>", unsafe_allow_html=True)
    st.markdown(
        f"Growth gap **{cur['growth_gap']:+.2f}** · Inflation gap "
        f"**{cur['inflation_gap']:+.2f}** (θ band ±{meta['theta']:.2f})")
with c2:
    probs = pd.Series(cur["probs"]).reindex(PAPER_ORDER)
    fig, ax = plt.subplots(figsize=(7, 1.6))
    lefts = probs.cumsum().shift(fill_value=0).values
    ax.barh([0] * 4, probs.values, left=lefts,
            color=[PAPER_COLORS[k] for k in PAPER_ORDER], height=0.5,
            edgecolor="white", linewidth=1.5)
    for k, p, l in zip(PAPER_ORDER, probs.values, lefts):
        if p > 0.08:
            ax.text(l + p / 2, 0, f"{PAPER_DISPLAY[k].split(':')[0]} {p:.0%}",
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, 1); ax.axis("off")
    st.pyplot(fig, clear_figure=True)

st.caption(
    f"FRED through {meta['sample']['end'][:7]} · run {meta['run_timestamp'][:16]} "
    f"· PIT/final agreement {meta['pit']['agreement']:.1%} "
    f"({meta['pit']['n_common']} months)")

# ---------------- Zone 2: explore (theta) ----------------
st.header("Explore: hysteresis θ")
theta = st.slider("θ (regime persistence band)", 0.0, 1.0,
                  float(meta["theta"]), 0.05,
                  help=f"Run value: {meta['theta']}. Live relabeling from cached gaps.")
gaps = bundle["gaps"]
live = assign_quadrants(gaps["g_gap"].dropna(), gaps["p_gap"].dropna(), theta)
avg, switches = duration_stats(live)

m1, m2, m3 = st.columns(3)
m1.metric("Avg regime duration", f"{avg:.1f} mo")
m2.metric("Regime switches", switches)
m3.metric("Months classified", len(live))

fig, ax = plt.subplots(figsize=(14, 1.4))
for code in live.unique():
    mask = (live == code).values
    ax.bar(live.index[mask], 1, width=32, color=_paper_color(code),
           label=_paper(code), edgecolor="white", linewidth=0.3)
ax.set_yticks([]); ax.margins(x=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4, fontsize=8,
          frameon=False)
st.pyplot(fig, clear_figure=True)

occ = live.value_counts().rename(_paper).rename_axis("Regime").to_frame("Months")
occ["% Time"] = (occ["Months"] / len(live)).map("{:.1%}".format)
st.dataframe(occ, width="stretch")
st.info(f"Tables and backtest below are pinned to run θ = {meta['theta']}. "
        "Change HYSTERESIS_THETA in the notebook and press Refresh to recompute them.")

# ---------------- Zone 3: results tabs ----------------
st.header("Results")
t1, t2, t3, t4, t5 = st.tabs(
    ["Regime returns", "Correlations", "Portfolios", "State space", "Backtest"])
xl = pd.ExcelFile(bundle["tables_path"])
with t1:
    if "Table1_RegimeReturns" in xl.sheet_names:
        st.dataframe(pd.read_excel(xl, sheet_name="Table1_RegimeReturns"),
                     use_container_width=True)
    else:
        st.info("Table1_RegimeReturns not found in tables.xlsx — "
                 "run the notebook to regenerate ui_data/.")
with t2:
    for sheet in [s for s in xl.sheet_names if s.startswith("T2_")]:
        st.subheader(_paper(sheet[3:]) if sheet[3:] in CODE_TO_PAPER else sheet)
        st.dataframe(pd.read_excel(xl, sheet_name=sheet), width="stretch")
with t3:
    if "Table3_Portfolios" in xl.sheet_names:
        st.dataframe(pd.read_excel(xl, sheet_name="Table3_Portfolios"),
                     use_container_width=True)
    else:
        st.info("Table3_Portfolios not found in tables.xlsx — "
                 "run the notebook to regenerate ui_data/.")
with t4:
    png = os.path.join(PROJECT_DIR, "state_space_regimes.png")
    if os.path.exists(png):
        st.image(png)
    else:
        st.info("state_space_regimes.png not found — run the notebook (Cell H).")
with t5:
    bt = bundle["backtest"]
    wealth = (1 + bt).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in wealth.columns:
        ax.plot(wealth.index, wealth[col],
                linewidth=2.0 if col.startswith("PIT") else 1.2, label=col)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, color="gray", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    st.pyplot(fig, clear_figure=True)
    perf = pd.DataFrame(meta["backtest"]["perf"]).T
    st.dataframe(perf.style.format("{:+.3f}"), width="stretch")
    if "Table3_Portfolios" in xl.sheet_names:
        t3df = pd.read_excel(xl, sheet_name="Table3_Portfolios")
        is_row = t3df[t3df["Regime"].str.startswith("Prob-Weighted")
                      & (t3df["Objective"] == "Max Sharpe")]
        if len(is_row):
            st.markdown(
                f"**Look-ahead decomposition** — in-sample Sharpe "
                f"{float(is_row['Port_Sharpe'].iloc[0]):+.2f} → oracle "
                f"{perf.loc['Oracle_MaxSharpe', 'Sharpe']:+.2f} → achievable (PIT) "
                f"{perf.loc['PIT_MaxSharpe', 'Sharpe']:+.2f} "
                f"vs 60/40 {perf.loc['Static_6040', 'Sharpe']:+.2f}")

# ---------------- Zone 4: refresh ----------------
st.header("Refresh")
st.caption("Re-executes the notebook (~5 min): downloads FRED/Yahoo, recomputes "
           "everything, rewrites ui_data/. Previous data stays live if it fails.")
if st.session_state.get("refreshing"):
    st.warning("A refresh is already running.")
elif st.button("Refresh data (≈5 min)"):
    st.session_state["refreshing"] = True
    try:
        with st.spinner("Executing notebook…"):
            ok, tail = ui_io.run_refresh(PROJECT_DIR)
    finally:
        st.session_state["refreshing"] = False
    if ok:
        st.success("Refresh complete."); st.cache_data.clear(); st.rerun()
    else:
        st.error("Refresh failed — previous data still shown. Log tail:")
        st.code(tail)
