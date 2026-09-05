"""Render the two site documents (docs/site/introduction.md, docs/site/methodology.md)
against a published run, per the contract in docs/site/CONTRACT.md.

Three entry points:
  numbers(pub)                              -> {{key}} -> pre-formatted string
  render(markdown_text, nums, figures)      -> ordered list of ("md", text) / ("fig", path, caption, name)
  to_html(markdown_text, nums, figures, title) -> one self-contained HTML document
  missing_placeholders(markdown_text, nums) -> unknown {{key}} references in the text

Reads only through the `Published` object (regime_v2.publish) and its .summary /.labels /
.acceptance / .figures — no file access of its own beyond that.
"""
from __future__ import annotations

import base64
import html as _html
import math
import re
import uuid
from pathlib import Path

from . import regimes as R
from .data import GROWTH_BLOCK, INFLATION_BLOCK

try:
    import markdown as _markdown_lib
except ImportError:  # pragma: no cover - the caller is responsible for installing it
    _markdown_lib = None

REGIMES = R.REGIMES
STRATEGIES = ["PIT_MaxSharpe", "PIT_MinVar", "ProbWeighted_MaxSharpe", "Oracle_MaxSharpe",
              "Static_6040", "EqualWeight", "InSample_MaxSharpe_expost"]
PERF_COLS = ["ann_ret", "ann_vol", "sharpe", "maxdd", "turnover"]
PERF_FMT = {"ann_ret": "{:+.1%}", "ann_vol": "{:.1%}", "sharpe": "{:+.2f}", "maxdd": "{:.1%}", "turnover": "{:.2f}"}

_NA = "n/a"


# ---------------------------------------------------------------------------
# small formatters
# ---------------------------------------------------------------------------

def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _ym(d) -> str:
    """A date (Timestamp, 'YYYY-MM-DD', or similar) -> 'YYYY-MM'."""
    if d is None:
        return _NA
    return str(d)[:7]


def _month_long(ym: str) -> str:
    """'YYYY-MM' -> 'Month YYYY'."""
    from datetime import datetime
    try:
        return datetime.strptime(ym[:7], "%Y-%m").strftime("%B %Y")
    except (ValueError, TypeError):
        return _NA


def _pct(x, dp: int = 0) -> str:
    if x is None or _is_nan(x):
        return _NA
    return f"{float(x):.{dp}%}"


def _signed(x, dp: int = 2) -> str:
    if x is None or _is_nan(x):
        return _NA
    return f"{float(x):+.{dp}f}"


def _num(x, dp: int = 2) -> str:
    """Fixed-precision, unsigned-format (negative numbers still show '-')."""
    if x is None or _is_nan(x):
        return _NA
    f = float(x)
    if f == int(f):
        return str(int(f))
    return f"{f:.{dp}f}"


def _trim(x) -> str:
    """Shortest reasonable representation of a parameter value: 0.5 -> '0.5', 10.0 -> '10'."""
    if x is None or _is_nan(x):
        return _NA
    f = float(x)
    if f == int(f):
        return str(int(f))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def _g(x) -> str:
    """General-format numeric, for values spanning many orders of magnitude (acceptance rows)."""
    if x is None:
        return _NA
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if _is_nan(f):
        return _NA
    if f == int(f) and abs(f) < 1e6:
        return str(int(f))
    return f"{f:.4g}"


def _ordinal(x) -> str:
    """A percentile as an English ordinal: 31 -> '31st', 12 -> '12th' (docfigs._ordinal's rule)."""
    if x is None or _is_nan(x):
        return _NA
    n = int(round(float(x)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _direction(pct) -> str:
    if pct is None or _is_nan(pct):
        return _NA
    return "below" if float(pct) < 50 else "above"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _md_transition_table(tm: dict) -> str:
    rows = []
    for r in REGIMES:
        row = tm.get(r, {})
        rows.append([r] + [f"{float(row.get(c, float('nan'))):.2f}" if c in row else _NA for c in REGIMES])
    return _md_table(["from \\ to"] + REGIMES, rows)


def _md_acc_table(acc_df) -> str:
    rows = []
    for _, r in acc_df.iterrows():
        if r["op"] == "report":
            continue
        if r.get("known_failure") and not r["passed"]:
            status = "known failure"
        else:
            status = "pass" if r["passed"] else "FAIL"
        rows.append([str(r["name"]), _g(r["value"]), str(r["op"]), _g(r["threshold"]), status])
    return _md_table(["name", "value", "op", "threshold", "status"], rows)


def _md_perf_table(perf: dict) -> str:
    rows = []
    for strat in STRATEGIES:
        p = perf.get(strat)
        if p is None:
            rows.append([strat] + [_NA] * len(PERF_COLS))
        else:
            rows.append([strat] + [PERF_FMT[c].format(p[c]) for c in PERF_COLS])
    return _md_table(["strategy"] + PERF_COLS, rows)


def _fmt_counters(counters: dict) -> str:
    if not counters:
        return _NA
    return ", ".join(f"{k} {v}" for k, v in counters.items())


# ---------------------------------------------------------------------------
# numbers()
# ---------------------------------------------------------------------------

def numbers(pub) -> dict[str, str]:
    S = pub.summary
    run = S.get("run") or {}
    cur = S.get("current") or {}
    params = S.get("params") or {}
    labels = pub.labels
    acc = pub.acceptance
    # A summary with no "assets" block at all (`--no-assets`, or a run published before the
    # asset stage existed) is a skipped stage, not a successful one: `.get("skipped")` on an
    # absent block returns None, which is the *success* sentinel, and would leave every
    # `<!-- if:assets -->` section in place filled with "n/a".
    raw_assets = S.get("assets")
    assets_blk = raw_assets if isinstance(raw_assets, dict) else {}
    skipped_reason = assets_blk.get("skipped") if isinstance(raw_assets, dict) else "asset stage not run"

    out: dict[str, str] = {}

    # -- run.* --------------------------------------------------------------
    vintage = run.get("vintage") or ""
    out["run.vintage"] = vintage or _NA
    m = re.search(r"(\d{4}-\d{2})", vintage)
    out["run.vintage_month"] = m.group(1) if m else _NA
    asof = run.get("asof")
    out["run.asof"] = _ym(asof) if asof else _NA
    out["run.asof_long"] = _month_long(_ym(asof)) if asof else _NA
    ts = run.get("timestamp") or ""
    out["run.date"] = ts[:10] if ts else _NA
    out["run.label_source"] = run.get("label_source") or _NA

    # -- current.* ------------------------------------------------------------
    out["current.regime"] = cur.get("regime") or _NA
    out["current.month_long"] = _month_long(cur["month"]) if cur.get("month") else _NA
    probs = cur.get("probs") or {}
    for r in REGIMES:
        out[f"current.prob_{r}"] = _pct(probs.get(r))
    out["current.growth_gap"] = _signed(cur.get("growth_gap"))
    out["current.inflation_gap"] = _signed(cur.get("inflation_gap"))
    out["current.quadrant"] = cur.get("quadrant") or _NA

    # -- sample.* -------------------------------------------------------------
    if labels is not None and len(labels):
        out["sample.start"] = _ym(labels.index[0])
        out["sample.n_months"] = str(len(labels))
    else:
        out["sample.start"] = _NA
        out["sample.n_months"] = _NA
    wf = S.get("walkforward")
    if wf:
        out["sample.wf_start"] = _ym(wf.get("start"))
        out["sample.wf_n"] = str(wf.get("n_months", _NA))
    else:
        out["sample.wf_start"] = _NA
        out["sample.wf_n"] = _NA

    # -- params.* -------------------------------------------------------------
    out["params.window"] = _trim(params.get("window"))
    out["params.theta"] = f"{float(params['theta']):.2f}" if "theta" in params else _NA
    out["params.persistence"] = _trim(params.get("persistence"))
    out["params.eps"] = _trim(params.get("eps"))
    out["params.lag"] = _trim(params.get("publication_lag_months"))
    mask = params.get("mask")
    out["params.mask"] = f"{_ym(mask[0])} to {_ym(mask[1])}" if mask and len(mask) == 2 else _NA
    out["params.k_outlier"] = _trim(params.get("k_outlier"))

    # -- panel.* ----------------------------------------------------------------
    out["panel.n_growth"] = str(len(GROWTH_BLOCK))
    out["panel.n_inflation"] = str(len(INFLATION_BLOCK))

    # -- hmm.* --------------------------------------------------------------
    exp_dur = S.get("expected_duration_months") or {}
    mean_run = S.get("mean_run_length_months") or {}
    counts = S.get("regime_counts") or {}
    total_counts = sum(counts.values()) if counts else 0
    for r in REGIMES:
        out[f"hmm.expected_duration_{r}"] = f"{float(exp_dur[r]):.1f} months" if r in exp_dur else _NA
        out[f"hmm.mean_run_{r}"] = f"{float(mean_run[r]):.1f} months" if r in mean_run else _NA
        out[f"hmm.share_{r}"] = _pct(counts.get(r, 0) / total_counts) if total_counts else _NA
    tm = S.get("transition_matrix")
    out["hmm.transition_table"] = _md_transition_table(tm) if tm else _NA
    out["hmm.filtered_vs_smoothed"] = _pct(S.get("filtered_vs_smoothed_agreement"))
    smp = S.get("share_max_prob_gt_095") or {}
    out["hmm.max_prob_share"] = _pct(smp.get("primary"))

    # -- acc.* ------------------------------------------------------------------
    if acc is not None and len(acc):
        thresholded = acc[acc["op"] != "report"]
        out["acc.n_tests"] = str(len(thresholded))
        out["acc.n_passed"] = str(int(thresholded["passed"].sum()))
        out["acc.table"] = _md_acc_table(acc)
        for _, row in acc.iterrows():
            out[f"acc.{row['name']}"] = _g(row["value"])
    else:
        out["acc.n_tests"] = "0"
        out["acc.n_passed"] = "0"
        out["acc.table"] = _NA
    known = S.get("acceptance_known_failures") or {}
    out["acc.known_failures"] = ", ".join(known.keys()) if known else "none"

    # -- nber.* -------------------------------------------------------------
    lags = S.get("nber_lags_rt") or []
    uncensored = [row["lag_months"] for row in lags
                  if not row.get("censored") and row.get("lag_months") is not None and not _is_nan(row.get("lag_months"))]
    if uncensored:
        srt = sorted(float(x) for x in uncensored)
        mean_lag = sum(srt) / len(srt)
        mid = len(srt) // 2
        median_lag = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
        out["nber.mean_lag"] = _g(round(mean_lag, 2))
        out["nber.median_lag"] = _g(median_lag)
    else:
        out["nber.mean_lag"] = _NA
        out["nber.median_lag"] = _NA
    out["nber.n_peaks"] = str(len(lags))
    # Peaks before the walk-forward window opened carry lag_months = None/NaN and are excluded
    # from the mean and the median; n_in_window is what those statistics are actually computed over.
    out["nber.n_in_window"] = str(sum(1 for row in lags
                                      if row.get("lag_months") is not None and not _is_nan(row.get("lag_months"))))
    out["nber.n_censored"] = str(sum(1 for row in lags if row.get("censored")))

    # -- assets.* / bt.* ----------------------------------------------------
    if skipped_reason:
        out["skipped.assets"] = str(skipped_reason)
        for r in REGIMES:
            out[f"assets.n_{r}"] = _NA
        for key in ("assets.window_start", "assets.window_end", "assets.n_months", "assets.universe",
                    "assets.growth_share", "assets.r2", "assets.spread_pct", "assets.spread_ord",
                    "assets.spread_n", "assets.spread_direction", "bt.start", "bt.min_obs",
                    "bt.perf0", "bt.perf10", "bt.placebo_pct", "bt.placebo_ord",
                    "bt.placebo_n", "bt.placebo_direction", "bt.placebo_sentence",
                    "bt.counters", "bt.insample", "bt.oracle", "bt.pit", "bt.moment_lookahead",
                    "bt.label_lookahead", "bt.total_lookahead"):
            out[key] = _NA
        for strat in STRATEGIES:
            out[f"bt.sharpe_{strat}"] = _NA
            out[f"bt.sharpe10_{strat}"] = _NA
    else:
        out["skipped.assets"] = ""
        window = assets_blk.get("window") or {}
        out["assets.window_start"] = _ym(window.get("start")) if window.get("start") else _NA
        out["assets.window_end"] = _ym(window.get("end")) if window.get("end") else _NA
        out["assets.n_months"] = str(window.get("n_months", _NA))
        universe = assets_blk.get("universe") or {}
        out["assets.universe"] = ", ".join(f"{tkr} ({name})" for tkr, name in universe.items()) if universe else _NA
        n_per = assets_blk.get("n_per_regime") or {}
        for r in REGIMES:
            out[f"assets.n_{r}"] = str(n_per.get(r, _NA))
        gs = assets_blk.get("growth_share_6040") or {}
        out["assets.growth_share"] = _pct(gs.get("growth_share")) if "growth_share" in gs else _NA
        out["assets.r2"] = f"{gs['r2']:.3f}" if "r2" in gs else _NA
        sp = assets_blk.get("sharpe_spread_placebo") or {}
        spread_pct = sp.get("percentile")
        out["assets.spread_pct"] = f"{float(spread_pct):.0f}" if spread_pct is not None else _NA
        out["assets.spread_ord"] = _ordinal(spread_pct)
        # The null array is the authoritative count of shuffles actually drawn; `n` (which
        # placebo() does not return) is only a fallback. Never a hard-coded literal.
        spread_null = sp.get("null")
        if spread_null is not None:
            out["assets.spread_n"] = str(len(spread_null))
        elif sp.get("n") is not None:
            out["assets.spread_n"] = str(sp["n"])
        else:
            out["assets.spread_n"] = _NA
        out["assets.spread_direction"] = _direction(spread_pct)

        bt = assets_blk.get("backtest") or {}
        bt0 = bt.get("cost_bp_0") or {}
        bt10 = bt.get("cost_bp_10") or {}
        bt_params = bt0.get("params") or {}
        out["bt.start"] = _ym(bt_params.get("start")) if bt_params.get("start") else _NA
        out["bt.min_obs"] = str(bt_params.get("min_regime_obs", _NA))
        perf0 = bt0.get("perf") or {}
        perf10 = bt10.get("perf") or {}
        out["bt.perf0"] = _md_perf_table(perf0) if perf0 else _NA
        out["bt.perf10"] = _md_perf_table(perf10) if perf10 else _NA
        for strat in STRATEGIES:
            out[f"bt.sharpe_{strat}"] = _num(perf0[strat]["sharpe"]) if strat in perf0 else _NA
            out[f"bt.sharpe10_{strat}"] = _num(perf10[strat]["sharpe"]) if strat in perf10 else _NA

        look = assets_blk.get("lookahead") or {}
        out["bt.insample"] = _num(look.get("insample_sharpe")) if "insample_sharpe" in look else _NA
        out["bt.oracle"] = _num(look.get("oracle_sharpe")) if "oracle_sharpe" in look else _NA
        out["bt.pit"] = _num(look.get("pit_sharpe")) if "pit_sharpe" in look else _NA
        out["bt.moment_lookahead"] = _signed(look.get("moment_lookahead")) if "moment_lookahead" in look else _NA
        out["bt.label_lookahead"] = _signed(look.get("label_lookahead")) if "label_lookahead" in look else _NA
        out["bt.total_lookahead"] = _signed(look.get("total")) if "total" in look else _NA

        bp = assets_blk.get("backtest_placebo")
        bt_pct = bp.get("percentile") if bp else None
        if bp:
            out["bt.placebo_pct"] = f"{float(bt_pct):.0f}"
            out["bt.placebo_ord"] = _ordinal(bt_pct)
            out["bt.placebo_n"] = str(bp.get("n", _NA))
        else:
            out["bt.placebo_pct"] = _NA
            out["bt.placebo_ord"] = _NA
            out["bt.placebo_n"] = _NA
        out["bt.placebo_direction"] = _direction(bt_pct)
        out["bt.placebo_sentence"] = _placebo_sentence(bt_pct, spread_pct)

        out["bt.counters"] = _fmt_counters(bt0.get("counters") or {})

    return out


def _placebo_sentence(bt_pct, spread_pct) -> str:
    """One sentence on the direction of both placebos (or the Sharpe-spread one alone
    when the backtest placebo was not computed for this run — spec §8, --skip-placebo)."""
    spread_dir = _direction(spread_pct)
    if bt_pct is None or _is_nan(bt_pct):
        if spread_dir == _NA:
            return _NA
        if spread_dir == "below":
            return ("The Sharpe-spread placebo sits below the fiftieth percentile: more than half of the "
                    "random shuffles beat the real value.")
        return ("The Sharpe-spread placebo sits above the fiftieth percentile: the real value beats more "
                "than half of the random shuffles.")
    bt_dir = _direction(bt_pct)
    if bt_dir == spread_dir == "below":
        return "Both sit below the fiftieth percentile: more than half of the random relabelings beat the real one."
    if bt_dir == spread_dir == "above":
        return "Both sit above the fiftieth percentile: the real labels beat more than half of the random relabelings."
    return f"The backtest placebo sits {bt_dir} the fiftieth percentile and the Sharpe-spread placebo {spread_dir} it."


# ---------------------------------------------------------------------------
# render() / missing_placeholders()
# ---------------------------------------------------------------------------

_IF_ASSETS_RE = re.compile(r"<!--\s*if:assets\s*-->(.*?)<!--\s*endif\s*-->", re.DOTALL)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.<>]+)\s*\}\}")
_FIG_RE = re.compile(r"^!\[(?P<caption>[^\]]*)\]\(fig:(?P<name>[A-Za-z0-9_]+)\)\s*$", re.MULTILINE)


def _apply_guard(text: str, skipped_assets: str) -> str:
    if skipped_assets:
        return _IF_ASSETS_RE.sub("", text)
    return _IF_ASSETS_RE.sub(lambda m: m.group(1), text)


def _substitute(text: str, nums: dict) -> str:
    def repl(m):
        key = m.group(1)
        return nums.get(key, f"[missing: {key}]")
    return _PLACEHOLDER_RE.sub(repl, text)


def render(markdown_text: str, nums: dict, figures: dict) -> list:
    """Ordered blocks: ("md", text) or ("fig", path_or_None, caption, name)."""
    text = _apply_guard(markdown_text, nums.get("skipped.assets", ""))
    text = _substitute(text, nums)
    figures = figures or {}

    blocks = []
    pos = 0
    for m in _FIG_RE.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            blocks.append(("md", before))
        blocks.append(("fig", figures.get(m.group("name")), m.group("caption"), m.group("name")))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        blocks.append(("md", tail))
    return blocks


def missing_placeholders(markdown_text: str, nums: dict) -> list[str]:
    seen: list[str] = []
    for m in _PLACEHOLDER_RE.finditer(markdown_text):
        key = m.group(1)
        if key not in nums and key not in seen:
            seen.append(key)
    return seen


# ---------------------------------------------------------------------------
# to_html()
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg: #fbfaf7; --bg-soft: #f4f2ec; --ink: #1a1a1a; --ink-soft: #4a4a4a;
  --ink-faint: #8a8780; --rule: #e8e4dc; --accent: #b8410e; --accent-soft: #f5e6dd;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--ink); font-family: 'Inter Tight', -apple-system, sans-serif;
  max-width: 760px; margin: 0 auto; padding: 3rem 1.5rem 5rem; line-height: 1.65; font-size: 1.02rem;
}}
h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif; font-weight: 500; letter-spacing: -0.01em; color: var(--ink); }}
h1 {{ font-size: 2.1rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.5rem; }}
h2 {{ font-size: 1.5rem; margin-top: 2.4rem; }}
h3 {{ font-size: 1.15rem; margin-top: 1.6rem; }}
a {{ color: var(--accent); }}
table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }}
th, td {{ border: 1px solid var(--rule); padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: var(--bg-soft); }}
figure {{ margin: 1.6rem 0; text-align: center; }}
figure img {{ max-width: 100%; border: 1px solid var(--rule); border-radius: 6px; }}
figcaption {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--ink-faint); margin-top: 0.5rem; }}
.missing-figure {{ font-family: 'JetBrains Mono', monospace; color: var(--accent); background: var(--accent-soft);
  padding: 0.6rem 0.8rem; border-radius: 6px; }}
code, pre {{ font-family: 'JetBrains Mono', monospace; background: var(--bg-soft); border-radius: 4px; }}
code {{ padding: 0.1rem 0.3rem; }}
blockquote {{ border-left: 3px solid var(--accent); margin: 1rem 0; padding: 0.2rem 1rem; color: var(--ink-soft); }}
@media print {{
  body {{ max-width: 100%; padding: 0.5in; font-size: 11pt; }}
  a {{ color: var(--ink); text-decoration: none; }}
  figure img {{ max-width: 100%; }}
  h1, h2 {{ page-break-after: avoid; }}
  table, figure {{ page-break-inside: avoid; }}
}}
</style>
<!-- KaTeX is loaded from a CDN. Offline it simply does not load and every .math span shows
     its readable $...$ TeX in JetBrains Mono; nothing else on the page depends on it. -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}], ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}});"></script>
</head>
<body>
{body}
</body>
</html>
"""


# Display `$$...$$` (DOTALL, may span lines) or inline `$...$`. The inline branch follows
# Pandoc's rule: the opening `$` must not be followed by whitespace and the closing `$` must
# not be preceded by it, so a bare dollar sign in prose ("a price of $100 and another of $250")
# is not read as a delimiter. A span may wrap across a single line break — the paper does that —
# but never across a blank line, so a stray `$` cannot swallow a whole paragraph.
_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(?!\s)((?:[^$\n]|\n(?!\s*\n))+?)(?<!\s)\$", re.DOTALL)


def _md_to_html(text: str) -> str:
    """Markdown -> HTML with LaTeX spans protected.

    The paper writes inline math as `$...$` (Streamlit renders it with KaTeX). Python-
    Markdown would read the underscores and asterisks inside those spans as emphasis, so
    each span is lifted out before conversion and put back afterwards as a `.math`
    element that the exported page's KaTeX auto-render turns into typeset math (or
    leaves as readable TeX when offline).
    """
    if _markdown_lib is None:
        raise RuntimeError("the 'markdown' package is required for sitedocs.to_html "
                           "(pip install markdown>=3.5, or see requirements.txt)")
    spans: list[str] = []
    # A per-call random token: a fixed sentinel ("MATHSPAN0X") is text an author could type,
    # and would then be replaced by someone else's formula. The trailing "X" keeps token+"1"
    # from matching inside token+"11".
    token = uuid.uuid4().hex

    def lift(m):
        display = m.group(1) is not None
        tex = m.group(1) if display else m.group(2)
        spans.append(f"<span class=\"math{' display' if display else ''}\">"
                     f"{'$$' if display else '$'}{_html.escape(tex)}{'$$' if display else '$'}</span>")
        return f"{token}{len(spans) - 1}X"

    protected = _MATH_RE.sub(lift, text)
    out = _markdown_lib.markdown(protected, extensions=["tables", "sane_lists"])
    for i, span in enumerate(spans):
        out = out.replace(f"{token}{i}X", span)
    return out


def to_html(markdown_text: str, nums: dict, figures: dict, title: str) -> str:
    blocks = render(markdown_text, nums, figures)
    parts = []
    for block in blocks:
        if block[0] == "md":
            parts.append(_md_to_html(block[1]))
            continue
        _, path, caption, name = block
        if path is not None and Path(path).exists():
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            alt = _html.escape(caption or name)
            cap_html = f"<figcaption>{_html.escape(caption)}</figcaption>" if caption else ""
            parts.append(f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">{cap_html}</figure>')
        else:
            parts.append(f'<p class="missing-figure">[missing figure: {_html.escape(name)}]</p>')
    body = "\n".join(parts)
    return _HTML_TEMPLATE.format(title=_html.escape(str(title)), body=body)
