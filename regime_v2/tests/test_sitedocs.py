"""sitedocs.py against the contract in docs/site/CONTRACT.md.

The placeholder-key list is parsed straight out of CONTRACT.md so this test tracks the
contract rather than a hand-copied list; <Regime>/<Strategy>/<name> families are expanded.
"""
import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from regime_v2 import publish, regimes as R, sitedocs

SITE_DIR = Path(__file__).resolve().parents[2] / "docs" / "site"
CONTRACT_PATH = SITE_DIR / "CONTRACT.md"
DOC_PATHS = {name: SITE_DIR / f"{name}.md" for name in ("introduction", "methodology")}

# The label column each label_source names; the app reads the same one (app.py's REALTIME).
LABEL_COLUMN = {"walk-forward filtered": "hmm_walkforward",
                "full-sample filtered (walk-forward disabled)": "hmm_filtered"}


def _parse_contract_keys() -> list[str]:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    keys, in_table = [], False
    for line in text.splitlines():
        if line.startswith("| key |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("|---"):
            continue
        key_cell = line.split("|")[1]
        keys.extend(re.findall(r"`([^`]+)`", key_cell))
    return keys


def _expand(keys: list[str], acceptance_names: list[str]) -> list[str]:
    out = []
    for k in keys:
        if "<Regime>" in k:
            out.extend(k.replace("<Regime>", r) for r in R.REGIMES)
        elif "<Strategy>" in k:
            out.extend(k.replace("<Strategy>", s) for s in sitedocs.STRATEGIES)
        elif "<name>" in k:
            out.extend(k.replace("<name>", n) for n in acceptance_names)
        else:
            out.append(k)
    return out


@pytest.fixture(scope="module")
def pub(published_dir):
    out, figs = published_dir
    return publish.load_published(out, figs)


@pytest.fixture(scope="module")
def nums(pub):
    return sitedocs.numbers(pub)


@pytest.fixture(scope="module")
def contract_keys(pub) -> list[str]:
    raw = _parse_contract_keys()
    assert raw, "could not parse any placeholder keys out of CONTRACT.md — did its table format change?"
    return _expand(raw, pub.acceptance["name"].tolist())


def test_every_contract_key_is_present(nums, contract_keys):
    missing = sorted(set(contract_keys) - set(nums))
    assert not missing, f"numbers(pub) is missing keys from CONTRACT.md: {missing}"


def test_no_value_is_empty_except_skipped_assets(nums):
    empty = [k for k, v in nums.items()
             if v == "" and k not in ("skipped.assets", "skipped.lo_placebo", "skipped.rt_vintage")]
    assert not empty, f"empty values for: {empty}"
    assert isinstance(nums["skipped.assets"], str)


def test_all_values_are_strings(nums):
    non_str = {k: type(v).__name__ for k, v in nums.items() if not isinstance(v, str)}
    assert not non_str, non_str


def test_if_guard_drops_content_when_assets_skipped():
    md = "before <!-- if:assets -->HIDDEN<!-- endif --> after"
    blocks = sitedocs.render(md, {"skipped.assets": "walk-forward disabled"}, {})
    text = " ".join(b[1] for b in blocks if b[0] == "md")
    assert "HIDDEN" not in text
    assert "before" in text and "after" in text


def test_if_guard_keeps_content_when_assets_published():
    md = "before <!-- if:assets -->SHOWN<!-- endif --> after"
    blocks = sitedocs.render(md, {"skipped.assets": ""}, {})
    text = " ".join(b[1] for b in blocks if b[0] == "md")
    assert "SHOWN" in text


LO_GUARDED = ("a <!-- if:assets -->b <!-- if:lo_placebo -->LONGONLY<!-- endif:lo_placebo --> c"
              "<!-- endif --> d")


def _md_text(blocks):
    return " ".join(b[1] for b in blocks if b[0] == "md")


def test_lo_placebo_guard_drops_only_its_own_span_when_the_block_is_absent():
    """Nested inside if:assets: the inner span goes, the rest of the asset section stays."""
    text = _md_text(sitedocs.render(LO_GUARDED, {"skipped.assets": "", "skipped.lo_placebo": "not computed"}, {}))
    assert "LONGONLY" not in text and "b" in text and "c" in text and "d" in text
    text = _md_text(sitedocs.render(LO_GUARDED, {"skipped.assets": "", "skipped.lo_placebo": ""}, {}))
    assert "LONGONLY" in text and "<!--" not in text
    text = _md_text(sitedocs.render(LO_GUARDED, {"skipped.assets": "skipped", "skipped.lo_placebo": ""}, {}))
    assert "LONGONLY" not in text and "c" not in text


# A phrase from each document's long-only placebo paragraph.
LO_MARKERS = {"introduction": "run through the long-only strategy", "methodology": "long-only strategy is scored"}


def test_documents_never_claim_a_longonly_placebo_the_run_did_not_compute(pub):
    """The fixture run uses --skip-placebo: no page may say the long-only strategy was scored."""
    nums = sitedocs.numbers(pub)
    assert nums["skipped.lo_placebo"]
    for name, path in DOC_PATHS.items():
        text = _md_text(sitedocs.render(path.read_text(encoding="utf-8"), nums, {}))
        assert LO_MARKERS[name] not in text and "n/a that median" not in text, name
    nums = sitedocs.numbers(_with_placebos(pub, bt_null=[0.1, 0.2, 0.3, 0.9], lo_null=[0.9, 1.0, 1.1, 1.2, 1.3]))
    assert nums["skipped.lo_placebo"] == ""
    for name, path in DOC_PATHS.items():
        assert LO_MARKERS[name] in _md_text(sitedocs.render(path.read_text(encoding="utf-8"), nums, {})), name


def test_unknown_placeholder_renders_as_missing():
    blocks = sitedocs.render("{{not.a.real.key}}", {}, {})
    text = " ".join(b[1] for b in blocks if b[0] == "md")
    assert "[missing: not.a.real.key]" in text


def test_known_placeholder_is_substituted():
    blocks = sitedocs.render("value is {{a.b}}", {"a.b": "42"}, {})
    text = " ".join(b[1] for b in blocks if b[0] == "md")
    assert "value is 42" in text
    assert "[missing" not in text


def test_figure_marker_resolves_to_real_path(pub):
    name = "fig1_factors_gaps"
    path = pub.figures[name]
    assert path is not None, "sanity: the published_dir fixture should publish this figure"
    md = f"before\n\n![Caption text](fig:{name})\n\nafter"
    blocks = sitedocs.render(md, {}, pub.figures)
    figs = [b for b in blocks if b[0] == "fig"]
    assert len(figs) == 1
    _, p, caption, nm = figs[0]
    assert p == path
    assert caption == "Caption text"
    assert nm == name


def test_figure_marker_resolves_to_none_when_missing():
    blocks = sitedocs.render("![x](fig:doc_pipeline)", {}, {})
    figs = [b for b in blocks if b[0] == "fig"]
    assert len(figs) == 1
    assert figs[0][1] is None
    assert figs[0][3] == "doc_pipeline"


def test_missing_placeholders_helper():
    nums_small = {"a.b": "1"}
    missing = sitedocs.missing_placeholders("{{a.b}} {{c.d}} {{c.d}} {{e.f}}", nums_small)
    assert missing == ["c.d", "e.f"]


def test_to_html_embeds_image_and_title(pub, nums):
    md = "# Title\n\nSome text with {{current.regime}}.\n\n![Caption](fig:fig1_factors_gaps)\n"
    out = sitedocs.to_html(md, nums, pub.figures, "My Document Title")
    assert "data:image/png;base64," in out
    assert "My Document Title" in out
    assert "<title>My Document Title</title>" in out
    assert nums["current.regime"] in out


def test_to_html_shows_missing_figure_visibly(nums):
    out = sitedocs.to_html("![x](fig:doc_nonexistent)", nums, {}, "T")
    assert "[missing figure: doc_nonexistent]" in out


def test_hmm_transition_table_is_a_markdown_table(nums):
    table = nums["hmm.transition_table"]
    assert table.startswith("|")
    for r in R.REGIMES:
        assert r in table


def test_bt_perf_tables_list_every_strategy(nums):
    for key in ("bt.perf0", "bt.perf10"):
        table = nums[key]
        for strat in sitedocs.STRATEGIES:
            assert strat in table, f"{strat} missing from {key}"


def test_acc_table_excludes_report_rows(pub, nums):
    report_names = pub.acceptance.loc[pub.acceptance["op"] == "report", "name"].tolist()
    table = nums["acc.table"]
    for name in report_names:
        assert f"| {name} |" not in table


def test_placebo_sentence_matches_direction(nums):
    sentence = nums["bt.placebo_sentence"]
    assert sentence and sentence != "n/a"
    bt_dir = nums["bt.placebo_direction"]
    spread_dir = nums["assets.spread_direction"]
    if bt_dir == "n/a":
        assert "Sharpe-spread placebo" in sentence
    elif bt_dir == spread_dir == "below":
        assert sentence == ("Both sit below the fiftieth percentile: more than half of the random "
                            "relabelings beat the real one.")
    elif bt_dir == spread_dir == "above":
        assert sentence == ("Both sit above the fiftieth percentile: the real labels beat more than half "
                            "of the random relabelings.")
    else:
        assert bt_dir in sentence and spread_dir in sentence


def test_placebo_direction_thresholds():
    assert sitedocs._direction(49.9) == "below"
    assert sitedocs._direction(50) == "above"
    assert sitedocs._direction(50.1) == "above"
    assert sitedocs._direction(None) == "n/a"


def test_to_html_protects_latex_spans():
    """Underscores inside `$...$` must reach the page as TeX, not Markdown emphasis."""
    from regime_v2 import sitedocs
    md = r"Gap $\mathrm{gap}_t = \bar y_t - x_{t-1}$ and *emphasis* stays." + "\n"
    html = sitedocs.to_html(md, {}, {}, "t")
    assert r'<span class="math">$\mathrm{gap}_t = \bar y_t - x_{t-1}$</span>' in html
    assert "<em>emphasis</em>" in html
    assert "MATHSPAN" not in html
    assert "katex" in html


# ---------------------------------------------------------------------------
# math protection: the cases that broke the exported paper
# ---------------------------------------------------------------------------

def _math_spans(html: str) -> list[str]:
    return re.findall(r'<span class="math[^"]*">(.*?)</span>', html, re.DOTALL)


def _body_outside_math(html: str) -> str:
    """The document body with every `.math` element removed. The <head> is excluded because
    KaTeX's auto-render config legitimately contains `'$'` delimiters."""
    body = html.split("<body>", 1)[1].split("</body>")[0]
    return re.sub(r'<span class="math[^"]*">.*?</span>', "", body, flags=re.DOTALL)


def test_to_html_spans_a_single_line_break():
    """The paper wraps `$...$` across a source line; the span must survive whole."""
    md = "prose $1 + \\varepsilon +\n\\kappa I$ with $\\varepsilon = 0.5$ off the diagonal.\n"
    html = sitedocs.to_html(md, {}, {}, "t")
    spans = _math_spans(html)
    assert spans == ["$1 + \\varepsilon +\n\\kappa I$", "$\\varepsilon = 0.5$"]
    assert "with" not in " ".join(spans) and "off the diagonal" not in " ".join(spans)
    assert "$" not in _body_outside_math(html)


def test_to_html_does_not_span_a_blank_line():
    md = "a stray $ dollar here.\n\nAnd another $ dollar there.\n"
    html = sitedocs.to_html(md, {}, {}, "t")
    assert _math_spans(html) == []


def test_to_html_ignores_bare_dollar_amounts():
    """Pandoc's rule: `$` followed by a space (or preceded by one) is not a delimiter."""
    md = "A price of $100 and another of $250 in one line.\n"
    html = sitedocs.to_html(md, {}, {}, "t")
    assert _math_spans(html) == []
    md2 = "Costs $ 100 to $ 250 of turnover.\n"
    assert _math_spans(sitedocs.to_html(md2, {}, {}, "t")) == []


def test_to_html_sentinel_cannot_be_typed_by_the_author():
    md = "literal MATHSPAN0X token and $x$ math\n"
    html = sitedocs.to_html(md, {}, {}, "t")
    assert "literal MATHSPAN0X token" in html
    assert _math_spans(html) == ["$x$"]


@pytest.mark.parametrize("name", sorted(DOC_PATHS))
def test_real_documents_export_cleanly(name, pub, nums):
    """Both real documents through to_html: every `$...$` span in the source becomes exactly
    one `.math` element, no sentinel survives, and Markdown never runs inside a formula."""
    path = DOC_PATHS[name]
    if not path.exists():
        pytest.skip(f"{path} does not exist yet")
    source = path.read_text(encoding="utf-8")
    assert source.count("$") % 2 == 0, f"{name}.md has an odd number of '$' characters"
    n_spans = source.count("$") // 2

    html = sitedocs.to_html(source, nums, pub.figures, name)

    assert html.count('class="math') == n_spans, f"{name}: math spans lost or invented"
    assert "MATHSPAN" not in html
    assert re.search(r"[0-9a-f]{32}\d+X", html) is None, "a lift sentinel survived into the export"
    for span in _math_spans(html):
        assert "<em>" not in span and "<strong>" not in span, f"Markdown ran inside math: {span[:80]}"
    assert "$" not in _body_outside_math(html), f"{name}: a '$' survived outside a math span"
    assert "[missing:" not in html          # figures may legitimately be absent, placeholders may not


# ---------------------------------------------------------------------------
# an absent asset block is a skipped stage, not a successful one
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nums_no_assets(pub):
    summary = {k: v for k, v in pub.summary.items() if k != "assets"}
    return sitedocs.numbers(replace(pub, summary=summary))


def test_absent_assets_block_reads_as_skipped(nums_no_assets):
    assert nums_no_assets["skipped.assets"] == "asset stage not run"
    assert nums_no_assets["bt.pit"] == "n/a"
    assert nums_no_assets["bt.placebo_sentence"] == "n/a"


def test_absent_assets_block_drops_the_guarded_section(nums_no_assets):
    path = DOC_PATHS["introduction"]
    if not path.exists():
        pytest.skip(f"{path} does not exist yet")
    blocks = sitedocs.render(path.read_text(encoding="utf-8"), nums_no_assets, {})
    text = " ".join(b[1] for b in blocks if b[0] == "md")
    assert "Does it make money?" not in text
    assert "n/a" not in text
    assert "What refreshes, and what stops it" in text     # the rest of the document is intact


# ---------------------------------------------------------------------------
# values: each expected number recomputed from summary.json / regime_labels.csv
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_summary(published_dir):
    out, _ = published_dir
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_labels(published_dir):
    out, _ = published_dir
    return pd.read_csv(out / "regime_labels.csv", index_col=0, parse_dates=[0])


def test_expected_durations_are_one_over_one_minus_the_diagonal(nums, raw_summary):
    tm = raw_summary["transition_matrix"]
    for r in R.REGIMES:
        expected = 1.0 / (1.0 - float(tm[r][r]))
        assert nums[f"hmm.expected_duration_{r}"] == f"{expected:.1f} months", r


def test_regime_shares_match_the_published_label_column(nums, raw_summary, raw_labels):
    col = LABEL_COLUMN[raw_summary["label_source"]]
    counts = raw_labels[col].dropna().value_counts()
    total = int(counts.sum())
    for r in R.REGIMES:
        expected = f"{int(counts.get(r, 0)) / total:.0%}"
        assert nums[f"hmm.share_{r}"] == expected, r


def test_current_probabilities_are_the_summary_probabilities_as_percent(nums, raw_summary):
    probs = raw_summary["current"]["probs"]
    for r in R.REGIMES:
        assert nums[f"current.prob_{r}"] == f"{float(probs[r]) * 100:.0f}%", r


def test_pit_sharpes_match_the_zero_and_ten_bp_perf_tables(nums, raw_summary):
    bt = raw_summary["assets"]["backtest"]
    for key, cost in (("bt.sharpe_PIT_MaxSharpe", "cost_bp_0"), ("bt.sharpe10_PIT_MaxSharpe", "cost_bp_10")):
        v = float(bt[cost]["perf"]["PIT_MaxSharpe"]["sharpe"])
        expected = str(int(v)) if v == int(v) else f"{v:.2f}"
        assert nums[key] == expected, key


def test_lo_pit_is_the_long_only_pit_perf_row(nums, raw_summary):
    """bt.lo_pit is the point-in-time leg of the long-only decomposition, which is the
    `PIT_LongOnly_MaxSharpe` row of the 0 bp perf table, formatted like bt.pit."""
    assets = raw_summary.get("assets") or {}
    look_lo = assets.get("lookahead_longonly")
    if not look_lo:
        pytest.skip("this published run has no assets.lookahead_longonly block")
    v = float(assets["backtest"]["cost_bp_0"]["perf"]["PIT_LongOnly_MaxSharpe"]["sharpe"])
    # the perf table is published rounded, the decomposition is not, so they agree to the
    # table's own precision rather than exactly
    assert float(look_lo["pit_sharpe"]) == pytest.approx(v, abs=1e-4)
    expected = str(int(v)) if v == int(v) else f"{v:.2f}"
    assert nums["bt.lo_pit"] == expected


@pytest.fixture(scope="module")
def nums_no_longonly(pub):
    """The published run with the long-only decomposition removed: a run from before the
    constrained strategies existed, which must still render."""
    summary = dict(pub.summary)
    summary["assets"] = {k: v for k, v in (summary.get("assets") or {}).items()
                         if k != "lookahead_longonly"}
    return sitedocs.numbers(replace(pub, summary=summary))


def test_absent_longonly_block_reads_as_na(nums_no_longonly):
    for key in ("bt.lo_insample", "bt.lo_oracle", "bt.lo_pit", "bt.lo_moment_lookahead",
                "bt.lo_label_lookahead", "bt.lo_total"):
        assert nums_no_longonly[key] == "n/a", key
    assert nums_no_longonly["skipped.assets"] == ""      # the asset stage itself still ran
    assert nums_no_longonly["bt.pit"] != "n/a"           # and the unconstrained legs are untouched


def test_sample_start_is_the_first_month_with_both_gaps(nums, raw_labels):
    gap_cols = [c for c in raw_labels.columns if c.endswith("_gap")]
    assert sorted(gap_cols) == ["growth_gap", "inflation_gap"], gap_cols
    complete = raw_labels[raw_labels[gap_cols].notna().all(axis=1)]
    assert nums["sample.start"] == str(complete.index[0])[:7]
    assert nums["sample.n_months"] == str(len(raw_labels))


def test_nber_counts_match_the_lag_records(nums, raw_summary):
    lags = raw_summary["nber_lags_rt"]
    assert nums["nber.n_peaks"] == str(len(lags))
    in_window = [row for row in lags
                 if row["lag_months"] is not None and not sitedocs._is_nan(row["lag_months"])]
    assert nums["nber.n_in_window"] == str(len(in_window))
    assert int(nums["nber.n_in_window"]) <= int(nums["nber.n_peaks"])
    assert nums["nber.n_censored"] == str(sum(1 for row in lags if row["censored"]))


def test_run_and_placebo_values_match_the_summary(nums, raw_summary):
    assert nums["run.asof"] == raw_summary["run"]["asof"][:7]
    assert nums["run.vintage"] == raw_summary["run"]["vintage"]
    assert nums["run.label_source"] == raw_summary["run"]["label_source"]
    sp = raw_summary["assets"]["sharpe_spread_placebo"]
    assert nums["assets.spread_n"] == str(len(sp["null"]))          # never a hard-coded 1000
    assert nums["assets.spread_pct"] == f"{float(sp['percentile']):.0f}"
    assert nums["assets.spread_ord"].startswith(nums["assets.spread_pct"])
    assert nums["assets.spread_ord"][-2:] in ("st", "nd", "rd", "th")


def _with_placebos(pub, bt_null, lo_null, bt_real=0.8, lo_real=1.05):
    """pub with both backtest placebos filled in (the fixture run uses --skip-placebo)."""
    def block(null, real):
        return {"real": real, "n": len(null), "null": list(null),
                "percentile": 100.0 * sum(x <= real for x in null) / len(null)}
    assets = dict(pub.summary["assets"], backtest_placebo=block(bt_null, bt_real),
                  backtest_placebo_longonly=block(lo_null, lo_real))
    return replace(pub, summary=dict(pub.summary, assets=assets))


def test_longonly_placebo_values_come_from_its_own_block(pub):
    nums = sitedocs.numbers(_with_placebos(pub, bt_null=[0.1, 0.2, 0.3, 0.9], lo_null=[0.9, 1.0, 1.1, 1.2, 1.3]))
    assert nums["bt.placebo_pct"] == "75" and nums["bt.placebo_median"] == "0.25"
    assert nums["bt.lo_placebo_pct"] == "40" and nums["bt.lo_placebo_ord"] == "40th"
    assert nums["bt.lo_placebo_direction"] == "below"
    assert nums["bt.lo_placebo_median"] == "1.10"          # median of the null draws, not of anything else


def test_longonly_placebo_absent_reads_na_without_borrowing(pub):
    """A run published before the long-only placebo existed (or with --skip-placebo) must say
    n/a for it rather than reuse the unconstrained numbers."""
    p = _with_placebos(pub, bt_null=[0.1, 0.2, 0.3, 0.9], lo_null=[1.0])
    assets = {k: v for k, v in p.summary["assets"].items() if k != "backtest_placebo_longonly"}
    nums = sitedocs.numbers(replace(p, summary=dict(p.summary, assets=assets)))
    assert nums["bt.placebo_pct"] == "75"
    for k in ("bt.lo_placebo_pct", "bt.lo_placebo_ord", "bt.lo_placebo_direction", "bt.lo_placebo_median"):
        assert nums[k] == "n/a", k


def test_ordinal_suffixes():
    assert [sitedocs._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31, 100)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "31st", "100th"]
    assert sitedocs._ordinal(None) == "n/a"


# ---------------------------------------------------------------------------
# rt.* — the real-time-vintage comparator (spec 2026-09-25)
# ---------------------------------------------------------------------------

def _rt_block(overall=0.787, recon=0.784, pub=0.791, same_peaks=True):
    lag_f = [{"peak": "2001-03", "first_low_growth_rt": "2001-03", "lag_months": 0, "censored": False},
             {"peak": "2007-12", "first_low_growth_rt": "2007-12", "lag_months": 0, "censored": False}]
    lag_r = [dict(r) for r in lag_f]
    if not same_peaks:
        lag_r[1]["first_low_growth_rt"] = "2008-02"; lag_r[1]["lag_months"] = 2
    return {"window": {"start": "1999-07", "end": "2026-07", "n_months": 324}, "start_requested": "1999-07",
            "gaps": {"2025-10": "x"}, "n_gaps": 1, "n_reconstructed": 185, "n_published": 139,
            "agreement": {"overall": overall, "reconstructed": recon, "published": pub},
            "crosstab": {}, "shares": {}, "mean_run_length": {}, "switches": {"final": 50, "rt": 48},
            "gap_revision": {"growth": {"mean": 0.07, "sd": 0.3, "corr": 0.97, "sign_agreement": 0.86},
                             "inflation": {"mean": -0.12, "sd": 0.3, "corr": 0.95, "sign_agreement": 0.89}},
            "nber_lags": {"final": lag_f, "rt": lag_r}, "loader": {}}


def _with_rt(pub, same_peaks=True):
    assets = dict(pub.summary["assets"], rt_vintage_backtest={
        "perf": {"cost_bp_0": {"PIT_MaxSharpe": {"sharpe": 1.179}, "PIT_LongOnly_MaxSharpe": {"sharpe": 0.841}, "Static_6040": {"sharpe": 1.051}},
                 "cost_bp_10": {"PIT_MaxSharpe": {"sharpe": 1.0}, "PIT_LongOnly_MaxSharpe": {"sharpe": 0.8}, "Static_6040": {"sharpe": 1.0}}},
        "counters": {}, "held_months": 1, "window": {"start": "2010-01-01", "end": "2026-08-01"}})
    return replace(pub, summary=dict(pub.summary, rt_vintage=_rt_block(same_peaks=same_peaks), assets=assets))


def test_rt_placeholders_come_from_the_block(pub):
    nums = sitedocs.numbers(_with_rt(pub))
    assert nums["rt.window_start"] == "1999-07" and nums["rt.n_months"] == "324" and nums["rt.n_gaps"] == "1"
    assert nums["rt.agreement"] == "79%" and nums["rt.agreement_reconstructed"] == "78%" and nums["rt.agreement_published"] == "79%"
    assert nums["rt.growth_corr"] == "0.97" and nums["rt.inflation_corr"] == "0.95"
    assert nums["rt.pit"] == "1.18" and nums["rt.pit_longonly"] == "0.84" and nums["rt.pit10"] == "1" and nums["rt.held_months"] == "1"
    assert nums["rt.nber_sentence"] == ("For the 2 recession peaks inside the window (2001, 2007) the first low-growth "
                                        "call falls in the same month under both labels.")
    assert nums["skipped.rt_vintage"] == ""
    nums2 = sitedocs.numbers(_with_rt(pub, same_peaks=False))
    assert nums2["rt.nber_sentence"].startswith("The first low-growth call differs for 1 of the 2 recession peaks")
    assert "2007-12: final 2007-12, real-time 2008-02" in nums2["rt.nber_sentence"]


def test_rt_backtest_placeholders_read_na_when_the_rt_backtest_was_skipped(pub):
    """The rt backtest has its own try in run_assets: a failure publishes {"skipped": reason}
    with no perf, beside the comparison block."""
    p = _with_rt(pub)
    assets = dict(p.summary["assets"], rt_vintage_backtest={"skipped": "IndexError: x"})
    nums = sitedocs.numbers(replace(p, summary=dict(p.summary, assets=assets)))
    for k in ("rt.pit", "rt.pit_longonly", "rt.pit10", "rt.pit_longonly10", "rt.held_months"):
        assert nums[k] == "n/a", k
    assert nums["rt.agreement"] == "79%"


def test_rt_placeholders_read_na_without_the_block(pub):
    # The fixture run itself always carries an rt_vintage key (run.py sets it unconditionally),
    # skipped for lack of a --vintage-archive; the "absent key" case (an older run's summary,
    # published before this stage existed) is exercised separately below.
    nums = sitedocs.numbers(pub)
    for k in ("rt.window_start", "rt.agreement", "rt.nber_sentence", "rt.pit", "rt.pit_longonly"):
        assert nums[k] == "n/a", k
    assert nums["skipped.rt_vintage"] == "no vintage archive configured"

    no_key = {k: v for k, v in pub.summary.items() if k != "rt_vintage"}
    nums_absent = sitedocs.numbers(replace(pub, summary=no_key))
    for k in ("rt.window_start", "rt.agreement", "rt.nber_sentence", "rt.pit", "rt.pit_longonly"):
        assert nums_absent[k] == "n/a", k
    assert nums_absent["skipped.rt_vintage"] == "real-time-vintage stage not run"

    skipped = replace(pub, summary=dict(pub.summary, rt_vintage={"skipped": "no vintage archive configured"}))
    assert sitedocs.numbers(skipped)["skipped.rt_vintage"] == "no vintage archive configured"


def test_rt_guards_select_the_measured_or_the_stated_limitation():
    md = ("x <!-- if:assets -->y <!-- if:rt_vintage -->MEASURED<!-- endif:rt_vintage -->"
          "<!-- ifnot:rt_vintage -->STATED<!-- endif:rt_vintage --> z<!-- endif --> w")
    on = _md_text(sitedocs.render(md, {"skipped.assets": "", "skipped.rt_vintage": ""}, {}))
    off = _md_text(sitedocs.render(md, {"skipped.assets": "", "skipped.rt_vintage": "not run"}, {}))
    assert "MEASURED" in on and "STATED" not in on and "<!--" not in on
    assert "STATED" in off and "MEASURED" not in off and "<!--" not in off


def test_documents_state_the_limitation_or_the_measurement_never_both(pub):
    nums_off, nums_on = sitedocs.numbers(pub), sitedocs.numbers(_with_rt(pub))
    for name, path in DOC_PATHS.items():
        src = path.read_text(encoding="utf-8")
        text_off = _md_text(sitedocs.render(src, nums_off, {}))
        text_on = _md_text(sitedocs.render(src, nums_on, {}))
        assert "<!--" not in text_off and "<!--" not in text_on, name
        # rt.window_start ("1999-07") only ever appears inside the if:rt_vintage text, unlike a
        # bare percentage, which this fixture's hmm.filtered_vs_smoothed also happens to format to.
        assert "1999-07" in text_on and "1999-07" not in text_off, name
        if name == "methodology":
            assert "ALFRED" in text_off and "ALFRED" not in text_on
            assert "reconstructed" in text_on
