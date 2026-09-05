"""sitedocs.py against the contract in docs/site/CONTRACT.md.

The placeholder-key list is parsed straight out of CONTRACT.md so this test tracks the
contract rather than a hand-copied list; <Regime>/<Strategy>/<name> families are expanded.
"""
import re
from pathlib import Path

import pytest

from regime_v2 import publish, regimes as R, sitedocs

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "docs" / "site" / "CONTRACT.md"


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
    empty = [k for k, v in nums.items() if v == "" and k != "skipped.assets"]
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
