from dataclasses import replace

from regime_v2 import docfigs
from regime_v2 import publish as P

ENGINE_ONLY = {"doc_pipeline", "doc_quadrants", "doc_timing", "doc_loadings", "doc_transition"}


def test_write_doc_figures_all_but_placebo(published_dir):
    """published_dir runs with --skip-placebo, so backtest_placebo has no null draws:
    doc_placebo must come back None while every other doc figure is drawn and non-trivial."""
    out, figs = published_dir
    pub = P.load_published(out, figs)
    result = docfigs.write_doc_figures(pub, figs)
    assert set(result) == set(docfigs.DOC_FIGURES)
    assert result["doc_placebo"] is None
    for name, path in result.items():
        if name == "doc_placebo":
            continue
        assert path is not None and path.exists(), name
        assert path.stat().st_size > 10_000, f"{name} is suspiciously small ({path.stat().st_size} bytes)"


def test_write_doc_figures_without_assets_summary(published_dir, tmp_path):
    """A summary with no 'assets' key at all (asset stage never ran) must still produce the
    five engine-only figures, skip the two asset-dependent ones, and never raise."""
    out, figs = published_dir
    pub = P.load_published(out, figs)
    summary_no_assets = {k: v for k, v in pub.summary.items() if k != "assets"}
    pub_no_assets = replace(pub, summary=summary_no_assets)
    figs2 = tmp_path / "figs2"

    result = docfigs.write_doc_figures(pub_no_assets, figs2)

    assert set(result) == set(docfigs.DOC_FIGURES)
    assert {n for n, p in result.items() if p is not None} == ENGINE_ONLY
    for n in ENGINE_ONLY:
        assert result[n].exists() and result[n].stat().st_size > 10_000, n
    assert result["doc_lookahead"] is None
    assert result["doc_placebo"] is None


def test_load_published_exposes_doc_figures(published_dir):
    out, figs = published_dir
    pub = P.load_published(out, figs)
    assert set(docfigs.DOC_FIGURES) <= set(pub.figures)
    for name in ENGINE_ONLY:
        assert pub.figures[name] is not None and pub.figures[name].exists(), name
    assert pub.figures["doc_placebo"] is None
