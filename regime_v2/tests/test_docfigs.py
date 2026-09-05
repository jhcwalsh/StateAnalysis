import json
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


def test_summary_records_doc_figures_by_name(published_dir):
    """summary.json is a published surface: it carries figure file names, never host paths."""
    out, _ = published_dir
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))["doc_figures"]
    assert set(doc) == set(docfigs.DOC_FIGURES)
    for name, value in doc.items():
        assert value is None or value == f"{name}.png", (name, value)


def test_pipeline_figure_reads_its_numbers_from_the_run(published_dir, tmp_path, monkeypatch):
    """_draw_pipeline must take the series counts, trend window and lag from the summary."""
    out, figs = published_dir
    pub = P.load_published(out, figs)
    seen = {}
    monkeypatch.setattr(docfigs, "_stage_box",
                        lambda ax, x, y, w, h, title, detail=None: seen.setdefault(title, detail))
    assert docfigs._draw_pipeline(pub, tmp_path / "pipeline.png")
    params, loadings = pub.summary["params"], pub.summary["loadings"]
    assert seen["Growth & inflation blocks"] == f"{len(loadings['growth'])} growth series, {len(loadings['inflation'])} inflation series"
    assert seen["One-sided trend gap"].startswith(f"{params['window']:g}-month trailing mean")
    assert seen["Labels + available_at"] == f"publication lag = {params['publication_lag_months']:g} month"


def _drawn_axes(pub, path, monkeypatch):
    """Draw doc_lookahead and report the axes of the figure it built."""
    seen = {}
    real_close = docfigs.plt.close

    def spy(fig=None):
        if hasattr(fig, "axes"):
            seen["n"] = len(fig.axes)
            seen["titles"] = [ax.get_title() for ax in fig.axes]
        real_close(fig)

    monkeypatch.setattr(docfigs.plt, "close", spy)
    assert docfigs._draw_lookahead(pub, path)
    return seen


def test_lookahead_figure_draws_two_panels_when_the_longonly_block_exists(published_dir, tmp_path, monkeypatch):
    """The published run carries assets.lookahead_longonly, so the figure compares the two families."""
    out, figs = published_dir
    pub = P.load_published(out, figs)
    assert "lookahead_longonly" in pub.summary["assets"]
    seen = _drawn_axes(pub, tmp_path / "two.png", monkeypatch)
    assert seen["n"] == 2 and seen["titles"] == ["Unconstrained long-short", "Long-only"]
    assert (tmp_path / "two.png").stat().st_size > 10_000


def test_lookahead_figure_falls_back_to_one_panel_without_the_longonly_block(published_dir, tmp_path, monkeypatch):
    """A summary from before the constrained strategies existed must still draw the single panel."""
    out, figs = published_dir
    pub = P.load_published(out, figs)
    a = {k: v for k, v in pub.summary["assets"].items() if k != "lookahead_longonly"}
    pub_old = replace(pub, summary={**pub.summary, "assets": a})
    seen = _drawn_axes(pub_old, tmp_path / "one.png", monkeypatch)
    assert seen["n"] == 1
    assert (tmp_path / "one.png").stat().st_size > 10_000


def test_load_published_exposes_doc_figures(published_dir):
    out, figs = published_dir
    pub = P.load_published(out, figs)
    assert set(docfigs.DOC_FIGURES) <= set(pub.figures)
    for name in ENGINE_ONLY:
        assert pub.figures[name] is not None and pub.figures[name].exists(), name
    assert pub.figures["doc_placebo"] is None
