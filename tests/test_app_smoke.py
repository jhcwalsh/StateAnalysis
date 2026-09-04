from streamlit.testing.v1 import AppTest

TABS = ["Factor gaps", "Factor levels", "Probabilities", "State space", "Regime returns", "Correlations",
        "Portfolios", "Backtest", "Acceptance", "Figures"]


def _run(monkeypatch, out, figs):
    monkeypatch.setenv("REGIME_OUTPUT_DIR", str(out))
    monkeypatch.setenv("REGIME_FIGS_DIR", str(figs))
    return AppTest.from_file("app.py", default_timeout=120).run()


def test_app_renders_status(published_dir, monkeypatch):
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    import json
    s = json.loads((out / "summary.json").read_text())
    page = " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
    assert s["current"]["regime"] in page
    assert s["current"]["month"] in page
    assert "walk-forward" in page.lower()


def test_app_tabs_and_asset_content(published_dir, monkeypatch):
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    labels = [t.label for t in at.tabs]
    for t in TABS:
        assert t in labels, t
    page = " ".join(md.value for md in at.markdown)
    assert "Static_6040" in page or "60/40" in page      # benchmark shown beside the PIT Sharpe
    assert "look-ahead" in page.lower()


def test_app_without_asset_stage(published_dir, monkeypatch, tmp_path):
    out, figs = published_dir
    o2 = tmp_path / "out"; o2.mkdir()
    for f in ["regime_labels.csv", "summary.json", "acceptance.csv"]:
        (o2 / f).write_bytes((out / f).read_bytes())
    at = _run(monkeypatch, o2, tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown) + " ".join(i.value for i in at.info)
    assert "asset stage" in page.lower()                 # the tabs explain what is missing


def test_app_empty_state(tmp_path, monkeypatch):
    at = _run(monkeypatch, tmp_path / "nowhere", tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "No published run" in page
