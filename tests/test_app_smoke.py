from streamlit.testing.v1 import AppTest

TABS = ["Explore θ", "Factor gaps", "Factor levels", "State space", "Regime returns", "Correlations",
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


def test_front_page_states_the_numbers_behind_the_current_label(published_dir, monkeypatch):
    """The opening screen must show the inputs, not just the answer (θ band, gaps, probabilities)."""
    import json
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    s = json.loads((out / "summary.json").read_text())
    cur, theta = s["current"], s["params"]["theta"]
    page = (" ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)
            + " ".join(h.value for h in at.header))
    assert f"Latest assessment · {cur['month']}" in page
    assert f"{cur['growth_gap']:+.2f}" in page and f"{cur['inflation_gap']:+.2f}" in page
    assert f"±{theta:.2f} dead band" in page or f"{theta:.2f}, so the" in page   # the trigger read
    assert "publication lag" in page                       # what the label is actable from
    assert "walk-forward gaps, which are not published" in page   # provenance of the gaps shown
    # The probabilities chart and its data table are on the opening screen, not in a tab.
    assert any("Full probability series" in e.label for e in at.expander)


def test_app_tabs_and_asset_content(published_dir, monkeypatch):
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    labels = [t.label for t in at.tabs]
    for t in TABS:
        assert t in labels, t
    page = " ".join(md.value for md in at.markdown)
    assert "Static_6040 over the same window" in page    # benchmark shown beside the PIT Sharpe
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


def _click_refresh(published_dir, monkeypatch, result):
    """Run the app, stub the engine call, and press Refresh. Returns the resulting AppTest."""
    out, figs = published_dir
    from regime_v2 import publish
    monkeypatch.setattr(publish, "run_refresh", lambda *a, **k: result)
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    assert at.button, "the published page must offer a Refresh button"
    return at.button[0].click().run(), out


def test_app_refresh_failure_keeps_the_published_run(published_dir, monkeypatch):
    at, out = _click_refresh(published_dir, monkeypatch, (False, "boom"))
    assert not at.exception
    assert any("boom" in e.value for e in at.error), [e.value for e in at.error]
    import json
    s = json.loads((out / "summary.json").read_text())
    page = " ".join(md.value for md in at.markdown)
    assert s["current"]["regime"] in page          # the old run is still on screen


def test_app_refresh_success_shows_no_error(published_dir, monkeypatch):
    at, _ = _click_refresh(published_dir, monkeypatch, (True, ""))
    assert not at.exception
    assert not at.error, [e.value for e in at.error]


def test_app_empty_state(tmp_path, monkeypatch):
    at = _run(monkeypatch, tmp_path / "nowhere", tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "No published run" in page


def test_app_rejects_a_pre_contract_summary(published_dir, monkeypatch, tmp_path):
    """A summary.json from before the current/run blocks existed must not crash the page."""
    import json
    out, figs = published_dir
    o2 = tmp_path / "out"; o2.mkdir()
    for f in ["regime_labels.csv", "acceptance.csv"]:
        (o2 / f).write_bytes((out / f).read_bytes())
    s = json.loads((out / "summary.json").read_text())
    s.pop("current"); s.pop("run")
    (o2 / "summary.json").write_text(json.dumps(s, default=str))
    at = _run(monkeypatch, o2, tmp_path / "nofigs")
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "predates the current contract" in page


def test_site_theme_is_applied(published_dir, monkeypatch):
    # The app carries the lazyeconomist.com look: a fixed light theme file and the masthead
    # with its link back to the landing page.
    from pathlib import Path
    cfg = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in cfg and 'primaryColor = "#b8410e"' in cfg and "Fraunces" in cfg
    out, figs = published_dir
    at = _run(monkeypatch, out, figs)
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "lazyeconomist.com" in page and "le-topbar" in page and "@import url('https://fonts.googleapis.com" in page
    # Without an explicit target Streamlit rewrites the anchor to target="_blank" and the link
    # back to the landing page opens a new tab instead of navigating.
    assert '<a href="https://lazyeconomist.com" target="_self">' in page
    assert [t.value for t in at.title] == ["States"]
