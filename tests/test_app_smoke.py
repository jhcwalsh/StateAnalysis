import os
from streamlit.testing.v1 import AppTest


def test_app_renders_status(ui_data_dir, monkeypatch):
    monkeypatch.setenv("UI_DATA_DIR", ui_data_dir)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    # st.caption content lands in at.caption, not at.markdown — include both.
    page = " ".join(md.value for md in at.markdown) + \
        " ".join(c.value for c in at.caption)
    assert "Overheating" in page          # current regime rendered
    assert "2021-12" in page              # data vintage rendered


def test_app_chart_tabs(ui_data_dir, monkeypatch):
    monkeypatch.setenv("UI_DATA_DIR", ui_data_dir)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert "Factors" in labels
    assert "Probabilities" in labels
    assert "State space" in labels
    captions = " ".join(c.value for c in at.caption)
    assert "GMM" in captions   # probs pinned-to-run note rendered


def test_app_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_DATA_DIR", str(tmp_path / "nowhere"))
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    page = " ".join(md.value for md in at.markdown)
    assert "No cached run found" in page
