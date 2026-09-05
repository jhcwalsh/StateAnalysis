"""Smoke tests for pages/1_Introduction.py and pages/2_Methodology.py.

Uses the root conftest's published_dir fixture (one real engine run per session) plus a
temporary docs/site/*.md stub pointed to via SITE_DOCS_DIR, since the real Markdown
documents are written by a separate workstream and may not exist yet.
"""
import re
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

STUB_MARKER_INTRO = "STUB-INTRODUCTION-MARKER"
STUB_MARKER_METH = "STUB-METHODOLOGY-MARKER"


def _write_stub_docs(tmp_path):
    docs_dir = tmp_path / "site_docs"
    docs_dir.mkdir()
    (docs_dir / "introduction.md").write_text(
        f"# Introduction\n\n{STUB_MARKER_INTRO}: the current regime is {{{{current.regime}}}}.\n", encoding="utf-8")
    (docs_dir / "methodology.md").write_text(
        f"# Methodology\n\n{STUB_MARKER_METH}: sample starts {{{{sample.start}}}}.\n", encoding="utf-8")
    return docs_dir


def _run_page(monkeypatch, out, figs, docs_dir, page):
    monkeypatch.setenv("REGIME_OUTPUT_DIR", str(out))
    monkeypatch.setenv("REGIME_FIGS_DIR", str(figs))
    monkeypatch.setenv("SITE_DOCS_DIR", str(docs_dir))
    return AppTest.from_file(page, default_timeout=120).run()


def _page_text(at):
    return " ".join(md.value for md in at.markdown) + " ".join(c.value for c in at.caption)


def test_introduction_page_renders(published_dir, monkeypatch, tmp_path):
    out, figs = published_dir
    docs_dir = _write_stub_docs(tmp_path)
    at = _run_page(monkeypatch, out, figs, docs_dir, "pages/1_Introduction.py")
    assert not at.exception
    assert [t.value for t in at.title] == ["Introduction"]
    text = _page_text(at)
    assert "lazyeconomist.com" in text            # shared masthead
    assert STUB_MARKER_INTRO in text
    assert "[missing:" not in text                # the stub's placeholder resolved
    assert at.download_button, "the Introduction page must offer a download button"


def test_methodology_page_renders(published_dir, monkeypatch, tmp_path):
    out, figs = published_dir
    docs_dir = _write_stub_docs(tmp_path)
    at = _run_page(monkeypatch, out, figs, docs_dir, "pages/2_Methodology.py")
    assert not at.exception
    assert [t.value for t in at.title] == ["Methodology"]
    text = _page_text(at)
    assert "lazyeconomist.com" in text
    assert STUB_MARKER_METH in text
    assert "[missing:" not in text
    assert at.download_button, "the Methodology page must offer a download button"


def test_pages_empty_state(monkeypatch, tmp_path):
    docs_dir = _write_stub_docs(tmp_path)
    at = _run_page(monkeypatch, tmp_path / "nowhere", tmp_path / "nofigs", docs_dir, "pages/1_Introduction.py")
    assert not at.exception
    text = _page_text(at)
    assert "No published run" in text


def test_app_masthead_links_to_the_two_pages(published_dir, monkeypatch):
    out, figs = published_dir
    monkeypatch.setenv("REGIME_OUTPUT_DIR", str(out))
    monkeypatch.setenv("REGIME_FIGS_DIR", str(figs))
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception
    # The masthead nav is a row of plain same-tab anchors (see site_theme._site_pages_nav).
    text = _page_text(at)
    assert "href='/Introduction'" in text and "href='/Methodology'" in text


def test_real_docs_have_no_missing_placeholders_and_known_figures(published_dir):
    sys.path.insert(0, str(ROOT / "regime_v2"))
    from regime_v2 import publish, sitedocs

    known_figs = set(publish.FIGURES) | {"doc_pipeline", "doc_quadrants", "doc_timing", "doc_lookahead",
                                          "doc_placebo", "doc_loadings", "doc_transition"}
    doc_paths = {name: ROOT / "docs" / "site" / f"{name}.md" for name in ("introduction", "methodology")}
    existing = {name: p for name, p in doc_paths.items() if p.exists()}
    if not existing:
        pytest.skip("docs/site/introduction.md and methodology.md do not exist yet")

    out, figs = published_dir
    pub = publish.load_published(out, figs)
    nums = sitedocs.numbers(pub)
    for name, path in existing.items():
        text = path.read_text(encoding="utf-8")
        missing = sitedocs.missing_placeholders(text, nums)
        assert not missing, f"{name}.md references unknown placeholders: {missing}"
        used = set(re.findall(r"\(fig:([A-Za-z0-9_]+)\)", text))
        unknown = used - known_figs
        assert not unknown, f"{name}.md references unknown figure names: {unknown}"
