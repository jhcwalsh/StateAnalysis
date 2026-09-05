"""The shared body of the two document pages (pages/1_Introduction.py, pages/2_Methodology.py).

Both pages are the same page with a different Markdown source, so the body lives here once:
load the published run exactly as app.py does (same REGIME_OUTPUT_DIR / REGIME_FIGS_DIR env
vars, same regime_v2.publish.load_published, same empty-state messages), render
docs/site/<name>.md through sitedocs against it, and offer the self-contained HTML export.

The export is cached on the published run's mtime and the document text: building it embeds
every figure as base64 (~2 MB for the paper) and would otherwise be redone on every rerun,
including the rerun the download button itself triggers.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "regime_v2"
for _p in (str(ROOT), str(ENGINE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from regime_v2 import publish, sitedocs  # noqa: E402  (path set above)
from site_theme import _masthead, _site_pages_nav  # noqa: E402


def _dir(env, default):
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


@st.cache_data
def _load(mtime, out_dir, figs_dir):
    return publish.load_published(out_dir, figs_dir)


@st.cache_data
def _export_html(mtime, md_text, out_dir, figs_dir, title) -> bytes:
    """The downloadable single-file export, keyed on the run and the document text."""
    pub = _load(mtime, out_dir, figs_dir)
    return sitedocs.to_html(md_text, sitedocs.numbers(pub), pub.figures, title).encode("utf-8")


def render_doc_page(page_name: str, page_title: str, subtitle: str) -> None:
    """Render docs/site/<page_name>.md as a page of the site. Call after set_page_config."""
    out_dir = _dir("REGIME_OUTPUT_DIR", "regime_v2/output")
    figs_dir = _dir("REGIME_FIGS_DIR", "regime_v2/figs")
    docs_dir = _dir("SITE_DOCS_DIR", "docs/site")

    def _empty_state(message):
        # The nav goes with the masthead here too: the sidebar is hidden site-wide, so a
        # visitor landing on this page without a published run would otherwise have no way
        # back to the main page the message tells them to press Refresh on.
        _masthead(title=page_title, subtitle=subtitle)
        _site_pages_nav(current=page_title)
        st.markdown(message)
        st.stop()

    mtime = publish.published_mtime(out_dir)
    try:
        pub = _load(mtime, str(out_dir), str(figs_dir))
    except publish.PublishedMissing:
        _empty_state("**No published run found.** Run the engine once to publish `output/` and `figs/`.")

    # An explicit test, not a bare `pub.summary["current"], pub.summary["run"]` expression:
    # Streamlit's magic rendering would print any bare expression to the page.
    if "current" not in pub.summary or "run" not in pub.summary:
        _empty_state("The published run predates the current contract (no `current`/`run` block). "
                     "Press Refresh on the main page to republish.")

    _masthead(title=page_title, subtitle=subtitle)
    _site_pages_nav(current=page_title)

    doc_path = docs_dir / f"{page_name}.md"
    if doc_path.exists():
        md_text = doc_path.read_text(encoding="utf-8")
    else:
        md_text = f"*No document found at `{doc_path}`. Once `docs/site/{page_name}.md` is written, it renders here.*"

    nums = sitedocs.numbers(pub)
    blocks = sitedocs.render(md_text, nums, pub.figures)
    for block in blocks:
        if block[0] == "md":
            st.markdown(block[1])
        else:
            _, path, caption, name = block
            if path is not None:
                st.image(str(path), caption=caption or None)
            else:
                st.markdown(f"[missing figure: {name}]")

    st.divider()
    html_doc = _export_html(mtime, md_text, str(out_dir), str(figs_dir),
                            f"{page_title} — The Lazy Economist")
    st.download_button(f"Download {page_title} as HTML", data=html_doc,
                       file_name=f"{page_name}.html", mime="text/html")
    st.caption("Opens in any browser; print to PDF from there.")
