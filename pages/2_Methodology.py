"""Methodology — the full method paper, rendered from docs/site/methodology.md with live
numbers from the published run (contract: docs/site/CONTRACT.md).

Loads the run exactly as app.py does: same REGIME_OUTPUT_DIR/REGIME_FIGS_DIR env vars, same
regime_v2.publish.load_published, same empty-state messages.
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "regime_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENGINE))
from regime_v2 import publish, sitedocs  # noqa: E402  (path set above)
from site_theme import _masthead, _site_pages_nav  # noqa: E402

PAGE_NAME = "methodology"
PAGE_TITLE = "Methodology"
SUBTITLE = "The full method: data, factors, HMM, walk-forward, assets, and the honesty checks."


def _dir(env, default):
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else ROOT / p


OUT_DIR = _dir("REGIME_OUTPUT_DIR", "regime_v2/output")
FIGS_DIR = _dir("REGIME_FIGS_DIR", "regime_v2/figs")
DOCS_DIR = _dir("SITE_DOCS_DIR", "docs/site")

st.set_page_config(page_title=f"{PAGE_TITLE} · The Lazy Economist", layout="wide",
                   initial_sidebar_state="collapsed")


@st.cache_data
def _load(mtime, out_dir, figs_dir):
    return publish.load_published(out_dir, figs_dir)


def _empty_state(message):
    _masthead(title=PAGE_TITLE, subtitle=SUBTITLE)
    st.markdown(message)
    st.stop()


try:
    pub = _load(publish.published_mtime(OUT_DIR), str(OUT_DIR), str(FIGS_DIR))
except publish.PublishedMissing:
    _empty_state("**No published run found.** Run the engine once to publish `output/` and `figs/`.")

# An explicit test, not a bare `pub.summary["current"], pub.summary["run"]` expression:
# Streamlit's magic rendering would print any bare expression to the page.
if "current" not in pub.summary or "run" not in pub.summary:
    _empty_state("The published run predates the current contract (no `current`/`run` block). "
                 "Press Refresh on the main page to republish.")

_masthead(title=PAGE_TITLE, subtitle=SUBTITLE)
_site_pages_nav(current=PAGE_TITLE)

doc_path = DOCS_DIR / f"{PAGE_NAME}.md"
if doc_path.exists():
    md_text = doc_path.read_text(encoding="utf-8")
else:
    md_text = f"*No document found at `{doc_path}`. Once `docs/site/{PAGE_NAME}.md` is written, it renders here.*"

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
html_doc = sitedocs.to_html(md_text, nums, pub.figures, f"{PAGE_TITLE} — The Lazy Economist")
st.download_button(f"Download {PAGE_TITLE} as HTML", data=html_doc.encode("utf-8"),
                   file_name=f"{PAGE_NAME}.html", mime="text/html")
st.caption("Opens in any browser; print to PDF from there.")
