"""Methodology — the full method paper, rendered from docs/site/methodology.md with live
numbers from the published run (contract: docs/site/CONTRACT.md).

The page body is `site_pages.render_doc_page`, shared with pages/1_Introduction.py.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "regime_v2"))

PAGE_NAME = "methodology"
PAGE_TITLE = "Methodology"
SUBTITLE = "The full method: data, factors, HMM, walk-forward, assets, and the honesty checks."

# First Streamlit call on the page, before anything the shared body renders.
st.set_page_config(page_title=f"{PAGE_TITLE} · The Lazy Economist", layout="wide",
                   initial_sidebar_state="collapsed")

from site_pages import render_doc_page  # noqa: E402  (path set above; set_page_config comes first)

render_doc_page(PAGE_NAME, PAGE_TITLE, SUBTITLE)
