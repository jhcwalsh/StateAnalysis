"""The lazyeconomist.com site theme, shared by app.py and the docs pages.

Moved out of app.py so pages/1_Introduction.py and pages/2_Methodology.py can render the
same masthead and CSS without re-running the dashboard. Behaviour is unchanged from the
original inline block in app.py: same palette, same SITE_CSS, same masthead markup.
"""
import matplotlib.pyplot as plt
import streamlit as st

# ---- site theme: the lazyeconomist.com landing page tokens (fixed light) ----------
# The Streamlit side of the theme lives in .streamlit/config.toml. The hex tokens and the
# rcParams live in regime_v2.theme (one source for the engine's PNGs and these inline
# charts); the inline charts stay transparent so they sit on the page's own ground.
from regime_v2.theme import (  # noqa: E402  (app.py puts regime_v2 on the path first)
    ACCENT, ACCENT_SOFT, BG, BG_SOFT, GROWTH_C, INFL_C, INK, INK_FAINT, INK_SOFT, RC_INLINE, RULE,
)

INK_MUTED, SURFACE = INK_FAINT, BG
FIG_W = 12
plt.rcParams.update(RC_INLINE)

SITE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
/* The header is transparent but still a full-width bar over the top 60px of the page, and it
   swallowed every click on the masthead's "← lazyeconomist.com" link underneath it. Pass clicks
   through the empty bar; the status widget keeps its own so a running script can still be stopped. */
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"], header[data-testid="stHeader"] [data-testid="stToolbar"] {
  pointer-events: none; }
header[data-testid="stHeader"] [data-testid="stStatusWidget"] { pointer-events: auto; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 1.2rem; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
h1 { font-size: 2.3rem !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stCaption, [data-testid="stCaptionContainer"] {
  font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stCaptionContainer"] { color: #8a8780 !important; font-size: 0.78rem; }
[data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; }
.le-topbar { display: flex; justify-content: space-between; align-items: baseline;
  border-bottom: 1px solid #e8e4dc; padding: 0 0 0.6rem 0; margin-bottom: 1.4rem;
  font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #8a8780; letter-spacing: 0.04em; }
.le-topbar a { color: #b8410e; text-decoration: none; }
.le-topbar a:hover { text-decoration: underline; }
/* Streamlit's automatic sidebar nav labels the entrypoint "app"; the masthead nav
   row replaces it, so the sidebar and its toggle are hidden on every page. */
[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.le-nav { display: flex; gap: 1.6rem; margin: -0.2rem 0 1.4rem 0; padding-bottom: 0.6rem;
  border-bottom: 1px solid #e8e4dc; font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
.le-nav a { color: #b8410e; text-decoration: none; }
.le-nav a:hover { text-decoration: underline; }
.le-nav-current { color: #1a1a1a; font-weight: 600; }
.le-banner { color: #fbfaf7; padding: 1.1em 1.2em; border-radius: 6px; font-family: 'Fraunces', Georgia, serif;
  font-size: 1.6rem; font-weight: 500; letter-spacing: -0.01em; }
.le-banner small { display: block; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em;
  opacity: 0.85; margin-bottom: 0.25rem; }
</style>
"""

DEFAULT_SUBTITLE = ("Which macro regime the US is in, from a walk-forward model on the FRED-MD panel — and what "
                     "that signal is worth beside a plain 60/40.")


def _masthead(tag="005 · MACRO", title="States", subtitle=DEFAULT_SUBTITLE):
    st.markdown(SITE_CSS, unsafe_allow_html=True)
    # target="_self" on purpose: Streamlit rewrites an anchor with no target to target="_blank",
    # so the link back to the landing page opened a new tab on every page of the site.
    st.markdown(f'<div class="le-topbar"><span>THE LAZY ECONOMIST · {tag}</span>'
                '<a href="https://lazyeconomist.com" target="_self">← lazyeconomist.com</a></div>',
                unsafe_allow_html=True)
    st.title(title)
    st.markdown(subtitle)


SITE_PAGES = [("States", "/"), ("Introduction", "/Introduction"), ("Methodology", "/Methodology")]


def _site_pages_nav(current=None):
    """The site's page nav, rendered under the masthead on every page.

    Streamlit's automatic sidebar nav is hidden by SITE_CSS (it labels the entrypoint
    "app"); this row of links replaces it. The links are plain same-tab anchors to the
    multipage URLs (st.markdown's own links open a new tab), and the current page is
    shown as text, not a link.
    """
    items = []
    for label, url in SITE_PAGES:
        if label == current:
            items.append(f"<span class='le-nav-current'>{label}</span>")
        else:
            items.append(f"<a href='{url}' target='_self'>{label}</a>")
    st.markdown("<div class='le-nav'>" + "".join(items) + "</div>", unsafe_allow_html=True)
