"""The lazyeconomist.com site theme, shared by app.py and the docs pages.

Moved out of app.py so pages/1_Introduction.py and pages/2_Methodology.py can render the
same masthead and CSS without re-running the dashboard. Behaviour is unchanged from the
original inline block in app.py: same palette, same SITE_CSS, same masthead markup.
"""
import matplotlib.pyplot as plt
import streamlit as st

# ---- site theme: the lazyeconomist.com landing page tokens (fixed light) ----------
# The Streamlit side of the theme lives in .streamlit/config.toml; these constants
# drive the matplotlib figures and the few inline styles so charts match the page.
BG, BG_SOFT = "#fbfaf7", "#f4f2ec"
INK, INK_SOFT, INK_FAINT, RULE = "#1a1a1a", "#4a4a4a", "#8a8780", "#e8e4dc"
ACCENT, ACCENT_SOFT = "#b8410e", "#f5e6dd"
INK_MUTED, SURFACE = INK_FAINT, BG
FIG_W = 12
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "text.color": INK, "axes.titlecolor": INK,
    "axes.labelcolor": INK_SOFT, "axes.titlesize": 10, "axes.edgecolor": RULE, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK_FAINT, "ytick.color": INK_FAINT, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.labelcolor": INK, "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": RULE, "grid.alpha": 0.9, "grid.linewidth": 0.6,
})
GROWTH_C, INFL_C = "#2C7FB8", "#D95F0E"

SITE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
header[data-testid="stHeader"] { background: transparent; }
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
    st.markdown(f'<div class="le-topbar"><span>THE LAZY ECONOMIST · {tag}</span>'
                '<a href="https://lazyeconomist.com">← lazyeconomist.com</a></div>', unsafe_allow_html=True)
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
