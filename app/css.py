"""Global CSS design system.

Palette
───────
  --blue      #4F8EF7   brand / primary accent
  --violet    #7C3AED   brand / secondary accent
  --navy-0    #0D1B2A   darkest surface (sidebar top)
  --navy-1    #132236   deep navy (sidebar bottom)
  --navy-2    #1C2E45   card bg on dark surfaces
  --navy-3    #243555   lighter card bg
  --navy-4    #344D73   border / muted on dark
  --surface   #EEF3FC   card bg on light pages (matches config.toml)
  --border    #D0DEFA   subtle border on light pages

Semantic
  --success   #10B981
  --warning   #F59E0B
  --danger    #EF4444
  --info      #4F8EF7

Score colours (1-5)
  1  #059669   Good
  2  #65A30D   Mild
  3  #D97706   Moderate
  4  #EA580C   Severe
  5  #DC2626   Critical
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ────────────────────────────────────────────────────── */
:root {
  --blue:    #4F8EF7;
  --violet:  #7C3AED;
  --navy-0:  #0D1B2A;
  --navy-1:  #132236;
  --navy-2:  #1C2E45;
  --navy-3:  #243555;
  --navy-4:  #344D73;
  --surface: #EEF3FC;
  --border:  #D0DEFA;
  --success: #10B981;
  --warning: #F59E0B;
  --danger:  #EF4444;
  --text:    #1A2C50;
  --text-2:  #3D5A8A;
}

/* ── Typography ────────────────────────────────────────────────────────── */
html, body,
.stApp, .stMarkdown, .stTextInput, .stSelectbox, .stButton,
.stDataFrame, .stTextArea, .stNumberInput, .stRadio,
p, h1, h2, h3, h4, h5, h6, div, label, input, textarea, select, td, th, li,
.stExpander summary span {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Protect Streamlit's icon fonts */
span[data-testid="stIconMaterial"],
span[data-testid="stIcon"],
.material-symbols-rounded,
[class*="material-symbols"],
[data-testid="stExpanderToggleIcon"],
[data-testid="collapsedControl"] span,
button[kind="header"] span,
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stMainMenu"] span,
[data-testid="stHeaderActionElements"] span {
    font-family: 'Material Symbols Rounded', sans-serif !important;
}

/* ── Layout ────────────────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #132236 60%, #1C2E45 100%);
    border-right: 1px solid rgba(79, 142, 247, 0.15);
}

/* Sidebar text */
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label {
    color: #C8D8F0 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(79, 142, 247, 0.2) !important;
}

/* Sidebar nav buttons — active (primary) */
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background: linear-gradient(135deg,
        rgba(79, 142, 247, 0.22) 0%,
        rgba(124, 58, 237, 0.16) 100%) !important;
    border-left:   3px solid #4F8EF7 !important;
    border-top:    1px solid rgba(79, 142, 247, 0.35) !important;
    border-right:  1px solid rgba(79, 142, 247, 0.35) !important;
    border-bottom: 1px solid rgba(79, 142, 247, 0.35) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(79, 142, 247, 0.18) !important;
}

/* Sidebar nav buttons — inactive (secondary) */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(155, 175, 206, 0.18) !important;
    color: #8AAACE !important;
    font-weight: 400 !important;
}

section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(79, 142, 247, 0.12) !important;
    border-color: rgba(79, 142, 247, 0.35) !important;
    color: #E0ECFF !important;
}

/* ── Section divider ───────────────────────────────────────────────────── */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg,
        #4F8EF7 0%,
        #7C3AED 50%,
        transparent 100%);
    border: none;
    margin: 1.75rem 0;
    border-radius: 2px;
    opacity: 0.7;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #4F8EF7 0%, #7C3AED 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 24px rgba(79, 142, 247, 0.28);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(79, 142, 247, 0.38);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    opacity: 0.88;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.metric-card-blue {
    background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.28);
}
.metric-card-green {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    box-shadow: 0 8px 24px rgba(5, 150, 105, 0.28);
}
.metric-card-orange {
    background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
    box-shadow: 0 8px 24px rgba(217, 119, 6, 0.28);
}
.metric-card-red {
    background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    box-shadow: 0 8px 24px rgba(220, 38, 38, 0.28);
}

/* ── Result badges ─────────────────────────────────────────────────────── */
.badge-yes {
    display: inline-block;
    background: linear-gradient(135deg, #DC2626, #EF4444);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
}
.badge-no {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #10B981);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
}
.badge-unknown {
    display: inline-block;
    background: linear-gradient(135deg, #64748B, #94A3B8);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.05rem;
}

/* ── Agent trace cards ─────────────────────────────────────────────────── */
.trace-card {
    background: var(--surface, #EEF3FC);
    border-left: 4px solid #4F8EF7;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.trace-card-header {
    font-weight: 700;
    color: #4F8EF7;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface, #EEF3FC);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border, #D0DEFA);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: 0.45rem 1rem;
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--text-2, #3D5A8A);
    background: transparent;
    border: none;
    transition: background 0.15s, color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4F8EF7 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(79, 142, 247, 0.18) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
.stExpander {
    border: 1px solid var(--border, #D0DEFA) !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    margin-bottom: 0.5rem !important;
}
.stExpander summary {
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
}
.stExpander summary:hover {
    background: var(--surface, #EEF3FC) !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border, #D0DEFA);
    border-radius: 10px;
    overflow: hidden;
}

/* ── Hide deploy button ────────────────────────────────────────────────── */
.stAppDeployButton,
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(79, 142, 247, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(79, 142, 247, 0.55);
}
</style>
"""
