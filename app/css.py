"""Global CSS stylesheet injected once at startup."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body,
.stApp, .stMarkdown, .stTextInput, .stSelectbox, .stButton,
.stDataFrame, .stTextArea, .stNumberInput, .stRadio,
p, h1, h2, h3, h4, h5, h6, div, label, input, textarea, select, td, th, li,
.stExpander summary span {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
/* Protect Streamlit's Material Symbols icon font */
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

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    font-weight: 400;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-card-blue {
    background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    box-shadow: 0 10px 30px rgba(33, 147, 176, 0.3);
}
.metric-card-green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 10px 30px rgba(17, 153, 142, 0.3);
}
.metric-card-orange {
    background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    box-shadow: 0 10px 30px rgba(242, 153, 74, 0.3);
}
.metric-card-red {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    box-shadow: 0 10px 30px rgba(235, 51, 73, 0.3);
}

/* Result badges */
.badge-yes {
    display: inline-block;
    background: linear-gradient(135deg, #eb3349, #f45c43);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.badge-no {
    display: inline-block;
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.badge-unknown {
    display: inline-block;
    background: linear-gradient(135deg, #bdc3c7, #95a5a6);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label {
    color: #e0e0e0 !important;
}

/* Agent trace cards */
.trace-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.trace-card-header {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}

/* Divider */
.section-divider {
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, transparent);
    border: none;
    margin: 2rem 0;
    border-radius: 2px;
}
    /* Hide only the Streamlit Deploy button, keep the header for theme settings */
    .stAppDeployButton {
        display: none !important;
    }
    [data-testid="stAppDeployButton"] {
        display: none !important;
    }
</style>
"""
