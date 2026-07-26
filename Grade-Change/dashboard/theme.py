"""
HMI theme layer — dark industrial instrument-panel aesthetic.

All dashboard tabs import from here so the visual language is consistent
across the entire application.  No default Streamlit alert boxes appear
anywhere in the UI — every callout uses .hmi-panel or .hmi-badge instead.

Palette (functional status-color system):
    Canvas          #0B1420
    Panel surface   #131F2E
    Primary data    #2DD4C8  (teal)
    Safe green      #34D399
    Caution amber   #F5A623
    Alarm red       #E5484D
"""

import streamlit as st


# ── color constants ───────────────────────────────────────────────────────────
COLOR_CANVAS   = "#0B1420"
COLOR_PANEL    = "#131F2E"
COLOR_PRIMARY  = "#2DD4C8"
COLOR_SAFE     = "#34D399"
COLOR_CAUTION  = "#F5A623"
COLOR_ALARM    = "#E5484D"
COLOR_TEXT     = "#E0E6ED"
COLOR_MUTED    = "#8A9BB0"
COLOR_BORDER   = "#1E2D40"


def inject_css():
    """
    Inject the full HMI CSS into the Streamlit page.
    Call once from app.py immediately after set_page_config.
    """
    st.markdown(
        f"""
        <style>
        /* ── Google Fonts — IBM Plex (degrade gracefully offline) ── */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

        /* ── root resets ── */
        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
            color: {COLOR_TEXT};
        }}

        /* ── app background ── */
        .stApp {{
            background-color: {COLOR_CANVAS};
        }}

        /* ── sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_PANEL};
            border-right: 1px solid {COLOR_BORDER};
        }}

        /* ── tab bar ── */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {COLOR_PANEL};
            border-radius: 8px 8px 0 0;
            padding: 4px 8px 0;
            gap: 4px;
            border-bottom: 2px solid {COLOR_PRIMARY};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {COLOR_MUTED};
            border-radius: 6px 6px 0 0;
            padding: 8px 18px;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.03em;
            border: none;
            transition: all 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_PRIMARY}18;
            color: {COLOR_PRIMARY} !important;
            border-bottom: 2px solid {COLOR_PRIMARY};
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {COLOR_TEXT} !important;
            background-color: {COLOR_BORDER};
        }}

        /* ── metric boxes ── */
        [data-testid="stMetric"] {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 16px 20px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLOR_MUTED};
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace;
            color: {COLOR_PRIMARY};
            font-size: 1.6rem;
        }}

        /* ── buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, {COLOR_PRIMARY}22, {COLOR_PRIMARY}11);
            border: 1px solid {COLOR_PRIMARY}60;
            color: {COLOR_PRIMARY};
            border-radius: 6px;
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 500;
            padding: 0.45rem 1.2rem;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLOR_PRIMARY}40, {COLOR_PRIMARY}22);
            border-color: {COLOR_PRIMARY};
            box-shadow: 0 0 12px {COLOR_PRIMARY}44;
        }}

        /* ── selectbox / dropdown ── */
        .stSelectbox > div > div {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            color: {COLOR_TEXT};
        }}

        /* ── dataframe / table ── */
        .stDataFrame {{
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            overflow: hidden;
        }}

        /* ── HMI PANEL component ── */
        .hmi-panel {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: 10px;
            padding: 20px 24px;
            margin: 12px 0;
            position: relative;
        }}
        .hmi-panel::before {{
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 3px; height: 100%;
            background: {COLOR_PRIMARY};
            border-radius: 10px 0 0 10px;
        }}
        .hmi-panel-title {{
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {COLOR_PRIMARY};
            margin-bottom: 10px;
        }}
        .hmi-panel p {{
            color: {COLOR_TEXT};
            margin: 0;
            line-height: 1.6;
        }}

        /* ── HMI BADGE component ── */
        .hmi-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.05em;
        }}
        .hmi-badge-known {{
            background-color: {COLOR_PRIMARY}22;
            color: {COLOR_PRIMARY};
            border: 1px solid {COLOR_PRIMARY}44;
        }}
        .hmi-badge-discovered {{
            background-color: {COLOR_CAUTION}22;
            color: {COLOR_CAUTION};
            border: 1px solid {COLOR_CAUTION}44;
        }}
        .hmi-badge-safe {{
            background-color: {COLOR_SAFE}22;
            color: {COLOR_SAFE};
            border: 1px solid {COLOR_SAFE}44;
        }}
        .hmi-badge-alarm {{
            background-color: {COLOR_ALARM}22;
            color: {COLOR_ALARM};
            border: 1px solid {COLOR_ALARM}44;
        }}

        /* ── section headers ── */
        .hmi-section-header {{
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: {COLOR_MUTED};
            border-bottom: 1px solid {COLOR_BORDER};
            padding-bottom: 6px;
            margin: 20px 0 12px;
        }}

        /* ── numeric readouts (mono) ── */
        .hmi-mono {{
            font-family: 'IBM Plex Mono', monospace;
            color: {COLOR_PRIMARY};
        }}

        /* ── recommendation card ── */
        .rec-card {{
            background: linear-gradient(135deg, {COLOR_PANEL}, {COLOR_CANVAS});
            border: 1px solid {COLOR_BORDER};
            border-left: 3px solid {COLOR_CAUTION};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 8px 0;
        }}
        .rec-card-header {{
            font-size: 0.8rem;
            font-weight: 600;
            color: {COLOR_CAUTION};
            margin-bottom: 6px;
        }}
        .rec-card-rationale {{
            font-size: 0.73rem;
            color: {COLOR_MUTED};
            font-style: italic;
            margin-top: 6px;
        }}

        /* ── accepted/rejected row coloring in tables ── */
        .accepted-row {{ color: {COLOR_SAFE}; }}
        .rejected-row {{ color: {COLOR_ALARM}; }}

        /* ── hide Streamlit default hamburger menu ── */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hmi_panel(title, content_html):
    """Render a styled HMI panel block."""
    st.markdown(
        f"""
        <div class="hmi-panel">
            <div class="hmi-panel-title">{title}</div>
            <p>{content_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hmi_badge(text, kind="known"):
    """
    Return an inline badge HTML string.

    kind: 'known' | 'discovered' | 'safe' | 'alarm'
    """
    css_class = {
        "known":      "hmi-badge hmi-badge-known",
        "discovered": "hmi-badge hmi-badge-discovered",
        "safe":       "hmi-badge hmi-badge-safe",
        "alarm":      "hmi-badge hmi-badge-alarm",
    }.get(kind, "hmi-badge hmi-badge-known")
    return f'<span class="{css_class}">{text}</span>'


def section_header(text):
    """Render a muted all-caps section divider."""
    st.markdown(f'<div class="hmi-section-header">{text}</div>', unsafe_allow_html=True)


def status_color(risk_prob):
    """Return COLOR_SAFE / COLOR_CAUTION / COLOR_ALARM based on probability."""
    if risk_prob < 0.35:
        return COLOR_SAFE
    elif risk_prob < 0.65:
        return COLOR_CAUTION
    else:
        return COLOR_ALARM
