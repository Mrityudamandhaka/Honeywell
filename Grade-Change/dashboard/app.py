"""
Grade Change Intelligence — Streamlit Dashboard Entry Point

Honeywell Hackathon project: predictive deviation risk, correlation discovery,
stabilization-time analysis, and operator feedback loop for paper-machine
grade changes.

Run with:
    python -m data.generate_data      # first time only
    streamlit run dashboard/app.py
"""

import functools
import sys
import os

# ensure project root is importable regardless of how streamlit launches
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── page config — must be first streamlit call ────────────────────────────
st.set_page_config(
    page_title="Grade Change Intelligence — Honeywell QCS",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── import HMI theme and inject CSS ──────────────────────────────────────
from dashboard.theme import inject_css, COLOR_PRIMARY, COLOR_MUTED, COLOR_PANEL, COLOR_BORDER, COLOR_TEXT
inject_css()

# ── business-logic imports (no logic in dashboard/*.py) ──────────────────
from data.loader import load_events
from ai.features import build_features
from ai.model import train_model
from ai.recommend import recommend_setpoints
from analytics.correlations import discover_correlations
from analytics.stabilization import stabilization_drivers
from analytics.projection import fit_settle_calibration, calibration_backtest_mae

# ── tab renderers ─────────────────────────────────────────────────────────
import dashboard.tab_model_performance   as tab1
import dashboard.tab_correlation_discovery as tab2
import dashboard.tab_stabilization       as tab3
import dashboard.tab_live_risk           as tab4
import dashboard.tab_feedback_log        as tab5


# ── cached data loading and model training ────────────────────────────────
@st.cache_data(show_spinner="Loading process data…")
def _load_data():
    return load_events()


@st.cache_data(show_spinner="Building feature table…")
def _build_features(_df):
    return build_features(_df)


@st.cache_data(show_spinner="Training risk model…")
def _train_model(_features_df):
    return train_model(_features_df)


@st.cache_data(show_spinner="Running correlation analysis…")
def _correlations(_features_df):
    return discover_correlations(_features_df)


@st.cache_data(show_spinner="Computing stabilization drivers…")
def _stab_drivers(_features_df):
    return stabilization_drivers(_features_df)


@st.cache_data(show_spinner="Calibrating settle-time projection…")
def _calibration(_features_df, _raw_df):
    return fit_settle_calibration(_features_df, _raw_df)


# ── header ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;
                padding:18px 0 8px;border-bottom:1px solid {COLOR_BORDER};
                margin-bottom:4px;">
        <div style="font-size:2.2rem;line-height:1;">⚙</div>
        <div>
            <div style="font-size:1.4rem;font-weight:700;
                        color:{COLOR_PRIMARY};letter-spacing:0.02em;">
                Grade Change Intelligence
            </div>
            <div style="font-size:0.78rem;color:{COLOR_MUTED};
                        letter-spacing:0.05em;margin-top:2px;">
                HONEYWELL QCS · PAPER MACHINE GRADE-CHANGE ANALYSIS SYSTEM
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── data pipeline ─────────────────────────────────────────────────────────
data_ok = True
try:
    raw_df = _load_data()
except FileNotFoundError:
    st.error(
        "**process_data.csv not found.**  "
        "Run `python -m data.generate_data` from the project root first, then refresh."
    )
    data_ok = False

if data_ok:
    features_df  = _build_features(raw_df)
    model_result = _train_model(features_df)
    corr_df      = _correlations(features_df)
    stab_df      = _stab_drivers(features_df)
    calibration  = _calibration(features_df, raw_df)
    calib_mae    = calibration_backtest_mae(features_df, raw_df, calibration)

    # partial for Tab 4 — binds model_result and features_df
    def _recs_for_event(feat_row):
        return recommend_setpoints(feat_row, model_result, features_df)

    # ── 5-tab layout ─────────────────────────────────────────────────
    tab_labels = [
        "📊 Model Performance",
        "🔍 Correlation Discovery",
        "⏱ Stabilization",
        "🚨 Live Risk",
        "📋 Feedback Log",
    ]

    t1, t2, t3, t4, t5 = st.tabs(tab_labels)

    with t1:
        tab1.render(model_result)

    with t2:
        tab2.render(corr_df)

    with t3:
        tab3.render(stab_df, features_df)

    with t4:
        tab4.render(
            raw_df, features_df, model_result,
            _recs_for_event, calibration, calib_mae,
        )

    with t5:
        tab5.render()
