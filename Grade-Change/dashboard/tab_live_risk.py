"""
Tab 4 — Live Risk Assessment

The operational heart of the dashboard.  For any selected grade-change event:

  1. Analog gauge (Plotly Indicator) showing deviation risk probability
  2. Deviation trajectory chart with ±2.5% spec band
  3. Calibrated settle-time projection with measured backtest error stated inline
  4. Setpoint recommendations, each source-tagged and with Accept/Reject buttons
     that write immediately to the feedback SQLite database

The Accept/Reject buttons are the mechanism for evaluating suggestion quality
over time — every click persists, nothing lives only in session state.
"""

import plotly.graph_objects as go
import streamlit as st
import numpy as np
from analytics.projection import project_settle_time
from backend.feedback import log_feedback
from dashboard.theme import (
    COLOR_ALARM, COLOR_BORDER, COLOR_CAUTION, COLOR_MUTED,
    COLOR_PANEL, COLOR_PRIMARY, COLOR_SAFE, COLOR_TEXT, COLOR_CANVAS,
    hmi_badge, hmi_panel, section_header, status_color,
)


DEVIATION_THRESHOLD = 0.025


def render(raw_df, features_df, model_result, recommendations_fn, calibration, calib_mae):
    """
    Render the Live Risk Assessment tab.

    Parameters
    ----------
    raw_df : pd.DataFrame            Minute-by-minute process data
    features_df : pd.DataFrame       Feature table (one row per event)
    model_result : dict              Output of ai.model.train_model()
    recommendations_fn : callable    ai.recommend.recommend_setpoints (partial)
    calibration : dict               Output of analytics.projection.fit_settle_calibration()
    calib_mae : float                Post-calibration backtest MAE
    """
    section_header("Select Grade-Change Event")

    event_ids = sorted(raw_df["event_id"].unique())
    selected_id = st.selectbox(
        "Event ID",
        options=event_ids,
        format_func=lambda eid: _event_label(eid, raw_df),
        label_visibility="collapsed",
    )

    event_df   = raw_df[raw_df["event_id"] == selected_id].sort_values("minute")
    event_feat = features_df[features_df["event_id"] == selected_id]

    if event_feat.empty:
        st.warning("Feature data not available for this event.")
        return

    feat_row = event_feat.iloc[0]

    # ── model risk probability ─────────────────────────────────────────
    clf           = model_result["model"]
    feature_names = model_result["feature_names"]

    drop_cols   = ["event_id", "label_off_spec", "settle_t"]
    feat_vals   = feat_row[[c for c in features_df.columns if c not in drop_cols]].values.reshape(1, -1)
    risk_prob   = float(clf.predict_proba(feat_vals)[0][1])
    risk_label  = feat_row["label_off_spec"]
    settle_t    = int(feat_row["settle_t"])

    # calibrated projection
    raw_proj     = project_settle_time(event_df)
    calib_proj   = calibration["m"] * raw_proj + calibration["c"]
    calib_proj   = float(np.clip(calib_proj, 0, 30))

    # ── layout: gauge left, stats right ──────────────────────────────
    col_gauge, col_stats = st.columns([1, 1])

    with col_gauge:
        section_header("Risk Gauge — Early-Window Probability")
        gauge_color = status_color(risk_prob)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=round(risk_prob * 100, 1),
                number=dict(
                    suffix="%",
                    font=dict(family="IBM Plex Mono", color=gauge_color, size=42),
                ),
                delta=dict(
                    reference=50,
                    increasing=dict(color=COLOR_ALARM),
                    decreasing=dict(color=COLOR_SAFE),
                    font=dict(size=14),
                ),
                title=dict(
                    text="Off-Spec Probability",
                    font=dict(family="IBM Plex Sans", color=COLOR_MUTED, size=13),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, 100],
                        tickwidth=1,
                        tickcolor=COLOR_MUTED,
                        tickfont=dict(family="IBM Plex Mono", size=10, color=COLOR_MUTED),
                    ),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor=COLOR_CANVAS if True else "white",
                    borderwidth=0,
                    steps=[
                        dict(range=[0,  35],  color="rgba(52, 211, 153, 0.15)"),
                        dict(range=[35, 65],  color="rgba(245, 166, 35, 0.15)"),
                        dict(range=[65, 100], color="rgba(229, 72, 77, 0.15)"),
                    ],
                    threshold=dict(
                        line=dict(color=COLOR_ALARM, width=2),
                        thickness=0.75,
                        value=65,
                    ),
                ),
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor=COLOR_PANEL,
            font=dict(family="IBM Plex Mono", color=COLOR_TEXT),
            height=300,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_stats:
        section_header("Transition Summary")

        grade_from = event_df["grade_from"].values[0]
        grade_to   = event_df["grade_to"].values[0]
        bw_target  = event_df["basis_weight_target"].values[0]

        st.markdown(
            f"""
            <div class="hmi-panel" style="margin-top:0;">
                <div class="hmi-panel-title">Event {selected_id} — {grade_from} → {grade_to}</div>
                <p>
                  <span class="hmi-mono">BW Target</span>
                  &nbsp;{bw_target:.1f} g/m²<br><br>
                  <span class="hmi-mono">Actual Outcome</span>
                  &nbsp;{'<span style="color:' + COLOR_ALARM + ';">OFF-SPEC</span>' if risk_label else '<span style="color:' + COLOR_SAFE + ';">WITHIN SPEC</span>'}<br><br>
                  <span class="hmi-mono">Actual Settle Time</span>
                  &nbsp;{settle_t} min<br><br>
                  <span class="hmi-mono">Calibrated Projection</span>
                  &nbsp;{calib_proj:.1f} min<br>
                  <span style="font-size:0.72rem;color:{COLOR_MUTED};">
                    ± {calib_mae:.2f} min avg backtest error
                  </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── deviation trajectory ──────────────────────────────────────────
    section_header("Basis Weight Deviation Trajectory")

    bw       = event_df["basis_weight"].values
    bw_tgt   = event_df["basis_weight_target"].values[0]
    minutes  = event_df["minute"].values
    dev_pct  = (bw - bw_tgt) / abs(bw_tgt) * 100

    fig_traj = go.Figure()

    # spec band
    fig_traj.add_hrect(
        y0=-2.5, y1=2.5,
        fillcolor="rgba(52, 211, 153, 0.09)",
        line_width=0,
        annotation_text="  ±2.5% spec band",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_SAFE, size=10),
    )
    fig_traj.add_hline(y=2.5,  line_color=COLOR_SAFE, line_dash="dash", line_width=1)
    fig_traj.add_hline(y=-2.5, line_color=COLOR_SAFE, line_dash="dash", line_width=1)

    # early window shading (minutes 0-5 = model input window)
    fig_traj.add_vrect(
        x0=0, x1=5,
        fillcolor="rgba(45, 212, 200, 0.07)",
        line_width=0,
        annotation_text="  model input window",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_PRIMARY, size=10),
    )

    # deviation line
    fig_traj.add_trace(go.Scatter(
        x=minutes, y=dev_pct,
        mode="lines+markers",
        name="Deviation %",
        line=dict(color=COLOR_PRIMARY, width=2),
        marker=dict(size=4, color=COLOR_PRIMARY),
    ))

    # projected settle line
    if 0 <= calib_proj <= 24:
        fig_traj.add_vline(
            x=calib_proj,
            line_color=COLOR_CAUTION,
            line_dash="dot",
            line_width=1.5,
            annotation_text=f"  projected: {calib_proj:.1f} min",
            annotation_font=dict(color=COLOR_CAUTION, size=10),
        )

    fig_traj.add_hline(y=0, line_color=COLOR_BORDER, line_width=1)

    fig_traj.update_layout(
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="IBM Plex Sans", color=COLOR_TEXT, size=11),
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Minute", gridcolor=COLOR_BORDER, zeroline=False),
        yaxis=dict(title="Deviation from Target (%)", gridcolor=COLOR_BORDER, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── setpoint recommendations ──────────────────────────────────────
    section_header("Setpoint Recommendations")

    if risk_prob < 0.35:
        hmi_panel(
            "LOW RISK — NO CORRECTIVE ACTION NEEDED",
            f"Model confidence: <span class='hmi-mono'>{risk_prob:.1%}</span> off-spec probability. "
            "This transition looks stable based on early-window data. No setpoint corrections recommended.",
        )
        return

    recs = recommendations_fn(feat_row)

    if not recs:
        hmi_panel("NO CONTROLLABLE RECOMMENDATIONS", "No controllable features flagged.")
        return

    hmi_panel(
        "HOW THESE ARE GENERATED",
        "Each recommendation moves the top-ranked controllable feature toward the "
        "historical mean of <em>safe</em> events with similar grade transitions. "
        "Restricted to stock flow, filler flow, steam pressure, and machine speed — "
        "never quality outputs or fixed process properties.",
    )

    for i, rec in enumerate(recs):
        _render_recommendation(rec, selected_id, i)


def _render_recommendation(rec, event_id, idx):
    """Render one recommendation card with Accept/Reject buttons."""
    direction_sym = {"increase": "▲", "decrease": "▼", "maintain": "●"}.get(
        rec["direction"], "→"
    )
    dir_color = {
        "increase": COLOR_SAFE,
        "decrease": COLOR_CAUTION,
        "maintain": COLOR_PRIMARY,
    }.get(rec["direction"], COLOR_PRIMARY)

    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-card-header">
                {hmi_badge(f'#{idx+1} RECOMMENDATION', 'discovered')}
                &nbsp;
                <span style="color:{COLOR_TEXT};font-size:0.88rem;">
                    {rec['readable_name']}
                </span>
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;
                        color:{dir_color};margin:6px 0;">
                {direction_sym}&nbsp;
                {rec['current_value']:.4g}
                &nbsp;→&nbsp;
                <strong>{rec['suggested_value']:.4g}</strong>
                &nbsp;
                <span style="font-size:0.78rem;color:{COLOR_MUTED};">
                    ({rec['direction'].upper()})
                </span>
            </div>
            <div class="rec-card-rationale">
                {rec['rationale']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_accept, col_reject, col_spacer = st.columns([1, 1, 4])

    key_accept = f"accept_{event_id}_{idx}_{rec['feature']}"
    key_reject = f"reject_{event_id}_{idx}_{rec['feature']}"

    with col_accept:
        if st.button("✅ Accept", key=key_accept):
            log_feedback(
                event_id=event_id,
                suggestion_type="setpoint_recommendation",
                suggestion_detail=f"{rec['readable_name']}: {rec['current_value']:.4g} → {rec['suggested_value']:.4g}",
                source=rec["source"],
                decision="accepted",
            )
            st.success("Logged: Accepted")

    with col_reject:
        if st.button("❌ Reject", key=key_reject):
            log_feedback(
                event_id=event_id,
                suggestion_type="setpoint_recommendation",
                suggestion_detail=f"{rec['readable_name']}: {rec['current_value']:.4g} → {rec['suggested_value']:.4g}",
                source=rec["source"],
                decision="rejected",
            )
            st.error("Logged: Rejected")


def _event_label(eid, raw_df):
    row = raw_df[raw_df["event_id"] == eid].iloc[0]
    return f"Event {eid:03d} — {row['grade_from']} → {row['grade_to']}"
