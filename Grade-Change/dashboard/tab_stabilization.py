"""
Tab 3 — Stabilization Analysis

Ranks controllable process variables by their correlation with settle_t —
the minute at which the transition permanently comes within the ±2.5% spec
band.  This is a separate analysis from deviation-risk (Tab 1) because the
question here is "what helps the machine settle *faster*", not "will this
transition go off-spec at all."

Directly actionable: each variable comes with a plain-English suggestion on
whether operators should increase or decrease it to shorten settling time.
"""

import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import (
    COLOR_ALARM, COLOR_BORDER, COLOR_CAUTION, COLOR_MUTED,
    COLOR_PANEL, COLOR_PRIMARY, COLOR_SAFE, COLOR_TEXT,
    hmi_badge, hmi_panel, section_header,
)


def render(stab_df, features_df):
    """
    Render the Stabilization tab.

    Parameters
    ----------
    stab_df : pd.DataFrame  Output of analytics.stabilization.stabilization_drivers()
    features_df : pd.DataFrame  Full feature table (for distribution stats)
    """
    section_header("What Drives Stabilization Time?")

    avg_settle = features_df["settle_t"].mean()
    min_settle = features_df["settle_t"].min()
    max_settle = features_df["settle_t"].max()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Settle Time", f"{avg_settle:.1f} min")
    col2.metric("Fastest Event",   f"{min_settle} min")
    col3.metric("Slowest Event",   f"{max_settle} min")

    hmi_panel(
        "HOW TO READ THIS ANALYSIS",
        "Correlation of controllable variables against <em>settle_t</em> (the minute "
        "the transition permanently enters the ±2.5% spec band). "
        "A <strong>positive</strong> correlation means higher values extend settling — "
        "consider reducing. A <strong>negative</strong> correlation means higher values "
        "shorten settling — consider increasing during transition. "
        "This is Pearson correlation, not model prediction — it's a direct statistical "
        "statement from historical data, not an inference.",
    )

    # ── ranked bar chart ─────────────────────────────────────────────────
    section_header("Controllable Variables Ranked by Impact on Settling Time")

    sorted_df = stab_df.sort_values("abs_correlation", ascending=True)

    bar_colors = []
    for r in sorted_df["correlation"]:
        if abs(r) < 0.05:
            bar_colors.append(COLOR_MUTED)
        elif r > 0:
            bar_colors.append(COLOR_ALARM)     # longer settling → alarm
        else:
            bar_colors.append(COLOR_SAFE)      # shorter settling → good

    fig = go.Figure(
        go.Bar(
            y=sorted_df["readable_name"],
            x=sorted_df["correlation"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"r = {v:+.4f}" for v in sorted_df["correlation"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color=COLOR_MUTED),
        )
    )
    fig.add_vline(x=0, line_color=COLOR_BORDER, line_width=1)
    fig.update_layout(
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="IBM Plex Sans", color=COLOR_TEXT, size=11),
        height=360,
        margin=dict(l=10, r=100, t=10, b=10),
        xaxis=dict(title="Correlation with Settle Time (r)", gridcolor=COLOR_BORDER,
                   zeroline=False),
        yaxis=dict(gridcolor=COLOR_BORDER),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── actionable ranked list ────────────────────────────────────────────
    section_header("Ranked Action List — Fastest Path to Stabilization")

    for rank, (_, row) in enumerate(
        stab_df.sort_values("abs_correlation", ascending=False).iterrows(), start=1
    ):
        r    = row["correlation"]
        feat = row["readable_name"]
        interp = row["interpretation"]

        if abs(r) < 0.05:
            badge = hmi_badge(f"#{rank} MINIMAL EFFECT", "known")
            card_color = COLOR_MUTED
        elif r > 0:
            badge = hmi_badge(f"#{rank} REDUCE TO SPEED UP", "alarm")
            card_color = COLOR_ALARM
        else:
            badge = hmi_badge(f"#{rank} INCREASE TO SPEED UP", "safe")
            card_color = COLOR_SAFE

        # look up the population stats for context
        col_name = row["feature"]
        if col_name in stab_df.get("feature", {}).values if hasattr(stab_df.get("feature", {}), "values") else False:
            pass

        st.markdown(
            f"""
            <div class="rec-card" style="border-left-color:{card_color};">
                <div class="rec-card-header">
                    {badge} &nbsp;
                    <span style="color:{COLOR_TEXT};font-size:0.85rem;">{feat}</span>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;
                            font-size:0.88rem;color:{card_color};margin:4px 0;">
                    r = {r:+.4f} &nbsp;·&nbsp; |r| = {abs(r):.4f}
                </div>
                <div class="rec-card-rationale">
                    {interp} — Source: historical settle-time correlation analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── settle time distribution ──────────────────────────────────────────
    section_header("Settle Time Distribution Across All Events")

    fig_hist = go.Figure(
        go.Histogram(
            x=features_df["settle_t"],
            nbinsx=20,
            marker=dict(
                color=COLOR_PRIMARY,
                opacity=0.75,
                line=dict(color=COLOR_BORDER, width=1),
            ),
            name="Events",
        )
    )
    fig_hist.add_vline(
        x=avg_settle,
        line_color=COLOR_CAUTION,
        line_dash="dash",
        annotation_text=f"  avg: {avg_settle:.1f} min",
        annotation_font=dict(color=COLOR_CAUTION, size=11),
    )
    fig_hist.update_layout(
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="IBM Plex Sans", color=COLOR_TEXT, size=11),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Settle Time (minutes)", gridcolor=COLOR_BORDER),
        yaxis=dict(title="Event Count", gridcolor=COLOR_BORDER),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
