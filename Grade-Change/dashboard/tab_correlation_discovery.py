"""
Tab 2 — Correlation Discovery

Pearson correlation of early-window variables against late-stage deviation,
split into Known (already in QCS recipe logic) vs Newly Discovered (ash
content, caliper, recipe headroom — variables not in any standard MPC recipe
and deliberately planted so this analysis has something real to find).

This is a statistical module, not ML.  The split is intentional: the brief
asks for correlation discovery as a distinct capability from prediction.
"""

import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import (
    COLOR_ALARM, COLOR_BORDER, COLOR_CAUTION, COLOR_MUTED,
    COLOR_PANEL, COLOR_PRIMARY, COLOR_SAFE, COLOR_TEXT,
    hmi_badge, hmi_panel, section_header,
)


def render(corr_df):
    """
    Render the Correlation Discovery tab.

    Parameters
    ----------
    corr_df : pd.DataFrame  Output of analytics.correlations.discover_correlations()
    """
    section_header("Variable Correlation with Late-Stage Deviation (Minutes 18–24)")

    hmi_panel(
        "WHAT THIS ANALYSIS DOES",
        "Pearson correlation between each early-window process variable and the "
        "off-spec label — computed <em>separately from the risk model</em>. "
        "Known variables are already tracked by the QCS recipe. Newly Discovered "
        "variables are not in any standard recipe or MPC spec sheet — they were "
        "found purely from historical patterns.",
    )

    # ── full correlation table ────────────────────────────────────────────
    section_header("Full Correlation Table")

    display_df = corr_df.copy()
    display_df["feature"] = display_df["feature"].apply(_readable)
    display_df["Strength"] = display_df["abs_correlation"].apply(_strength_label)

    # color-code by category
    def row_style(row):
        if row["category"] == "Newly Discovered":
            return [f"color: {COLOR_CAUTION}"] * len(row)
        elif row["category"] == "Known":
            return [f"color: {COLOR_PRIMARY}"] * len(row)
        return [""] * len(row)

    styled = (
        display_df[["feature", "correlation", "abs_correlation", "category", "Strength"]]
        .rename(columns={
            "feature": "Variable",
            "correlation": "Pearson r",
            "abs_correlation": "|r|",
            "category": "Category",
        })
        .style.apply(row_style, axis=1)
        .format({"Pearson r": "{:.4f}", "|r|": "{:.4f}"})
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", COLOR_PANEL),
                ("color", COLOR_MUTED),
                ("font-size", "0.72rem"),
                ("letter-spacing", "0.08em"),
                ("text-transform", "uppercase"),
            ]},
            {"selector": "tbody td", "props": [
                ("font-family", "IBM Plex Mono, monospace"),
                ("font-size", "0.82rem"),
            ]},
        ])
    )
    st.dataframe(
        display_df[["feature", "correlation", "abs_correlation", "category", "Strength"]]
        .rename(columns={
            "feature": "Variable",
            "correlation": "Pearson r",
            "abs_correlation": "|r|",
            "category": "Status",
        }),
        use_container_width=True,
        height=320,
    )

    # ── visual correlation chart ─────────────────────────────────────────
    section_header("Correlation Strength — Known vs Newly Discovered")

    known_df = corr_df[corr_df["category"] == "Known"].copy()
    new_df   = corr_df[corr_df["category"] == "Newly Discovered"].copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Known (in QCS recipe)",
        x=known_df["feature"].apply(_readable),
        y=known_df["correlation"],
        marker=dict(color=COLOR_PRIMARY, opacity=0.85),
    ))

    fig.add_trace(go.Bar(
        name="Newly Discovered",
        x=new_df["feature"].apply(_readable),
        y=new_df["correlation"],
        marker=dict(color=COLOR_CAUTION, opacity=0.85),
    ))

    fig.update_layout(
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="IBM Plex Sans", color=COLOR_TEXT, size=11),
        height=380,
        margin=dict(l=10, r=10, t=10, b=120),
        xaxis=dict(tickangle=-35, gridcolor=COLOR_BORDER),
        yaxis=dict(title="Pearson r", gridcolor=COLOR_BORDER, zeroline=True,
                   zerolinecolor=COLOR_BORDER),
        legend=dict(bgcolor=COLOR_PANEL, bordercolor=COLOR_BORDER, borderwidth=1),
        barmode="group",
        bargap=0.2,
    )
    fig.add_hline(y=0, line_color=COLOR_BORDER, line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── newly discovered callout ─────────────────────────────────────────
    section_header("Newly Discovered Correlations — Not in Standard MPC Recipe")

    discovered = corr_df[corr_df["category"] == "Newly Discovered"].copy()

    if discovered.empty:
        st.info("No newly discovered correlations found in this dataset.")
        return

    for _, row in discovered.iterrows():
        r   = row["correlation"]
        abs_r = row["abs_correlation"]
        feat  = _readable(row["feature"])
        strength = _strength_label(abs_r)

        direction = "positively" if r > 0 else "negatively"
        badge_html = hmi_badge("NEWLY DISCOVERED", "discovered")
        known_badge = hmi_badge("NOT IN RECIPE", "alarm")

        st.markdown(
            f"""
            <div class="rec-card" style="border-left-color:{COLOR_CAUTION};">
                <div class="rec-card-header">
                    {badge_html} &nbsp;{known_badge} &nbsp;
                    <span style="color:{COLOR_TEXT};font-size:0.85rem;">{feat}</span>
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;
                            color:{COLOR_CAUTION};margin:6px 0;">
                    r = {r:+.4f} &nbsp;·&nbsp; |r| = {abs_r:.4f} &nbsp;·&nbsp; {strength}
                </div>
                <div class="rec-card-rationale">
                    Correlated {direction} with late-stage deviation.
                    This variable is <strong>not present</strong> in any standard MPC
                    recipe or QCS spec sheet — it was found purely from historical
                    transition data.  If this relationship is causal, operators could
                    gain a new lever for managing transition quality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _readable(feature_name):
    return (
        feature_name
        .replace("_early_mean",  " (avg)")
        .replace("_early_slope", " (slope)")
        .replace("_", " ")
        .title()
    )


def _strength_label(abs_r):
    if abs_r >= 0.5:
        return "Strong"
    elif abs_r >= 0.3:
        return "Moderate"
    elif abs_r >= 0.1:
        return "Weak"
    return "Negligible"
