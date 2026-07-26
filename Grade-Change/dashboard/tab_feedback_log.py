"""
Tab 5 — Feedback Log

Shows every Accept/Reject operator decision, reads directly from the SQLite
feedback database.  The acceptance rate is the headline KPI.

This is how Honeywell would measure and improve suggestion quality over time
in production.  The log's value is longitudinal, not just a UI nicety:
recurring rejections of the same recommendation type signal a calibration
problem in the recommendation engine.
"""

import streamlit as st
from backend.feedback import acceptance_rate, get_feedback_log
from dashboard.theme import (
    COLOR_ALARM, COLOR_BORDER, COLOR_MUTED,
    COLOR_PANEL, COLOR_PRIMARY, COLOR_SAFE, COLOR_TEXT,
    hmi_badge, hmi_panel, section_header,
)


def render():
    """Render the Feedback Log tab."""
    section_header("Operator Feedback Log — Persistent Decision Record")

    # ── headline KPIs ──────────────────────────────────────────────────
    rate = acceptance_rate()
    log_df = get_feedback_log()
    n_total = len(log_df)

    col_rate, col_total, col_accepted, col_rejected = st.columns(4)

    with col_rate:
        if rate is None:
            rate_display = "—"
            rate_color = COLOR_MUTED
        else:
            rate_display = f"{rate:.1%}"
            rate_color = COLOR_SAFE if rate >= 0.6 else COLOR_ALARM

        st.markdown(
            f"""
            <div style="background:{COLOR_PANEL};border:1px solid {COLOR_BORDER};
                        border-left:3px solid {rate_color};border-radius:8px;
                        padding:16px 20px;">
                <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
                            text-transform:uppercase;color:{COLOR_MUTED};">
                    Acceptance Rate
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;
                            color:{rate_color};margin:6px 0 4px;">
                    {rate_display}
                </div>
                <div style="font-size:0.72rem;color:{COLOR_MUTED};">
                    of all logged suggestions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_total:
        st.metric("Total Logged", str(n_total))

    with col_accepted:
        n_acc = len(log_df[log_df["decision"] == "accepted"]) if not log_df.empty else 0
        st.metric("Accepted", str(n_acc))

    with col_rejected:
        n_rej = len(log_df[log_df["decision"] == "rejected"]) if not log_df.empty else 0
        st.metric("Rejected", str(n_rej))

    # ── purpose callout ────────────────────────────────────────────────
    hmi_panel(
        "WHY THIS LOG EXISTS",
        "Every operator accept or reject is written to <span class='hmi-mono'>db/feedback.db</span> "
        "immediately — nothing lives only in session state. "
        "In production this is how you would identify which suggestion types have low acceptance rates "
        "(signal that the recommendation needs recalibration), which events the operator overrides "
        "most often (signal that the model's risk flags aren't matching real-world judgement), "
        "and whether acceptance rate improves as the recommendation engine is tuned. "
        "The log's value is <em>longitudinal</em> — a single session tells you little; "
        "a month of operations tells you everything.",
    )

    if log_df.empty:
        st.markdown(
            f"""
            <div style="background:{COLOR_PANEL};border:1px dashed {COLOR_BORDER};
                        border-radius:10px;padding:40px;text-align:center;
                        margin:20px 0;">
                <div style="font-size:2rem;margin-bottom:12px;">📋</div>
                <div style="color:{COLOR_MUTED};font-size:0.9rem;">
                    No feedback logged yet.
                    Go to Tab 4 (Live Risk), select an event, and click Accept or Reject
                    on a recommendation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── log table ──────────────────────────────────────────────────────
    section_header("Decision Log — Most Recent First")

    display_df = log_df.copy()
    display_df["decision"] = display_df["decision"].apply(
        lambda d: "✅ Accepted" if d == "accepted" else "❌ Rejected"
    )
    display_df = display_df.rename(columns={
        "id":               "#",
        "event_id":         "Event",
        "suggestion_type":  "Type",
        "suggestion_detail": "Suggestion",
        "source":           "Source",
        "decision":         "Decision",
        "timestamp":        "Timestamp (UTC)",
    })

    st.dataframe(display_df, use_container_width=True, height=400)

    # ── breakdown by suggestion type ───────────────────────────────────
    if n_total >= 2:
        section_header("Breakdown by Suggestion Type")

        type_stats = (
            log_df.groupby(["suggestion_type", "decision"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        for _, row in type_stats.iterrows():
            acc_n = row.get("accepted", 0)
            rej_n = row.get("rejected", 0)
            total_n = acc_n + rej_n
            acc_pct = acc_n / total_n if total_n else 0

            color = COLOR_SAFE if acc_pct >= 0.6 else COLOR_ALARM
            badge = hmi_badge("HIGH ACCEPTANCE", "safe") if acc_pct >= 0.6 \
                else hmi_badge("LOW ACCEPTANCE", "alarm")

            st.markdown(
                f"""
                <div class="rec-card" style="border-left-color:{color};">
                    <div class="rec-card-header">
                        {badge}
                        &nbsp;
                        <span style="color:{COLOR_TEXT};">
                            {row['suggestion_type']}
                        </span>
                    </div>
                    <div style="font-family:'IBM Plex Mono',monospace;
                                font-size:0.88rem;color:{color};margin:4px 0;">
                        {acc_n} accepted &nbsp;·&nbsp; {rej_n} rejected
                        &nbsp;·&nbsp; {acc_pct:.0%} acceptance rate
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
