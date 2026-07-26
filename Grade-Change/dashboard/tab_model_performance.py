"""
Tab 1 — Model Performance

Shows how well the RandomForest risk classifier performs on held-out data.
The false-negative rate is the headline KPI here — a false negative means
the model said "safe" and the process actually went off-spec. That's the
costly direction to be wrong in a control context, and it needs to be front
and center, not buried in a confusion matrix footnote.
"""

import plotly.graph_objects as go
import streamlit as st
from dashboard.theme import (
    COLOR_ALARM, COLOR_BORDER, COLOR_CAUTION, COLOR_MUTED,
    COLOR_PANEL, COLOR_PRIMARY, COLOR_SAFE, COLOR_TEXT,
    hmi_panel, section_header,
)


def render(model_result):
    """
    Render the Model Performance tab.

    Parameters
    ----------
    model_result : dict  Output of ai.model.train_model()
    """
    acc  = model_result["accuracy"]
    auc  = model_result["roc_auc"]
    fnr  = model_result["false_negative_rate"]
    fpr  = model_result["false_positive_rate"]
    cm   = model_result["confusion_matrix"]
    imps = model_result["feature_importances"]

    n_test   = int(cm.sum())
    tn, fp, fn, tp = cm.ravel()

    # ── headline KPIs ─────────────────────────────────────────────────────
    section_header("Classifier Performance — Hold-Out Test Set")

    # FNR first and largest — it's the costly error
    col_fnr, col_acc, col_auc, col_n = st.columns(4)

    with col_fnr:
        fnr_color = COLOR_SAFE if fnr < 0.10 else (COLOR_CAUTION if fnr < 0.20 else COLOR_ALARM)
        st.markdown(
            f"""
            <div style="background:{COLOR_PANEL};border:1px solid {COLOR_BORDER};
                        border-left:3px solid {fnr_color};border-radius:8px;
                        padding:16px 20px;">
                <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
                            text-transform:uppercase;color:{COLOR_MUTED};">
                    ⚠ False-Negative Rate
                </div>
                <div style="font-family:IBM Plex Mono,monospace;font-size:2rem;
                            color:{fnr_color};margin:6px 0 4px;">
                    {fnr:.1%}
                </div>
                <div style="font-size:0.72rem;color:{COLOR_MUTED};">
                    model said safe → actually off-spec
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_acc:
        st.metric("Accuracy", f"{acc:.1%}")

    with col_auc:
        st.metric("ROC-AUC", f"{auc:.3f}" if auc == auc else "N/A")

    with col_n:
        st.metric("Test Events", str(n_test))

    # ── why FNR matters ───────────────────────────────────────────────────
    hmi_panel(
        "WHY FALSE-NEGATIVE RATE IS THE CRITICAL METRIC",
        "A false <em>positive</em> means the model raises an alarm that doesn't "
        "materialise — the operator checks and clears it, minor nuisance. "
        "A false <em>negative</em> means the model says <strong>\"safe\"</strong> "
        "while the process is actually drifting off-spec — the operator sees no "
        "warning, the transition produces broke or cull material, and the QCS "
        "never intervenes. That asymmetry is why this number leads the page.",
    )

    # ── confusion matrix ──────────────────────────────────────────────────
    section_header("Confusion Matrix")

    col_cm, col_calls = st.columns([1, 1])

    with col_cm:
        z_values = [[tn, fp], [fn, tp]]
        text_vals = [
            [f"TN: {tn}", f"FP: {fp}"],
            [f"FN: {fn}<br><span style='font-size:10px'>← costly</span>", f"TP: {tp}"],
        ]

        fig_cm = go.Figure(
            go.Heatmap(
                z=z_values,
                x=["Predicted Safe", "Predicted Off-Spec"],
                y=["Actually Safe", "Actually Off-Spec"],
                text=text_vals,
                texttemplate="%{text}",
                colorscale=[
                    [0.0, COLOR_PANEL],
                    [0.5, "rgba(45, 212, 200, 0.26)"],
                    [1.0, "rgba(45, 212, 200, 0.60)"],
                ],
                showscale=False,
            )
        )
        fig_cm.update_layout(
            paper_bgcolor=COLOR_PANEL,
            plot_bgcolor=COLOR_PANEL,
            font=dict(family="IBM Plex Mono", color=COLOR_TEXT, size=12),
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_calls:
        st.markdown(
            f"""
            <div class="hmi-panel" style="margin-top:0;">
                <div class="hmi-panel-title">Plain-English Breakdown</div>
                <p>
                  Of <span class="hmi-mono">{n_test}</span> test events:<br><br>
                  ✅ &nbsp;<span class="hmi-mono">{tp}</span> correctly flagged as off-spec<br>
                  ✅ &nbsp;<span class="hmi-mono">{tn}</span> correctly cleared as safe<br>
                  <br>
                  <span style="color:{COLOR_CAUTION};">
                  ⚠ &nbsp;<span class="hmi-mono">{fp}</span> false positives</span>
                  — alarmed but settled fine<br>
                  <span style="color:{COLOR_ALARM};">
                  ⚠ &nbsp;<span class="hmi-mono">{fn}</span> false negatives</span>
                  — missed, no alarm raised
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── feature importances ───────────────────────────────────────────────
    section_header("Feature Importances — The Model's Rationale Mechanism")

    hmi_panel(
        "HOW THIS REPLACES A SEPARATE EXPLAINABILITY LAYER",
        "Every prediction this model makes is driven by exactly these importances. "
        "There is no black box and no separate SHAP/LIME layer — the RandomForest "
        "reports its own reasoning directly. Each setpoint recommendation in Tab 4 "
        "is tagged with the importance rank shown here.",
    )

    # sort and truncate to top 12 for readability
    sorted_imps = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:12]
    feat_names  = [_readable(k) for k, _ in sorted_imps]
    feat_vals   = [v for _, v in sorted_imps]
    bar_colors  = [COLOR_PRIMARY if v > 0.07 else COLOR_MUTED for v in feat_vals]

    fig_imp = go.Figure(
        go.Bar(
            y=feat_names,
            x=feat_vals,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in feat_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color=COLOR_MUTED),
        )
    )
    fig_imp.update_layout(
        paper_bgcolor=COLOR_PANEL,
        plot_bgcolor=COLOR_PANEL,
        font=dict(family="IBM Plex Sans", color=COLOR_TEXT, size=11),
        height=380,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(
            title="Importance",
            gridcolor=COLOR_BORDER,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(autorange="reversed", gridcolor=COLOR_BORDER),
        bargap=0.3,
    )
    st.plotly_chart(fig_imp, use_container_width=True)


def _readable(feature_name):
    """Short human-readable feature label for the importance chart."""
    return (
        feature_name
        .replace("_early_mean",  " (avg)")
        .replace("_early_slope", " (slope)")
        .replace("_", " ")
        .title()
    )
