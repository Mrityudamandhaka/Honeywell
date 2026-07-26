"""
Stabilization-time driver analysis.

Same technique as correlations.py (Pearson correlation), but a *different
target*: correlates controllable variables against settle_t specifically.
This is a separate analysis from deviation-risk because the brief asks for
stabilization-time drivers as their own deliverable — "which loops/parameters
most affect stabilization time" is a distinct question from "will this
transition go off-spec."
"""

import numpy as np
import pandas as pd


# only controllable variables — this analysis answers "what setpoints help
# stabilize the system faster," not "what predicts risk"
CONTROLLABLE_COLS = [
    "stock_flow_early_mean", "stock_flow_early_slope",
    "filler_flow_early_mean", "filler_flow_early_slope",
    "steam_pressure_early_mean", "steam_pressure_early_slope",
    "machine_speed_early_mean", "machine_speed_early_slope",
]

READABLE_NAMES = {
    "stock_flow_early_mean":      "Stock Flow (early avg)",
    "stock_flow_early_slope":     "Stock Flow (early trend)",
    "filler_flow_early_mean":     "Filler Flow (early avg)",
    "filler_flow_early_slope":    "Filler Flow (early trend)",
    "steam_pressure_early_mean":  "Steam Pressure (early avg)",
    "steam_pressure_early_slope": "Steam Pressure (early trend)",
    "machine_speed_early_mean":   "Machine Speed (early avg)",
    "machine_speed_early_slope":  "Machine Speed (early trend)",
}


def stabilization_drivers(features_df):
    """
    Rank controllable variables by their correlation with settle_t.

    A positive correlation means higher values of that variable are associated
    with *longer* settling times (bad).  A negative correlation means higher
    values are associated with *shorter* settling (good).

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ai.features.build_features().

    Returns
    -------
    pd.DataFrame with columns:
        feature, readable_name, correlation, abs_correlation, interpretation
    """
    rows = []
    for col in CONTROLLABLE_COLS:
        if col not in features_df.columns:
            continue

        corr = features_df[col].corr(features_df["settle_t"])
        if np.isnan(corr):
            corr = 0.0

        # plain-language interpretation
        if corr > 0.05:
            interp = "Higher values → longer settling (consider reducing)"
        elif corr < -0.05:
            interp = "Higher values → faster settling (consider increasing)"
        else:
            interp = "Minimal effect on settling time"

        rows.append({
            "feature":         col,
            "readable_name":   READABLE_NAMES.get(col, col),
            "correlation":     round(corr, 4),
            "abs_correlation": round(abs(corr), 4),
            "interpretation":  interp,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return result
