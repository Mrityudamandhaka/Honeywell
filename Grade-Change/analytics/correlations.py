"""
Correlation discovery — statistical, deliberately NOT ML.

The brief asks for correlation discovery as a distinct capability from
prediction.  This module uses Pearson correlation to find which early-window
variables are most associated with late-stage deviation, and splits them into
"Known" (already in QCS recipe logic) vs "Newly discovered" (ash_content,
caliper, recipe_headroom — planted in the synthetic data specifically so
this analysis has something genuine to find).
"""

import numpy as np
import pandas as pd


# variables that are already part of standard MPC recipe/spec logic
KNOWN_VARS = {
    "stock_flow_early_mean", "stock_flow_early_slope",
    "filler_flow_early_mean", "filler_flow_early_slope",
    "steam_pressure_early_mean", "steam_pressure_early_slope",
    "machine_speed_early_mean", "machine_speed_early_slope",
}

# variables deliberately planted as hidden correlators
DISCOVERED_VARS = {
    "ash_content_early_mean", "ash_content_early_slope",
    "caliper_early_mean", "caliper_early_slope",
    "recipe_headroom_early_mean", "recipe_headroom_early_slope",
}


def discover_correlations(features_df):
    """
    Compute Pearson correlation of every early-window feature against the
    off-spec label, and tag each as Known or Newly Discovered.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ai.features.build_features().

    Returns
    -------
    pd.DataFrame with columns:
        feature, correlation, abs_correlation, category
    """
    # grab all the early-window feature columns
    exclude = {"event_id", "label_off_spec", "settle_t"}
    feature_cols = [c for c in features_df.columns if c not in exclude]

    rows = []
    for col in feature_cols:
        corr = features_df[col].corr(features_df["label_off_spec"])
        if np.isnan(corr):
            corr = 0.0

        if col in KNOWN_VARS:
            category = "Known"
        elif col in DISCOVERED_VARS:
            category = "Newly Discovered"
        else:
            category = "Other"

        rows.append({
            "feature":         col,
            "correlation":     round(corr, 4),
            "abs_correlation": round(abs(corr), 4),
            "category":        category,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return result
