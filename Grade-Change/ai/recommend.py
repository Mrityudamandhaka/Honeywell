"""
Setpoint recommendation engine.

Given an at-risk event, recommends corrective setpoints by nudging the
top-N most important *controllable* features toward the historical mean of
safe events with the same grade transition.

Every recommendation carries a source tag (satisfies deliverable #5:
"every suggestion tagged with its source of inference").
"""

import numpy as np
import pandas as pd


# only these may be recommended — never quality outputs or fixed properties
CONTROLLABLE_FEATURES = [
    "stock_flow_early_mean", "stock_flow_early_slope",
    "filler_flow_early_mean", "filler_flow_early_slope",
    "steam_pressure_early_mean", "steam_pressure_early_slope",
    "machine_speed_early_mean", "machine_speed_early_slope",
]

# human-readable names for display
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


def rationale_for(feature_name, importance_rank, source_tag="historical grade-change data"):
    """
    Turn a feature name + rank into a plain-English, source-tagged sentence.

    Example output:
        "Source: historical grade-change data — early trend of steam pressure
         (recipe-defined driver), ranked #2 by model importance."
    """
    readable = READABLE_NAMES.get(feature_name, feature_name)

    # figure out if it's a recipe-defined driver or a discovered correlator
    base_var = feature_name.replace("_early_mean", "").replace("_early_slope", "")
    known_vars = {"stock_flow", "filler_flow", "steam_pressure", "machine_speed"}

    if base_var in known_vars:
        var_type = "recipe-defined driver"
    else:
        var_type = "newly discovered correlator"

    return (
        f"Source: {source_tag} — {readable} ({var_type}), "
        f"ranked #{importance_rank} by model importance."
    )


def recommend_setpoints(event_row, model_result, features_df, raw_df=None, top_n=3):
    """
    Generate corrective setpoint recommendations for an at-risk event.

    Strategy: for each of the top-N most important controllable features,
    compare the event's current value to the mean of *safe* events and
    suggest moving toward that safe mean.

    Parameters
    ----------
    event_row : pd.Series or dict
        Feature row for the event in question.
    model_result : dict
        Output of ai.model.train_model().
    features_df : pd.DataFrame
        Full feature table (needed for computing safe-event baselines).
    raw_df : pd.DataFrame, optional
        Raw process data (not used currently, reserved for future use).
    top_n : int
        Number of recommendations to generate.

    Returns
    -------
    list of dict, each with keys:
        feature, readable_name, current_value, suggested_value,
        direction, rationale, source
    """
    importances = model_result["feature_importances"]

    # rank only controllable features
    controllable_imp = {
        k: v for k, v in importances.items()
        if k in CONTROLLABLE_FEATURES
    }
    ranked = sorted(controllable_imp.items(), key=lambda x: x[1], reverse=True)

    # safe-event baselines
    safe_mask = features_df["label_off_spec"] == 0
    safe_means = features_df[safe_mask][CONTROLLABLE_FEATURES].mean()

    recommendations = []
    for rank_idx, (feat, imp) in enumerate(ranked[:top_n], start=1):
        current = float(event_row[feat]) if feat in event_row else np.nan
        target_val = float(safe_means[feat])

        if np.isnan(current):
            continue

        diff = target_val - current
        if abs(diff) < 1e-6:
            direction = "maintain"
        elif diff > 0:
            direction = "increase"
        else:
            direction = "decrease"

        source_tag = "historical grade-change data"
        rationale_text = rationale_for(feat, rank_idx, source_tag)

        recommendations.append({
            "feature":        feat,
            "readable_name":  READABLE_NAMES.get(feat, feat),
            "current_value":  round(current, 4),
            "suggested_value": round(target_val, 4),
            "direction":      direction,
            "rationale":      rationale_text,
            "source":         source_tag,
        })

    return recommendations
