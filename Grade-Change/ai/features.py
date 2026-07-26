"""
Feature engineering for grade-change risk prediction.

Transforms the raw minute-by-minute process data into one row per event,
using only the early window (minutes 0–5) — because the whole point is to
predict deviation *before* the transition finishes (minutes 18–24).

Pure function: no I/O, no Streamlit dependency. Fully unit-testable.
"""

import numpy as np
import pandas as pd


# the 2.5% deviation threshold from the spec — this is a business rule,
# not a model parameter
DEVIATION_THRESHOLD = 0.025

# process variables we extract early-window features from
PROCESS_VARS = [
    "stock_flow", "filler_flow", "steam_pressure", "machine_speed",
    "ash_content", "caliper", "recipe_headroom",
]


def _early_window_features(event_df):
    """
    Compute mean and slope for each process variable over minutes 0–5.

    The slope is a simple linear regression coefficient (change per minute),
    which captures whether the variable is moving in the right direction
    early on.
    """
    early = event_df[event_df["minute"] <= 5].copy()
    if len(early) < 2:
        # edge case: not enough data points for a slope
        early = event_df.head(2)

    features = {}
    for var in PROCESS_VARS:
        values = early[var].values
        features[f"{var}_early_mean"] = np.mean(values)

        # slope via polyfit degree-1
        minutes = early["minute"].values.astype(float)
        if len(minutes) >= 2 and np.std(minutes) > 0:
            slope = np.polyfit(minutes, values, 1)[0]
        else:
            slope = 0.0
        features[f"{var}_early_slope"] = slope

    return features


def _compute_settle_time(event_df):
    """
    Compute settle_t: the first minute after which basis_weight NEVER again
    exceeds the ±2.5% band around target.

    Method: walk backward from the last minute. The settle minute is the
    last point at which the deviation exceeds the threshold — everything
    after that is within spec permanently.
    """
    bw = event_df["basis_weight"].values
    target = event_df["basis_weight_target"].values[0]

    exceeds = np.abs(bw - target) / abs(target) > DEVIATION_THRESHOLD

    # walk from the end — find the last minute that exceeds
    settle_t = 0
    for i in range(len(exceeds) - 1, -1, -1):
        if exceeds[i]:
            settle_t = event_df["minute"].values[i] + 1
            break

    return int(settle_t)


def _compute_off_spec_label(event_df):
    """
    label_off_spec: True if the event is still >2.5% off target at any point
    during minutes 18–24 (the late window).
    """
    target = event_df["basis_weight_target"].values[0]
    late = event_df[event_df["minute"] >= 18]

    if len(late) == 0:
        return False

    late_bw = late["basis_weight"].values
    deviation_pct = np.abs(late_bw - target) / abs(target)
    return bool(np.any(deviation_pct > DEVIATION_THRESHOLD))


def build_features(df):
    """
    Build the per-event feature table from raw minute-by-minute data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw process data with columns matching data/generate_data.py schema.

    Returns
    -------
    pd.DataFrame
        One row per event_id, with early-window means/slopes for each
        process variable, plus label_off_spec (bool) and settle_t (int).
    """
    records = []
    for event_id, event_df in df.groupby("event_id"):
        event_df = event_df.sort_values("minute")

        row = {"event_id": event_id}
        row.update(_early_window_features(event_df))
        row["label_off_spec"] = _compute_off_spec_label(event_df)
        row["settle_t"] = _compute_settle_time(event_df)

        records.append(row)

    features_df = pd.DataFrame(records)
    features_df["label_off_spec"] = features_df["label_off_spec"].astype(int)
    features_df["settle_t"] = features_df["settle_t"].astype(int)
    return features_df
