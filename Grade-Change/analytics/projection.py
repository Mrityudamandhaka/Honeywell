"""
Settle-time projection with measured calibration.

A naive linear extrapolation of the early-window deviation trend systematically
*under*-estimates true settle time because real settling is asymptotic, not
linear.  Rather than presenting an inaccurate number confidently, we:

1. Fit the raw linear projection (project_settle_time)
2. Calibrate it against historical outcomes (fit_settle_calibration)
3. Report the post-correction MAE honestly (calibration_backtest_mae)

This kind of self-checking is a stronger engineering signal than a number
that merely looks precise.
"""

import numpy as np
import pandas as pd


DEVIATION_THRESHOLD = 0.025     # same 2.5% band


def project_settle_time(event_df):
    """
    Fit a line through the early-window (minutes 0–5) deviation trend and
    extrapolate to find when it crosses the ±2.5% spec band.

    Parameters
    ----------
    event_df : pd.DataFrame
        Minute-by-minute data for a single event.

    Returns
    -------
    float
        Projected minute at which deviation crosses into the spec band.
        Clamped to [0, 30] for sanity.
    """
    target = event_df["basis_weight_target"].values[0]
    early = event_df[event_df["minute"] <= 5].copy()

    if len(early) < 2:
        return 25.0  # can't extrapolate, return conservative estimate

    minutes = early["minute"].values.astype(float)
    deviations = np.abs(early["basis_weight"].values - target) / abs(target)

    # fit a line: deviation = slope * minute + intercept
    slope, intercept = np.polyfit(minutes, deviations, 1)

    if slope >= 0:
        # deviation is flat or increasing — it won't cross on its own
        return 25.0

    # solve:  slope * t + intercept = DEVIATION_THRESHOLD
    projected_t = (DEVIATION_THRESHOLD - intercept) / slope

    # clamp to reasonable range
    return float(np.clip(projected_t, 0.0, 30.0))


def fit_settle_calibration(features_df, raw_df):
    """
    Fit a linear correction: actual_settle ≈ m * raw_projection + c.

    The raw linear extrapolation is biased because exponential settling is
    asymptotic.  This calibration step measures and corrects that bias.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature table with settle_t column.
    raw_df : pd.DataFrame
        Raw minute-by-minute process data.

    Returns
    -------
    dict with keys:
        m, c            linear correction coefficients
        uncalibrated_mae  MAE before correction
    """
    projections = []
    actuals = []

    for event_id in features_df["event_id"].values:
        event_data = raw_df[raw_df["event_id"] == event_id]
        if len(event_data) == 0:
            continue

        raw_proj = project_settle_time(event_data)
        actual_t = features_df.loc[
            features_df["event_id"] == event_id, "settle_t"
        ].values[0]

        projections.append(raw_proj)
        actuals.append(actual_t)

    projections = np.array(projections)
    actuals = np.array(actuals)

    uncalibrated_mae = float(np.mean(np.abs(projections - actuals)))

    # fit linear correction
    if len(projections) >= 2 and np.std(projections) > 0:
        m, c = np.polyfit(projections, actuals, 1)
    else:
        m, c = 1.0, 0.0

    return {
        "m": float(m),
        "c": float(c),
        "uncalibrated_mae": uncalibrated_mae,
    }


def calibration_backtest_mae(features_df, raw_df, calibration):
    """
    Compute the post-calibration MAE by applying the correction to every
    event's raw projection and comparing to actual settle_t.

    Parameters
    ----------
    features_df : pd.DataFrame
    raw_df : pd.DataFrame
    calibration : dict
        Output of fit_settle_calibration().

    Returns
    -------
    float
        Mean absolute error after calibration (minutes).
    """
    m = calibration["m"]
    c = calibration["c"]

    errors = []
    for event_id in features_df["event_id"].values:
        event_data = raw_df[raw_df["event_id"] == event_id]
        if len(event_data) == 0:
            continue

        raw_proj = project_settle_time(event_data)
        calibrated = m * raw_proj + c

        actual_t = features_df.loc[
            features_df["event_id"] == event_id, "settle_t"
        ].values[0]

        errors.append(abs(calibrated - actual_t))

    return float(np.mean(errors)) if errors else 0.0
