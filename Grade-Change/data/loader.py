"""
Single point of entry for loading the raw process dataset.

Swapping to a real Honeywell dataset means changing only the default `path`
argument here — every downstream function depends on column schema, never
on the data source itself.
"""

import pandas as pd


def load_events(path="data/process_data.csv"):
    """
    Load the minute-by-minute grade-change event data.

    Parameters
    ----------
    path : str
        Path to the CSV file matching the schema defined in
        data/generate_data.py (event_id, minute, grade_from, grade_to,
        basis_weight, basis_weight_target, stock_flow, filler_flow,
        steam_pressure, machine_speed, ash_content, caliper,
        recipe_headroom).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    return df
