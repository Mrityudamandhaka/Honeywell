"""
Tests for ai/features.py — build_features()

Verifies label_off_spec and settle_t are computed correctly on a hand-constructed
small synthetic event, independent of the full 140-event dataset.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.features import build_features, DEVIATION_THRESHOLD


def _make_event(event_id, bw_values, bw_target, minutes=None):
    """Build a minimal single-event DataFrame for testing."""
    n = len(bw_values)
    if minutes is None:
        minutes = list(range(n))

    return pd.DataFrame({
        "event_id":            [event_id] * n,
        "minute":              minutes,
        "grade_from":          ["A"] * n,
        "grade_to":            ["B"] * n,
        "basis_weight":        bw_values,
        "basis_weight_target": [bw_target] * n,
        "stock_flow":          [400.0] * n,
        "filler_flow":         [15.0] * n,
        "steam_pressure":      [3.0] * n,
        "machine_speed":       [1100.0] * n,
        "ash_content":         [9.0] * n,
        "caliper":             [0.10] * n,
        "recipe_headroom":     [2.5] * n,
    })


class TestLabelOffSpec:
    """Tests for the off-spec label computation."""

    def test_event_off_spec_late_window(self):
        """Event with large deviation at minutes 18-24 → label=1."""
        # minutes 0-17: within spec; minutes 18-24: way off
        bw_target = 60.0
        bw_values = (
            [60.0 * (1 + 0.10 * np.exp(-m / 3)) for m in range(18)]  # settling
            + [65.0] * 7  # back off-spec at the late window
        )
        df = _make_event(1, bw_values, bw_target)
        features = build_features(df)
        assert features.iloc[0]["label_off_spec"] == 1, (
            "Expected off-spec label when late window has >2.5% deviation"
        )

    def test_event_within_spec(self):
        """Event settled by minute 10 and stays there → label=0."""
        bw_target = 60.0
        bw_values = [bw_target + 5.0 * np.exp(-m / 2) for m in range(25)]
        df = _make_event(2, bw_values, bw_target)
        features = build_features(df)
        assert features.iloc[0]["label_off_spec"] == 0, (
            "Expected within-spec label when event settles early"
        )

    def test_event_exactly_at_threshold(self):
        """Deviation exactly at 2.5% boundary → within spec (> not >=)."""
        bw_target = 60.0
        # exactly 2.5% = 1.5g/m² above target, not *over* 2.5%
        bw_values = [bw_target + bw_target * DEVIATION_THRESHOLD] * 25
        df = _make_event(3, bw_values, bw_target)
        features = build_features(df)
        assert features.iloc[0]["label_off_spec"] == 0, (
            "Deviation exactly at threshold should NOT trigger off-spec (strict >)"
        )


class TestSettleTime:
    """Tests for the settle_t computation."""

    def test_settle_time_correct_minute(self):
        """settle_t should be the first minute after which BW stays in band."""
        bw_target = 60.0
        # exceeds spec at minutes 0-9, within spec at minutes 10-24
        bw_values = (
            [bw_target * 1.10] * 10    # 10% off — clearly exceeds
            + [bw_target * 1.01] * 15  # 1% off — within spec
        )
        df = _make_event(4, bw_values, bw_target)
        features = build_features(df)
        settle = features.iloc[0]["settle_t"]
        # last exceedance is at minute 9, so settle_t = 10
        assert settle == 10, f"Expected settle_t=10, got {settle}"

    def test_settle_time_never_settles(self):
        """Event that never enters spec → settle_t == last minute + 1."""
        bw_target = 60.0
        bw_values = [bw_target * 1.10] * 25   # always off-spec
        df = _make_event(5, bw_values, bw_target)
        features = build_features(df)
        settle = features.iloc[0]["settle_t"]
        assert settle == 25, f"Expected settle_t=25 (never settles), got {settle}"

    def test_settle_time_immediate(self):
        """Event already within spec from minute 0 → settle_t=0."""
        bw_target = 60.0
        bw_values = [bw_target * 1.01] * 25   # always within spec
        df = _make_event(6, bw_values, bw_target)
        features = build_features(df)
        settle = features.iloc[0]["settle_t"]
        assert settle == 0, f"Expected settle_t=0 (already in spec), got {settle}"


class TestFeatureColumns:
    """Tests for the shape and schema of the feature table."""

    def test_one_row_per_event(self):
        bw_target = 60.0
        bw_values = [bw_target + 5 * np.exp(-m / 3) for m in range(25)]
        df = pd.concat([
            _make_event(10, bw_values, bw_target),
            _make_event(11, bw_values, bw_target),
        ])
        features = build_features(df)
        assert len(features) == 2, "Expected one row per event_id"

    def test_early_mean_columns_exist(self):
        bw_target = 60.0
        bw_values = [bw_target + 5 * np.exp(-m / 3) for m in range(25)]
        df = _make_event(20, bw_values, bw_target)
        features = build_features(df)
        assert "stock_flow_early_mean" in features.columns
        assert "steam_pressure_early_slope" in features.columns
        assert "ash_content_early_mean" in features.columns

    def test_settle_t_is_int(self):
        bw_target = 60.0
        bw_values = [bw_target * 1.10] * 10 + [bw_target * 1.01] * 15
        df = _make_event(21, bw_values, bw_target)
        features = build_features(df)
        import pandas as pd
        # check the column dtype directly — accessing via iloc[0] on a mixed-dtype
        # row Series causes pandas to upcast int64 to float64 (known pandas behavior)
        col_dtype = features["settle_t"].dtype
        assert np.issubdtype(col_dtype, np.integer), (
            f"settle_t column should have integer dtype, got {col_dtype}"
        )


if __name__ == "__main__":
    # run with: python tests/test_features.py
    import traceback

    suites = [TestLabelOffSpec, TestSettleTime, TestFeatureColumns]
    passed = 0
    failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        for name in dir(suite_cls):
            if not name.startswith("test_"):
                continue
            try:
                getattr(suite, name)()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {suite_cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
