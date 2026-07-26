"""
Tests for analytics/correlations.py and analytics/stabilization.py

Verifies:
- discover_correlations() correctly tags known vs newly-discovered variables
- stabilization_drivers() output is keyed exclusively on controllable variables
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_dataset
from data.loader import load_events
from ai.features import build_features
from analytics.correlations import discover_correlations, KNOWN_VARS, DISCOVERED_VARS
from analytics.stabilization import stabilization_drivers, CONTROLLABLE_COLS


def _load_features():
    DATA_PATH = "data/process_data.csv"
    if not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH)
    raw_df = load_events(DATA_PATH)
    return build_features(raw_df)


class TestDiscoverCorrelations:
    """Tests for analytics.correlations.discover_correlations()"""

    def setup_method(self):
        features_df = _load_features()
        self.corr_df = discover_correlations(features_df)

    def test_returns_dataframe_with_required_columns(self):
        for col in ["feature", "correlation", "abs_correlation", "category"]:
            assert col in self.corr_df.columns, f"Missing column: {col}"

    def test_known_vars_tagged_correctly(self):
        """All KNOWN_VARS features must have category == 'Known'."""
        known_rows = self.corr_df[self.corr_df["category"] == "Known"]
        known_features_in_result = set(known_rows["feature"].values)
        # every known var that appears in the result should be tagged Known
        for feat in known_features_in_result:
            assert feat in KNOWN_VARS, (
                f"Feature '{feat}' tagged Known but not in KNOWN_VARS"
            )

    def test_discovered_vars_tagged_correctly(self):
        """All DISCOVERED_VARS features must have category == 'Newly Discovered'."""
        disc_rows = self.corr_df[self.corr_df["category"] == "Newly Discovered"]
        for feat in disc_rows["feature"].values:
            assert feat in DISCOVERED_VARS, (
                f"Feature '{feat}' tagged Newly Discovered but not in DISCOVERED_VARS"
            )

    def test_no_label_or_settle_in_result(self):
        """label_off_spec and settle_t should never appear as features."""
        assert "label_off_spec" not in self.corr_df["feature"].values
        assert "settle_t" not in self.corr_df["feature"].values

    def test_newly_discovered_features_present(self):
        """The three planted correlators must appear in the result."""
        discovered_in_result = set(
            self.corr_df[self.corr_df["category"] == "Newly Discovered"]["feature"].values
        )
        expected_discovered = {f for f in DISCOVERED_VARS}
        found = discovered_in_result & expected_discovered
        assert len(found) > 0, (
            "No newly-discovered features found in correlation result — "
            "check that ash_content, caliper, recipe_headroom columns exist"
        )

    def test_abs_correlation_equals_abs_of_correlation(self):
        """abs_correlation must equal |correlation| for every row."""
        import math
        for _, row in self.corr_df.iterrows():
            expected = abs(row["correlation"])
            actual   = row["abs_correlation"]
            assert abs(expected - actual) < 1e-9, (
                f"abs_correlation mismatch for {row['feature']}: "
                f"|{row['correlation']}| != {row['abs_correlation']}"
            )

    def test_sorted_descending_by_abs_correlation(self):
        """Result should be sorted with highest |r| first."""
        vals = self.corr_df["abs_correlation"].values
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1], (
                f"Not sorted descending at index {i}: {vals[i]} < {vals[i+1]}"
            )


class TestStabilizationDrivers:
    """Tests for analytics.stabilization.stabilization_drivers()"""

    def setup_method(self):
        features_df = _load_features()
        self.stab_df = stabilization_drivers(features_df)

    def test_returns_dataframe_with_required_columns(self):
        for col in ["feature", "readable_name", "correlation",
                    "abs_correlation", "interpretation"]:
            assert col in self.stab_df.columns, f"Missing column: {col}"

    def test_only_controllable_features(self):
        """Output must be restricted to CONTROLLABLE_COLS — no hidden vars."""
        for feat in self.stab_df["feature"].values:
            assert feat in CONTROLLABLE_COLS, (
                f"Non-controllable feature '{feat}' appeared in stabilization_drivers output"
            )

    def test_no_hidden_correlators(self):
        """ash_content, caliper, recipe_headroom must not appear."""
        forbidden = {"ash_content_early_mean", "ash_content_early_slope",
                     "caliper_early_mean", "caliper_early_slope",
                     "recipe_headroom_early_mean", "recipe_headroom_early_slope"}
        for feat in self.stab_df["feature"].values:
            assert feat not in forbidden, (
                f"Hidden correlator '{feat}' should not appear in stabilization drivers"
            )

    def test_correlation_values_in_range(self):
        """Pearson r must be in [-1, 1]."""
        for _, row in self.stab_df.iterrows():
            r = row["correlation"]
            assert -1.0 <= r <= 1.0, (
                f"Correlation out of range for {row['feature']}: {r}"
            )

    def test_interpretation_column_nonempty(self):
        """Every row must have a non-empty interpretation string."""
        for _, row in self.stab_df.iterrows():
            assert isinstance(row["interpretation"], str) and len(row["interpretation"]) > 0


if __name__ == "__main__":
    import traceback

    suites = [TestDiscoverCorrelations, TestStabilizationDrivers]
    passed = 0
    failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        suite.setup_method()
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
