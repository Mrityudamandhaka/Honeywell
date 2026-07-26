"""
Tests for ai/model.py — train_model()

Verifies that train_model runs end-to-end on the full generated dataset and
returns all required metric keys with sensible values.  Confusion matrix
values must sum to the test set size.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_dataset
from data.loader import load_events
from ai.features import build_features
from ai.model import train_model


def _get_model_result():
    """Load data, build features, train model — shared fixture."""
    DATA_PATH = "data/process_data.csv"
    if not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH)
    raw_df = load_events(DATA_PATH)
    features_df = build_features(raw_df)
    return train_model(features_df), features_df


class TestModelResult:
    """Tests for the structure of train_model() output."""

    def setup_method(self):
        self.result, self.features_df = _get_model_result()

    def test_required_keys_present(self):
        required = [
            "model", "feature_names", "X_train", "X_test",
            "y_train", "y_test", "accuracy", "roc_auc",
            "confusion_matrix", "false_negative_rate",
            "false_positive_rate", "feature_importances",
        ]
        for key in required:
            assert key in self.result, f"Missing key: {key}"

    def test_accuracy_in_range(self):
        acc = self.result["accuracy"]
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_roc_auc_in_range_or_nan(self):
        auc = self.result["roc_auc"]
        assert np.isnan(auc) or (0.0 <= auc <= 1.0), f"ROC-AUC invalid: {auc}"

    def test_false_negative_rate_in_range(self):
        fnr = self.result["false_negative_rate"]
        assert 0.0 <= fnr <= 1.0, f"FNR out of range: {fnr}"

    def test_false_positive_rate_in_range(self):
        fpr = self.result["false_positive_rate"]
        assert 0.0 <= fpr <= 1.0, f"FPR out of range: {fpr}"

    def test_confusion_matrix_sums_to_test_set_size(self):
        cm = self.result["confusion_matrix"]
        y_test = self.result["y_test"]
        assert cm.sum() == len(y_test), (
            f"CM sum {cm.sum()} != test set size {len(y_test)}"
        )

    def test_confusion_matrix_shape(self):
        cm = self.result["confusion_matrix"]
        assert cm.shape == (2, 2), f"Expected 2×2 confusion matrix, got {cm.shape}"

    def test_feature_importances_sum_to_one(self):
        imps = self.result["feature_importances"]
        total = sum(imps.values())
        assert abs(total - 1.0) < 0.01, (
            f"Feature importances should sum to ~1.0, got {total:.4f}"
        )

    def test_feature_importances_all_nonnegative(self):
        imps = self.result["feature_importances"]
        for k, v in imps.items():
            assert v >= 0.0, f"Negative importance for {k}: {v}"

    def test_model_can_predict(self):
        """Model must be callable and return probabilities in [0,1]."""
        clf = self.result["model"]
        X_test = self.result["X_test"]
        probs = clf.predict_proba(X_test)
        assert probs.shape[1] == 2
        assert np.all((probs >= 0) & (probs <= 1))

    def test_25_percent_test_split(self):
        """With 140 events and 25% split, test set should be ~35 events."""
        n_test = len(self.result["y_test"])
        # allow ±5 for stratification rounding
        assert 30 <= n_test <= 40, f"Unexpected test set size: {n_test}"


if __name__ == "__main__":
    import traceback

    suite = TestModelResult()
    suite.setup_method()

    passed = 0
    failed = 0
    for name in dir(TestModelResult):
        if not name.startswith("test_"):
            continue
        try:
            getattr(suite, name)()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
