"""
Risk-prediction model: RandomForest classifier for grade-change deviation.

Trains on early-window features to predict whether a transition will still
be off-spec (>2.5% deviation) at minutes 18–24.  Feature importances serve
directly as the rationale/explanation mechanism — no separate SHAP or
explainability layer needed.

Why RandomForest and not a neural net: feature importances come for free and
are directly usable as source-tagged rationale.  A more complex model would
need a separate explainability layer for the same guarantee, with no accuracy
benefit on ~140 events.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)


def train_model(features_df):
    """
    Train a RandomForest classifier on the feature table.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of ai.features.build_features().  Must contain
        'label_off_spec' and 'event_id' columns; everything else
        (except 'settle_t') is used as a feature.

    Returns
    -------
    dict with keys:
        model               fitted RandomForestClassifier
        feature_names        list of feature column names
        X_train, X_test      numpy arrays
        y_train, y_test      numpy arrays
        accuracy             float
        roc_auc              float
        confusion_matrix     2×2 numpy array
        false_negative_rate  float  (the costly error: said "safe", was off-spec)
        false_positive_rate  float
        feature_importances  dict {feature_name: importance}
    """
    # separate features from labels and metadata
    drop_cols = ["event_id", "label_off_spec", "settle_t"]
    feature_cols = [c for c in features_df.columns if c not in drop_cols]

    X = features_df[feature_cols].values
    y = features_df["label_off_spec"].values

    # 75/25 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # train the forest — 300 trees, max_depth=6 as spec'd
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # metrics
    acc = accuracy_score(y_test, y_pred)

    # roc_auc needs both classes present in test set
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # cm layout:  [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # feature importances — the built-in rationale mechanism
    importances = dict(zip(feature_cols, clf.feature_importances_))

    return {
        "model":               clf,
        "feature_names":       feature_cols,
        "X_train":             X_train,
        "X_test":              X_test,
        "y_train":             y_train,
        "y_test":              y_test,
        "accuracy":            acc,
        "roc_auc":             auc,
        "confusion_matrix":    cm,
        "false_negative_rate": fn_rate,
        "false_positive_rate": fp_rate,
        "feature_importances": importances,
    }
