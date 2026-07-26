# Grade Change Intelligence — Project Documentation

## Problem Statement

Honeywell's Quality Control System (QCS) already runs Multivariable MPC for grade changes on paper machines. Two gaps remain:

1. Grade changes still cause losses — off-spec, broke, or cull material while variables stabilize after a transition.
2. The system executes but doesn't learn or explain — no predictive warning, no plain-language guidance for operators.

---

## Architecture Overview

```
Raw CSV (minute-by-minute)
        │
        ▼
data/loader.py ──→ pd.DataFrame
        │
        ▼
ai/features.py ──→ feature table (one row per event)
        │
        ├──→ ai/model.py ──────────────→ RandomForest + metrics
        │         │
        │         └──→ ai/recommend.py ──→ setpoint suggestions
        │
        ├──→ analytics/correlations.py ──→ Known / Newly Discovered table
        ├──→ analytics/stabilization.py ──→ settle_t driver ranking
        └──→ analytics/projection.py ──→ calibrated settle-time estimate
                                              │
                                              ▼
                                    dashboard/app.py (5 tabs)
                                              │
                                              ▼
                                    backend/feedback.py → SQLite
```

---

## Data Schema

### Raw process data (data/process_data.csv)

| Column | Type | Description |
|---|---|---|
| `event_id` | int | Unique grade-change event (1–140) |
| `minute` | int | Minutes since grade-change start (0–24) |
| `grade_from` | str | Originating grade code |
| `grade_to` | str | Target grade code |
| `basis_weight` | float | Measured basis weight |
| `basis_weight_target` | float | Setpoint basis weight |
| `stock_flow` | float | Controllable |
| `filler_flow` | float | Controllable |
| `steam_pressure` | float | Controllable |
| `machine_speed` | float | Controllable |
| `ash_content` | float | Hidden correlator (planted) |
| `caliper` | float | Hidden correlator (planted) |
| `recipe_headroom` | float | Hidden correlator (planted) |

### Feature table (output of ai/features.py)

One row per event. For each process variable: early-window (min 0–5) mean and slope.
Target labels: `label_off_spec` (bool), `settle_t` (int).

---

## Module Reference

### `data/generate_data.py`
- 140 events, reproducible seed=42
- Exponential settling + overshoot + Gaussian noise
- Three planted hidden correlations via physics-motivated parameters

### `ai/features.py::build_features(df)`
- Pure function — no I/O
- Early-window features: mean and linear slope for each variable over minutes 0–5
- `label_off_spec`: True if >2.5% off target at any point in minutes 18–24
- `settle_t`: last minute at which deviation exceeds 2.5%, computed by walking backward from end

### `ai/model.py::train_model(features_df)`
- RandomForestClassifier: 300 trees, max_depth=6, balanced class weight
- 75/25 stratified split
- Returns: accuracy, ROC-AUC, confusion matrix, FNR, FPR, feature importances

### `ai/recommend.py::recommend_setpoints(...)`
- Restricted to CONTROLLABLE_FEATURES only
- Nudge top-N important features toward safe-event historical mean
- Every recommendation carries a source tag

### `analytics/correlations.py::discover_correlations(features_df)`
- Pearson r of each feature against label_off_spec
- Tags: Known (in QCS recipe) | Newly Discovered (ash_content, caliper, recipe_headroom)

### `analytics/stabilization.py::stabilization_drivers(features_df)`
- Pearson r of controllable features against settle_t only
- Separate from deviation-risk analysis

### `analytics/projection.py`
- `project_settle_time()`: linear extrapolation of early deviation trend to ±2.5% band crossing
- `fit_settle_calibration()`: fits actual ≈ m × raw + c correction
- `calibration_backtest_mae()`: reports post-correction MAE

### `backend/db.py` + `backend/feedback.py`
- SQLite at db/feedback.db
- log_feedback(), get_feedback_log(), acceptance_rate()
- WAL mode for concurrent Streamlit access

---

## Dashboard Tabs

| Tab | Content |
|---|---|
| Model Performance | FNR as headline KPI, feature importance chart, confusion matrix |
| Correlation Discovery | Known/Discovered table, bar chart, discovered-var callouts |
| Stabilization | Ranked controllable vars by settle_t correlation, histogram |
| Live Risk | Gauge, deviation trajectory, calibrated projection, Accept/Reject |
| Feedback Log | Full decision table, acceptance rate, breakdown by type |

---

## Honest Limitations

1. Trained and evaluated on synthetic data — real accuracy on Honeywell's actual process data is unknown until tested.
2. 140 events / ~35 test cases is a small test set — strong FNR is a good sign, not a statistical guarantee.
3. The settle-time projection is a calibrated linear extrapolation, not a full process model — won't capture complex non-monotonic dynamics as well as a proper dynamical model.
4. Setpoint recommendations are historical-average-based, not constrained-optimizer-based — a reasonable, explainable v1, with a real MPC-style optimizer as the natural next iteration.

---

## Engineering Decisions

**No LLM, no deep learning.** Both were considered and dropped. They add failure points (latency, cost, training instability) without adding to the requirement checklist. Every requirement is met with simpler, fully-explainable methods. This is a deliberate engineering decision, not a limitation.

**RandomForest over neural net.** Feature importances come for free and are directly usable as rationale. A more complex model needs a separate explainability layer (SHAP) for the same guarantee, with no accuracy benefit on ~140 events.

**Synthetic data.** No historical dataset was provided. `generate_data.py` produces physically-motivated data with three planted hidden correlations so discovery has something real to verify against. Swapping to real data requires changing one line (`data/loader.py`).

**Calibrated projection.** A raw linear extrapolation is systematically wrong in a predictable direction. We measure the bias, correct for it, and report the post-correction error honestly.

**Module separation.** `dashboard/*.py` may only import from `ai/`, `analytics/`, `backend/`, and `data/`. No business logic lives inside dashboard files. Every non-UI function is independently testable and swappable.
