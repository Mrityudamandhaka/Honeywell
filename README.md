# Honeywell
# Grade Change Intelligence — Paper Making Process

## Problem
During a grade change, Basis Weight (the main quality variable) drifts before settling to
its new target. The existing QCS/MD-MPC control system executes the transition well but
does not learn from history or explain itself to operators. This system adds an
intelligence layer on top: it predicts deviation risk early in a transition, recommends
corrective setpoints, and explains every suggestion with a source.

## Architecture / building blocks

```
data/process_data.csv              (per-minute process data across 140 grade-change events)
        |
        v
core.build_features()              collapses each event into:
                                      - early-window (t=0-5 min) feature averages + trends
                                      - label: still off-spec (>2.5% deviation) at t=18-24 min
        |
        +----------------------------------------------+
        v                                               v
core.discover_correlations()               core.train_model()
  correlates early features vs.              RandomForestClassifier predicts
  late-stage deviation; tags each              off-spec risk from early-window
  as "known" (already in QCS logic)            features; feature_importances_
  vs "newly discovered"                        double as the rationale source
        |                                               |
        +----------------------------------------------+
                            v
                core.recommend_setpoints()
                  for a high-risk event, nudges the top
                  CONTROLLABLE features (stock flow, filler
                  flow, steam pressure, machine speed) toward
                  the historical average of SAFE events of the
                  same grade transition
                            |
                            v
                        app.py (Streamlit)
        Tab 1: model performance + feature importance
        Tab 2: correlation discovery (known vs newly discovered)
        Tab 3: pick an event -> risk %, deviation trajectory chart,
               recommendations with Accept/Reject buttons
        Tab 4: feedback log (SQLite) + acceptance rate
```

## Why these design choices
- **Prediction is made from the first 5 minutes of a transition**, so it's genuinely
  "before quality limits are exceeded," not hindsight.
- **Rationale is not a text generator** — it's a direct readout of which historical
  variable/trend the model actually weighted, tagged with its source (historical data)
  and whether it's a known recipe-defined driver or a newly discovered one. This keeps
  every claim traceable and auditable, which matters more in a process-control context
  than a fluent-sounding explanation would.
- **Recommendations are restricted to controllable process inputs** (stock flow, filler
  flow, steam pressure, machine speed) — never quality outputs (ash, caliper, moisture)
  or fixed characteristics (which grades are involved), since an operator can't act on
  those in real time.
- **Accept/Reject logging** lets the system's suggestion quality be evaluated over time,
  as required — every response is timestamped and stored in SQLite.

## Data note
Honeywell's problem statement did not include a real historical dataset in what was
provided at preview time, so `generate_data.py` produces a physically-motivated synthetic
dataset (exponential settling curves with grade-appropriate overshoot, plus deliberately
"hidden" correlations — ash, caliper, and recipe headroom — that are not part of the
standard control logic, so the discovery step has something genuine to find). If a real
dataset is supplied, `core.load_events()` just needs the CSV path swapped — the rest of
the pipeline is unchanged.

## How to run
```
pip install streamlit scikit-learn pandas plotly
streamlit run app.py
```

## Model performance (on synthetic data)
- Accuracy: ~91%
- ROC-AUC: ~0.97
- 140 grade-change events, 25/75 train/test split, stratified
