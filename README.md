# Grade Change Intelligence

**Honeywell Hackathon — Paper Machine QCS Add-On**

Predictive deviation risk, correlation discovery, stabilization analysis, and operator feedback loop for grade changes on paper machines.

---

## What It Does

Honeywell's QCS already runs Multivariable MPC for grade changes. This system adds:

| Capability | How |
|---|---|
| **Predict** off-spec risk before minute 18 | RandomForest on early-window (min 0–5) features |
| **Explain** every prediction | Feature importances — no black box |
| **Recommend** corrective setpoints | Nudge toward safe-event historical means |
| **Discover** new correlations | Pearson correlation vs. late-stage deviation |
| **Rank** stabilization drivers | Correlation vs. settle_t, controllable vars only |
| **Project** settle time (calibrated) | Linear extrapolation + bias correction + measured MAE |
| **Log** operator Accept/Reject | SQLite — persists across sessions |

---

## Quick Start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. generate the synthetic dataset (once)
python -m data.generate_data

# 3. run the dashboard
streamlit run dashboard/app.py
```

To use real Honeywell data: change only the default `path` in `data/loader.py::load_events()`.

---

## Folder Structure

```
gradechange-intelligence/
├── ai/            # ML: RandomForest risk prediction + recommendations
├── analytics/     # Statistics: correlations, stabilization, projection
├── data/          # Synthetic data generation + loading
├── backend/       # SQLite feedback persistence
├── dashboard/     # Streamlit UI — one file per tab
├── tests/         # Unit tests (no Streamlit dependency)
└── docs/          # This file + full project documentation
```

---

## Running Tests

```bash
# run all three test files
python tests/test_features.py
python tests/test_model.py
python tests/test_correlations.py
```

---

## Design Decisions

**Why RandomForest and not a neural net?**
Feature importances come for free and are directly usable as source-tagged rationale — a hard requirement. A more complex model needs a separate SHAP/LIME layer for the same guarantee, with no accuracy benefit on ~140 events.

**Why no LLM?**
Adds failure points (latency, cost) without adding to the requirement checklist. Every requirement is met with simpler, fully-explainable methods. This is a deliberate engineering decision, not a limitation.

**Why synthetic data?**
No historical dataset was provided at build time. The generator produces physically-motivated data with three planted hidden correlations. Swapping to real data requires changing one line in `data/loader.py`.

**Why calibrate the settle-time projection?**
A raw linear extrapolation is systematically wrong in a predictable, measurable direction. We measure the bias, correct for it, and report the post-correction error honestly.
