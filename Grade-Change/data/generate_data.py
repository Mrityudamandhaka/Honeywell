"""
Synthetic grade-change event generator for the Grade Change Intelligence system.

Produces data/process_data.csv with 140 grade-change events, minute-by-minute.
Each event simulates an exponential settling dynamic with realistic overshoot,
noise, and three deliberately planted "hidden" correlations that aren't part of
standard MPC recipe logic (ash_content, caliper, recipe_headroom).

Column schema (one row per event-minute):
    event_id            int     Unique grade-change event (1..140)
    minute              int     Minutes since grade-change start (0..~24)
    grade_from          str     Originating grade code
    grade_to            str     Target grade code
    basis_weight        float   Measured basis weight (g/m²)
    basis_weight_target float   Setpoint for the new grade
    stock_flow          float   Controllable — stock flow rate
    filler_flow         float   Controllable — filler flow rate
    steam_pressure      float   Controllable — dryer steam pressure
    machine_speed       float   Controllable — machine speed
    ash_content         float   Hidden correlator — not in standard recipe
    caliper             float   Hidden correlator — not in standard recipe
    recipe_headroom     float   Hidden correlator — not in standard recipe

Usage:
    python -m data.generate_data
"""

import os
import numpy as np
import pandas as pd


# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.RandomState(SEED)

NUM_EVENTS = 140
MINUTES_PER_EVENT = 25          # 0 through 24

# grade catalogue — loosely modelled on real paper-machine grades
GRADES = {
    "LWC-58":  {"bw_target": 58.0,  "stock": 420, "filler": 18, "steam": 3.2, "speed": 1150},
    "LWC-65":  {"bw_target": 65.0,  "stock": 460, "filler": 20, "steam": 3.5, "speed": 1120},
    "SC-52":   {"bw_target": 52.0,  "stock": 380, "filler": 15, "steam": 2.9, "speed": 1200},
    "SC-60":   {"bw_target": 60.0,  "stock": 430, "filler": 17, "steam": 3.3, "speed": 1170},
    "NEWS-45": {"bw_target": 45.0,  "stock": 340, "filler": 12, "steam": 2.6, "speed": 1250},
    "NEWS-48": {"bw_target": 48.5,  "stock": 360, "filler": 13, "steam": 2.7, "speed": 1230},
    "FINE-80": {"bw_target": 80.0,  "stock": 520, "filler": 24, "steam": 4.0, "speed": 1020},
    "FINE-90": {"bw_target": 90.0,  "stock": 570, "filler": 27, "steam": 4.3, "speed":  980},
}

GRADE_NAMES = list(GRADES.keys())


def _pick_transition():
    """Pick a from→to grade pair (never the same grade)."""
    g_from = GRADE_NAMES[RNG.randint(len(GRADE_NAMES))]
    g_to = g_from
    while g_to == g_from:
        g_to = GRADE_NAMES[RNG.randint(len(GRADE_NAMES))]
    return g_from, g_to


def _exponential_settle(target, start, tau, minutes, noise_std):
    """
    Simulate exponential approach from `start` to `target` with time constant
    `tau` and additive Gaussian noise.  Returns array of length `minutes`.
    """
    t = np.arange(minutes, dtype=float)
    trajectory = target + (start - target) * np.exp(-t / tau)
    trajectory += RNG.normal(0, noise_std, size=minutes)
    return trajectory


def _generate_single_event(event_id):
    """Build a DataFrame of minute-by-minute rows for one grade-change event."""
    g_from, g_to = _pick_transition()
    spec_from = GRADES[g_from]
    spec_to   = GRADES[g_to]

    bw_target = spec_to["bw_target"]
    bw_start  = spec_from["bw_target"] + RNG.normal(0, 1.5)

    # settling dynamics — tau controls how quickly things converge
    base_tau = RNG.uniform(3.0, 8.0)

    # ── hidden correlators (planted relationships) ───────────────────────
    # ash_content: higher ash → slower settling (longer tau)
    ash_content = RNG.uniform(4.0, 14.0)
    tau_ash_effect = 0.15 * (ash_content - 9.0)       # centred around 9

    # caliper: deviation from nominal increases overshoot
    caliper = RNG.uniform(0.06, 0.14)
    overshoot_caliper = 1.0 + 2.5 * abs(caliper - 0.10)   # centred on 0.10

    # recipe_headroom: tighter headroom → more deviation noise
    recipe_headroom = RNG.uniform(0.5, 5.0)
    noise_boost = max(0.0, 2.0 - recipe_headroom) * 0.4

    effective_tau = base_tau + tau_ash_effect
    effective_tau = max(2.0, effective_tau)              # floor

    # basis weight trajectory
    overshoot_mag = (bw_start - bw_target) * overshoot_caliper
    bw_start_eff  = bw_target + overshoot_mag
    noise_std     = 0.3 + noise_boost

    bw_trajectory = _exponential_settle(
        bw_target, bw_start_eff, effective_tau, MINUTES_PER_EVENT, noise_std
    )

    # controllable variable trajectories — same settling shape, own noise
    stock_traj = _exponential_settle(
        spec_to["stock"],  spec_from["stock"]  + RNG.normal(0, 8),
        effective_tau + RNG.normal(0, 0.5), MINUTES_PER_EVENT, 2.0
    )
    filler_traj = _exponential_settle(
        spec_to["filler"], spec_from["filler"] + RNG.normal(0, 1),
        effective_tau + RNG.normal(0, 0.5), MINUTES_PER_EVENT, 0.5
    )
    steam_traj = _exponential_settle(
        spec_to["steam"],  spec_from["steam"]  + RNG.normal(0, 0.1),
        effective_tau + RNG.normal(0, 0.5), MINUTES_PER_EVENT, 0.05
    )
    speed_traj = _exponential_settle(
        spec_to["speed"],  spec_from["speed"]  + RNG.normal(0, 15),
        effective_tau + RNG.normal(0, 0.5), MINUTES_PER_EVENT, 5.0
    )

    rows = []
    for m in range(MINUTES_PER_EVENT):
        rows.append({
            "event_id":           event_id,
            "minute":             m,
            "grade_from":         g_from,
            "grade_to":           g_to,
            "basis_weight":       round(bw_trajectory[m], 3),
            "basis_weight_target": round(bw_target, 3),
            "stock_flow":         round(stock_traj[m], 2),
            "filler_flow":        round(filler_traj[m], 2),
            "steam_pressure":     round(steam_traj[m], 3),
            "machine_speed":      round(speed_traj[m], 1),
            "ash_content":        round(ash_content, 3),
            "caliper":            round(caliper, 4),
            "recipe_headroom":    round(recipe_headroom, 3),
        })

    return pd.DataFrame(rows)


def generate_dataset(output_path="data/process_data.csv"):
    """Generate the full 140-event dataset and write to CSV."""
    frames = []
    for eid in range(1, NUM_EVENTS + 1):
        frames.append(_generate_single_event(eid))

    df = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[generate_data] wrote {len(df)} rows across {NUM_EVENTS} events -> {output_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
