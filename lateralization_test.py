#!/usr/bin/env python3
"""
lateralization_test.py — one-off metrology script (nothing in the live
pipeline imports it).

Question: does asymmetric olfactory stimulation produce an asymmetric
motor output? That is the entire premise of steering — food odour on the
left should drive the left/right motor-neuron populations differently than
the same odour on the right, because only that difference can yaw the
MuJoCo body.

Why this design is strong even though the earlier sham test came back
null: the two conditions here are exact MIRROR IMAGES of each other
(left-only vs right-only, identical drive, near-identical population
sizes). Any network drift, any slow upward trend, any global excitability
change affects BOTH conditions the same way and cancels in the
left-minus-right asymmetry. The measured quantity is a difference of
differences, so it does not depend on a stable absolute baseline — which
is precisely the weakness that invalidated the first cascade test.

Reported metric per ON block: ASYM = MN_L - MN_R.
Prediction if the pathway is functional and lateralized:
    ASYM(left-odour) > ASYM(right-odour)
with the sign consistent across cycles. A null here means the ORN ->
... -> MN pathway does not carry a usable lateral signal at this drive.

Run:
    python3 lateralization_test.py
"""
from __future__ import annotations

import numpy as np

from run_simulation import VisionFlightBridge

GRAPH = "data/brain_graph.json"
SEED = 0
DRIVE = 5.0          # set_group_drive clamps at 5.0 — use the ceiling for max sensitivity
N_CYCLES = 5
OFF_STEPS = 60
ON_STEPS = 80
SETTLE_TAIL = 40
WARMUP = 80
CHANNEL = "food"
# Extreme-SNR ("acoustic isolation") mode: silence the background so the
# only thing driving the network is the stimulus under test.
#   noise_std=0    -> removes the tonic background Poisson drive on all cells
#   sensory_drive=0 -> removes the camera/visual pathway AND the per-class
#                      photoreceptor afference (both scale by sensory_drive)
# The valence channels are NOT affected by either: group_drive injection in
# _substep multiplies by R_POI only, never by sensory_drive, so the food
# stimulus still reaches the network at full strength.
QUIET = True
SPIKE_COL = 3


def block(bridge, n_steps: int, tail: int) -> np.ndarray:
    rows = []
    for _ in range(n_steps):
        s = bridge.step()
        rows.append((s.mn_activation_left, s.mn_activation_right,
                     s.mn_activation_left - s.mn_activation_right,
                     len(s.spiking_ids)))
    return np.array(rows[-tail:]).mean(axis=0)


def main():
    bridge = VisionFlightBridge(GRAPH, seed=SEED)
    brain = bridge.brain
    nl = int(brain._group_masks[f"{CHANNEL}_L"].sum())
    nr = int(brain._group_masks[f"{CHANNEL}_R"].sum())
    print(f"channel '{CHANNEL}': L={nl} neurons, R={nr} neurons, drive={DRIVE}")
    if QUIET:
        brain.noise_std = 0.0
        brain.sensory_drive = 0.0
        for k in brain.group_drive:
            brain.group_drive[k] = 0.0
        for k in brain.visual_afference:
            brain.visual_afference[k] = 0.0
        print("QUIET MODE: noise_std=0, sensory_drive=0, all other channels zeroed")

    def set_side(side):
        brain.set_group_drive(f"{CHANNEL}_L", DRIVE if side == "L" else 0.0)
        brain.set_group_drive(f"{CHANNEL}_R", DRIVE if side == "R" else 0.0)

    set_side(None)
    quiet_check = block(bridge, WARMUP, SETTLE_TAIL)
    print(f"baseline spiking with no stimulus: {quiet_check[SPIKE_COL]:.0f} neurons/step "
          f"({100 * quiet_check[SPIKE_COL] / brain.n:.3f}% of {brain.n})")

    res = {"L": [], "R": []}
    print(f"\n{N_CYCLES} mirror-image cycles (OFF -> Lodour -> OFF -> Rodour)...")
    for cyc in range(N_CYCLES):
        for side in ("L", "R"):
            set_side(None)
            off = block(bridge, OFF_STEPS, SETTLE_TAIL)
            set_side(side)
            on = block(bridge, ON_STEPS, SETTLE_TAIL)
            res[side].append(on - off)          # paired against its own OFF
        set_side(None)
        print(f"  cycle {cyc + 1}/{N_CYCLES}")

    L = np.array(res["L"])
    R = np.array(res["R"])
    print("\n" + "=" * 62)
    print("PAIRED DELTA (ON minus its own preceding OFF)")
    print("=" * 62)
    print(f"{'measure':<22}{'L-odour':>18}{'R-odour':>18}")
    for i, lab in enumerate(["MN_L", "MN_R", "ASYM (MN_L-MN_R)", "spiking neurons"]):
        ls, rs = L[:, i], R[:, i]
        print(f"{lab:<22}{ls.mean():>11.5f}±{ls.std(ddof=1) / np.sqrt(len(ls)):<6.5f}"
              f"{rs.mean():>11.5f}±{rs.std(ddof=1) / np.sqrt(len(rs)):<6.5f}")

    d = L[:, 2] - R[:, 2]        # difference of asymmetries: the steering signal
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else float("nan")
    print(f"\nSTEERING SIGNAL  ASYM(L-odour) - ASYM(R-odour)")
    print(f"  mean={d.mean():+.6f}  sem={sem:.6f}  t({N_CYCLES - 1})={t:+.3f}")
    print(f"  per-cycle: {np.round(d, 6).tolist()}")
    print(f"  sign consistent across all {N_CYCLES} cycles: "
          f"{bool(np.all(d > 0) or np.all(d < 0))}")
    print("\n(two-sided critical |t| at df=4: 2.776 for p<0.05)")


if __name__ == "__main__":
    main()
