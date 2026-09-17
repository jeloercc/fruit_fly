#!/usr/bin/env python3
"""
intention_integration.py — read-only metrology (nothing in the live
pipeline imports it; it mutates only its own in-process brain).

Question: does dopamine-induced depression at the KC->MBON layer change
the motor INTENTION signal, measured at the premotor descending neurons —
i.e. upstream of the fan-out that Phase 1 showed dilutes everything by
hop 4? Phase 2 established the weights really do change. This asks whether
that change propagates far enough to matter.

Protocol: pre-measure, train, roll back state (weights kept), post-measure.

The control is the part that makes this interpretable. Measuring only the
paired odour cannot distinguish associative learning from a global shift:
the depression could simply lower every MBON's output regardless of which
odour is presented. So TWO odours are measured at every stage —

    A = food glomeruli (ORN_DM1/DM2/DM4/VA2/VM2)   <- PAIRED with reward
    B = CO2/aversive glomeruli (ORN_V/DA2/DL5)     <- NEVER paired

and the quantity of interest is the DIFFERENCE of their changes. If only A
moves, learning is odour-specific and reached the DNs. If A and B move
together, the depression is generalised (which is what Phase 2 predicted
for the unclamped network) or the network drifted.

Every measurement resets transient state (V, g, eligibility, dopamine EMAs,
intent latch) but NOT the weights — that is the "rollback" — then settles
before recording, because an unsettled network was the artifact that
invalidated three earlier tests in this project.
"""
from __future__ import annotations

import numpy as np

from run_simulation import VisionFlightBridge

GRAPH = "data/brain_graph.json"
SEED = 0
ODOUR_DRIVE = 3.0
REPEATS = 2          # measurement blocks per odour per stage
SETTLE = 200         # steps to settle after a reset before recording
MEASURE = 150        # steps averaged per block
TRAIN_STEPS = 1200   # sized for ~25% depression at eta=5e-5
TRAIN_WARM = 600


def main():
    bridge = VisionFlightBridge(GRAPH, seed=SEED)
    br = bridge.brain
    pam_c, ppl_c = br.plastic_in_pam_comp, br.plastic_in_ppl_comp
    pam_ids = br.body_ids[br.is_pam].tolist()

    def silence():
        for k in br.group_drive:
            br.group_drive[k] = 0.0

    def measure(channel):
        """Premotor-DN and MBON spike fraction driven by one odour, from a
        rolled-back transient state with the learned weights intact."""
        br.set_learning(False)
        br.reset_state()          # clears V/g/eligibility/EMAs/intent, NOT weights
        silence()
        br.set_group_drive(f"{channel}_L", ODOUR_DRIVE)
        br.set_group_drive(f"{channel}_R", ODOUR_DRIVE)
        for _ in range(SETTLE):
            bridge.step()
        dn, mb = [], []
        for _ in range(MEASURE):
            bridge.step()
            dn.append(br.spikes[br.dn_premotor].mean())
            mb.append(br.spikes[br.is_mbon].mean())
        silence()
        return float(np.mean(dn)), float(np.mean(mb))

    def stage(label):
        out = {}
        for ch in ("food", "co2"):
            vals = [measure(ch) for _ in range(REPEATS)]
            out[ch] = (np.array([v[0] for v in vals]), np.array([v[1] for v in vals]))
            print(f"  {label:<5} odour {ch:<5} premotorDN={out[ch][0].mean():.5f} "
                  f"MBON={out[ch][1].mean():.5f}")
        return out

    print(f"eta={br.plasticity_eta}  premotor DNs={int(br.dn_premotor.sum())}  MBONs={int(br.is_mbon.sum())}")
    print("\n--- PRE ---")
    pre = stage("PRE")

    print("\n--- TRAINING (odour A = food, + PAM reward, synthetic clamp ON) ---")
    br.set_dan_clamp(True)
    silence()
    br.set_group_drive("food_L", ODOUR_DRIVE)
    br.set_group_drive("food_R", ODOUR_DRIVE)
    br.set_learning(False)
    for i in range(TRAIN_WARM):          # warm the dopamine EMAs before learning
        if i % 4 == 0:
            br.force_spike(pam_ids)
        bridge.step()
    w0 = br.W.data[br.plastic_pos].copy()
    br.set_learning(True)
    for i in range(TRAIN_STEPS):
        if i % 4 == 0:
            br.force_spike(pam_ids)
        bridge.step()
    br.set_learning(False)
    m0, m1 = np.abs(w0), np.abs(br.W.data[br.plastic_pos])
    r = (m1 - m0) / np.maximum(m0, 1e-12)
    print(f"  weight change: PAM comp {100 * r[pam_c].mean():+.2f}%  "
          f"PPL comp {100 * r[ppl_c].mean():+.2f}%  at floor: {int((r[pam_c] < -0.899).sum())}")

    print("\n--- POST (state rolled back, weights kept) ---")
    post = stage("POST")

    print("\n" + "=" * 70)
    print("MOTOR INTENTION AT THE PREMOTOR DNs")
    print("=" * 70)
    res = {}
    for ch in ("food", "co2"):
        a, b = pre[ch][0], post[ch][0]
        d = b.mean() - a.mean()
        res[ch] = d
        print(f"  odour {ch:<5} pre={a.mean():.5f}  post={b.mean():.5f}  "
              f"delta={d:+.6f}  ({100 * d / max(a.mean(), 1e-12):+.2f}%)")
    print(f"\n  PAIRED(food) - UNPAIRED(co2) delta = {res['food'] - res['co2']:+.6f}")
    print("  (associative learning reaching the DNs would make this clearly non-zero;")
    print("   a shared shift means generalised depression or drift, not association)")

    print("\nMBON layer (the site of the depression, for reference):")
    for ch in ("food", "co2"):
        a, b = pre[ch][1], post[ch][1]
        print(f"  odour {ch:<5} pre={a.mean():.5f}  post={b.mean():.5f}  delta={b.mean() - a.mean():+.6f}")


if __name__ == "__main__":
    main()
