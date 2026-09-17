#!/usr/bin/env python3
"""
sham_causality_test.py — one-off metrology script, NOT part of the live
server/dashboard pipeline (imports run_simulation only, nothing imports
this).

Question: when the olfactory channel is driven, is the measured rise in
motor-neuron activation actually CAUSED by signal propagating along the
ORN -> ... -> DN -> VNC -> MN pathway, or is it just the network drifting
upward over time (the unresolved caveat from the first cascade test, where
the post-stimulus "recovery" block never returned to its own baseline)?

Design decisions that make this a real control rather than a demo:

  * SIZE-MATCHED SHAM. The sham is not a token handful of neurons — it is
    a random sample of exactly as many neurons as the ORN population
    (2,635), so both conditions inject the same total current into the same
    number of cells. A smaller sham would trivially produce a smaller
    response for reasons having nothing to do with connectivity.
  * SHAM IS CHOSEN BY MEASURED GRAPH DISTANCE, not by assumption: drawn
    only from neurons that cannot reach any vnc_motor neuron within
    MAX_HOPS synapses (reverse BFS over the real adjacency matrix). If the
    pathway story is true, current injected here must NOT move the motor
    readout.
  * INTERLEAVED, REPEATED BLOCKS. Conditions alternate
    OFF -> SHAM -> OFF -> ORN within every cycle, repeated N_CYCLES times,
    so any slow monotonic drift is shared by both conditions instead of
    landing on whichever one happened to run last. This is the specific
    confound the first test could not rule out.
  * PAIRED MEASUREMENT. Each ON block is compared against the OFF block
    immediately preceding it, so the statistic is a within-cycle
    difference, not a comparison against one distant global baseline.
  * SETTLING. Only the last SETTLE_TAIL steps of each block are averaged,
    because dn_rate (tau=0.08s) and the muscle filter (tau=0.05s) need
    roughly 25-40 steps to equilibrate after a transition.

Run:
    python3 sham_causality_test.py
"""
from __future__ import annotations

import numpy as np

from run_simulation import VisionFlightBridge

GRAPH = "data/brain_graph.json"
SEED = 0
DRIVE = 3.0
MAX_HOPS = 4
N_CYCLES = 5
OFF_STEPS = 60
ON_STEPS = 80
SETTLE_TAIL = 40
WARMUP = 80


def motor_unreachable_mask(brain, max_hops: int) -> np.ndarray:
    """Reverse BFS over the real connectivity: which neurons have NO
    synaptic path of <= max_hops onto any vnc_motor neuron. brain.W is
    [post, pre], so the presynaptic partners of a frontier are exactly the
    column indices appearing in those rows."""
    n = brain.n
    reached = brain.is_vnc_motor.copy()
    frontier = brain.is_vnc_motor.copy()
    for _ in range(max_hops):
        pre = np.zeros(n, dtype=bool)
        pre[np.unique(brain.W[frontier].indices)] = True
        new = pre & ~reached
        if not new.any():
            break
        reached |= new
        frontier = new
    return ~reached


def block(bridge, n_steps: int, tail: int) -> np.ndarray:
    """Step n_steps, return the mean of the last `tail` samples of
    (MN_L, MN_R, premotor_DN) — all in Hz."""
    rows = []
    for _ in range(n_steps):
        s = bridge.step()
        rows.append((s.mn_activation_left, s.mn_activation_right, s.dn_premotor_rate))
    return np.array(rows[-tail:]).mean(axis=0)


def main():
    rng = np.random.default_rng(SEED)
    bridge = VisionFlightBridge(GRAPH, seed=SEED)
    brain = bridge.brain

    orn = brain._group_masks["olfactory"]
    unreachable = motor_unreachable_mask(brain, MAX_HOPS)
    pool = np.flatnonzero(unreachable)
    n_target = int(orn.sum())
    print(f"ORN population              : {n_target}")
    print(f"motor-unreachable pool (<={MAX_HOPS} hops): {pool.size}")
    if pool.size < n_target:
        print(f"WARNING: pool smaller than ORN; sham will be {pool.size} neurons, NOT size-matched")
    chosen = rng.choice(pool, size=min(n_target, pool.size), replace=False)
    sham = np.zeros(brain.n, dtype=bool)
    sham[chosen] = True
    print(f"sham population             : {int(sham.sum())} "
          f"(size-matched: {int(sham.sum()) == n_target})")

    # Register the sham as a drivable channel using the exact same
    # injection machinery the real sensory channels use (_substep reads
    # group_drive/_group_masks), so the two conditions differ ONLY in which
    # neurons receive the current.
    brain._group_masks["__sham__"] = sham
    brain.group_drive["__sham__"] = 0.0

    def set_drive(ch, v):
        brain.group_drive["olfactory"] = v if ch == "orn" else 0.0
        brain.group_drive["__sham__"] = v if ch == "sham" else 0.0

    set_drive(None, 0.0)
    block(bridge, WARMUP, 1)

    deltas = {"sham": [], "orn": []}
    print(f"\nrunning {N_CYCLES} interleaved cycles "
          f"(OFF{OFF_STEPS} -> SHAM{ON_STEPS} -> OFF{OFF_STEPS} -> ORN{ON_STEPS})...")
    for cyc in range(N_CYCLES):
        for cond in ("sham", "orn"):
            set_drive(None, 0.0)
            off = block(bridge, OFF_STEPS, SETTLE_TAIL)
            set_drive(cond, DRIVE)
            on = block(bridge, ON_STEPS, SETTLE_TAIL)
            deltas[cond].append(on - off)
        set_drive(None, 0.0)
        print(f"  cycle {cyc + 1}/{N_CYCLES} done")

    # NOTE the differing units: mn_activation is the muscle-filter output,
    # a normalized 0..1-ish activation (NOT converted to Hz), while
    # dn_premotor_rate IS in Hz (dn_readout() applies the 1/dt factor).
    labels = ["MN_L (activation)", "MN_R (activation)", "premotorDN (Hz)"]
    sh = np.array(deltas["sham"])
    orn_d = np.array(deltas["orn"])

    print("\n" + "=" * 68)
    print("PAIRED WITHIN-CYCLE DELTA (ON block minus its own preceding OFF block)")
    print("=" * 68)
    print(f"{'signal':<18}{'SHAM Δ':>16}{'ORN Δ':>16}{'ORN-SHAM':>16}")
    for i, lab in enumerate(labels):
        s, o = sh[:, i], orn_d[:, i]
        s_sem = s.std(ddof=1) / np.sqrt(len(s))
        o_sem = o.std(ddof=1) / np.sqrt(len(o))
        print(f"{lab:<18}{s.mean():>9.3f}±{s_sem:<6.3f}{o.mean():>9.3f}±{o_sem:<6.3f}"
              f"{o.mean() - s.mean():>16.3f}")

    # Paired t-test, ORN vs SHAM, same cycle index — implemented directly
    # so the arithmetic is visible rather than hidden behind a library call.
    print("\npaired t-test  ORN vs SHAM  (n=%d cycles, df=%d):" % (N_CYCLES, N_CYCLES - 1))
    for i, lab in enumerate(labels):
        d = orn_d[:, i] - sh[:, i]
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / sem if sem > 0 else float("nan")
        print(f"  {lab:<18} mean_diff={d.mean():+8.4f}  sem={sem:7.4f}  t({N_CYCLES - 1})={t:+7.3f}")
    print("\n(two-sided critical |t| at df=4: 2.776 for p<0.05, 4.604 for p<0.01)")


if __name__ == "__main__":
    main()
