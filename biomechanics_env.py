#!/usr/bin/env python3
"""
biomechanics_env.py — R&D sandbox, still isolated from production (not
imported by run_simulation.py / telemetry_server.py / the dashboard, and
imports nothing from them).

Phase 1 (phase1_gravity_test): confirmed flygym/MuJoCo initialize and step
headlessly on this Mac with no GLFW/main-thread crash, and that flygym's
`Fly` is a 42-joint LEG-based NeuroMechFly model (no wing actuators) under
position (PD target-angle) control — not the torque-driven wing-hinge model
the original migration plan assumed. See that function's docstring/the
Phase-1 report for the full findings.

Phase 2 (phase2_forward_walk): a stable resting pose (fly's own
`init_pose="stretch"`, not a hand-picked one) plus a tripod-gait CPG
controller, adapted directly from flygym's own bundled reference example
at flygym/examples/locomotion/cpg_controller.py — not reimplemented from
scratch, so this is flygym's own tested tripod phase-bias matrix and
step-angle mapping (PreprogrammedSteps), not a guess at what a plausible
gait might look like. The one real addition: a single "forward drive"
scalar (standing in for the eventual SNN Descending Neuron readout) scales
the CPG's shared stepping frequency across all six legs.

Superseded: this file previously had a bake_walk_cycle()/--bake mode that
exported a baked walk cycle to dashboard/public/walk_cycle_data.json for
frontend kinematic replay. Removed by explicit direction — the project
moved to a live, unbaked flygym/MuJoCo thread instead (see
physics_worker.py, used by telemetry_server.py), accepting the same
~0.03x-real-time cost measured here rather than approximating it. This
file remains the isolated sandbox where that cost was originally measured
and the CPG approach validated; it is still not imported by
telemetry_server.py / run_simulation.py / the dashboard.

Run:
    python3 biomechanics_env.py            # runs Phase 2 (console report)
    python3 biomechanics_env.py --phase1   # runs Phase 1 instead
"""

import sys

import numpy as np


def phase1_gravity_test():
    import flygym

    print(f"flygym {flygym.__name__} loaded from {flygym.__file__}")

    fly = flygym.Fly(enable_adhesion=True, enable_vision=False)
    sim = flygym.SingleFlySimulation(fly=fly)  # flygym.NeuroMechFly doesn't exist in this version

    obs, info = sim.reset(seed=0)
    print("obs keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: shape={getattr(v, 'shape', None)} sample={np.asarray(v).ravel()[:3]}")

    action_space = sim.action_space
    null_action = {k: np.zeros_like(v) for k, v in action_space.sample().items()}

    n_steps = round(1.0 / sim.timestep)
    print(f"\nstepping {n_steps} times with a NULL action (dt={sim.timestep}s -> {n_steps * sim.timestep:.3f}s sim time)...")
    for i in range(n_steps):
        obs, reward, terminated, truncated, info = sim.step(null_action)
        if terminated or truncated:
            print(f"  episode ended early at step {i}: terminated={terminated} truncated={truncated}")
            break

    print("\nfinal obs['fly'] (pos/vel/orientation block):")
    print(obs.get("fly"))
    print("\n-> a null action snaps all 42 joint PD targets to 0 rad at once, which is")
    print("   NOT the resting pose (init_pose='stretch' is) — this is why it tumbled")
    print("   rather than settling. See phase2_forward_walk() for the fix.")


def phase2_forward_walk(forward_drive: float = 1.0, run_time: float = 1.0):
    """forward_drive in roughly [0, 1.5]: 0 = legs still cycle in place at a
    slow idle frequency (a real CPG's oscillators don't have a hard "off"
    state without also risking the same non-resting-pose problem Phase 1
    hit), higher = faster tripod stepping = faster forward walking. This
    scalar is where the SNN's DN readout will eventually plug in — nothing
    else about the wiring changes when that happens, just this one number's
    source (see run_simulation.py's existing flight_command() for the real
    analogous DN-rate -> motor-scalar reduction the tunnel model already
    uses)."""
    import flygym
    from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps

    timestep = 1e-4

    # Tripod phase-bias matrix + coupling weights: flygym's own reference
    # values (flygym/examples/locomotion/cpg_controller.py), not invented.
    # Adjacent legs on the same side alternate (bias=pi); each leg's
    # tripod partner (the two diagonal-opposite legs) moves in phase.
    base_freq = 4.0    # Hz at forward_drive=0 — idle stepping, not stopped
    freq_gain = 8.0    # Hz added per unit of forward_drive
    intrinsic_freqs = np.ones(6) * (base_freq + forward_drive * freq_gain)
    intrinsic_amps = np.ones(6) * 1.0
    phase_biases = np.pi * np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ])
    coupling_weights = (phase_biases > 0) * 10
    convergence_coefs = np.ones(6) * 20

    cpg = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )
    steps = PreprogrammedSteps()

    # init_pose="stretch" (the fly's own default) is what actually fixes
    # the Phase-1 collapse — not a substitute action array. The CPG's
    # per-step joint angles (below) take over from there once stepping starts.
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    sim = flygym.SingleFlySimulation(fly=fly, timestep=timestep)

    obs, info = sim.reset(seed=0)
    x0, y0, z0 = obs["fly"][0]
    print(f"forward_drive={forward_drive} -> stepping frequency {intrinsic_freqs[0]:.1f}Hz")
    print(f"start position: x={x0:.4f} y={y0:.4f} z={z0:.4f}")

    n_steps = round(run_time / timestep)
    for _ in range(n_steps):
        cpg.step()
        joints_angles, adhesion_onoff = [], []
        for i, leg in enumerate(steps.legs):
            joints_angles.append(steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i]))
            adhesion_onoff.append(steps.get_adhesion_onoff(leg, cpg.curr_phases[i]))
        action = {
            "joints": np.concatenate(joints_angles),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        obs, reward, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            print(f"  episode ended early: terminated={terminated} truncated={truncated}")
            break

    x1, y1, z1 = obs["fly"][0]
    dist = np.hypot(x1 - x0, y1 - y0)
    print(f"end position:   x={x1:.4f} y={y1:.4f} z={z1:.4f}")
    print(f"horizontal distance traveled: {dist:.4f} (z drift: {z1 - z0:+.4f} -- should stay small if standing stably)")


if __name__ == "__main__":
    if "--phase1" in sys.argv:
        phase1_gravity_test()
    else:
        phase2_forward_walk(forward_drive=1.0)
