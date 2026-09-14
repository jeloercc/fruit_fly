#!/usr/bin/env python3
"""
run_simulation.py — Bridges the real connectome (brain_graph.json, from
fetch_brain.py) to flygym's *native*, already-trained locomotor controller.

We do not train anything. `flygym.examples.locomotion.HybridTurningFly`
already contains a working CPG + sensory-correction walking controller
(Lobato-Rios et al. / NeuroMechFly ecosystem); it just needs a 2-D
[left_drive, right_drive] command each control step. That command is what
the brain model produces:

  1. BrainLIF loads the connectome as a sparse adjacency matrix and runs a
     simple current-based leaky integrate-and-fire (LIF) simulation on it
     (numpy/scipy — no training, no learned weights: synaptic weights come
     directly from neuPrint synapse counts, signed by each neuron's
     predicted/consensus neurotransmitter).
  2. Sensory-class neurons (superclass contains "sensory") are driven by a
     noisy input current every brain step — the "stimulus" the directive
     asks for, standing in for real visual/mechanosensory input.
  3. Descending Neurons (superclass in {'descending_neuron',
     'descending_neuron_tbc'}) are read out: their smoothed spike rate,
     split by soma side (L/R), is mapped to the 2-D turning command that
     HybridTurningFly's native CPG consumes directly.

World: flygym's MixedTerrain (blocks + gaps + slopes mixed along the
track), not an empty plane.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("run_simulation")

DN_SUPERCLASSES = {"descending_neuron", "descending_neuron_tbc"}
SENSORY_SUPERCLASSES = {
    "vnc_sensory", "ol_sensory", "cb_sensory", "sensory_ascending",
    "sensory_descending", "vnc_sensory_tbc", "cb_sensory_tbc",
    "sensory_ascending_tbc",
}
NT_SIGN = {
    "acetylcholine": 1.0,
    "dopamine": 1.0,
    "octopamine": 1.0,
    "serotonin": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}


# --------------------------------------------------------------------------
# Brain: connectome-driven LIF network
# --------------------------------------------------------------------------

class BrainLIF:
    """Leaky integrate-and-fire network built directly from the connectome
    adjacency matrix — synaptic weights are real neuPrint synapse counts
    (signed by neurotransmitter), not learned.

    Parameters and equations are ported directly from the published,
    connectome-constrained whole-brain LIF model — Shiu et al. 2023, "A
    leaky integrate-and-fire computational model based on the connectome of
    the entire adult Drosophila brain reveals insights into sensorimotor
    processing" (https://doi.org/10.1101/2023.05.02.539144), reference
    implementation at github.com/philshiu/Drosophila_brain_model/model.py:

        dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
        dg/dt = -g / tau                 (unless refractory)
        spike when v > v_th; on spike: v = v_rst, g = 0, refractory for t_rfc
        synapse: on presynaptic spike (after t_dly delay), g += w
        w per connection = (+1 or -1 per NT) * w_syn * (raw synapse count)
        external/sensory drive: independent Poisson input per neuron at
        rate r_poi, each event adding w_syn * f_poi directly to v (not g) —
        strong enough to reliably trigger a spike; Poisson-target neurons
        have no refractory period, matching the reference model exactly.

    v_0/v_rst/v_th/t_mbr are from Kakaria & de Bivort 2017; tau (synaptic)
    from Jürgensen et al.; t_rfc from Lazar et al.; t_dly from Paul et al.
    2015 — citations as given in the reference model.py.

    The one deliberate addition beyond the paper: an optional low-rate
    background Poisson drive on *all* neurons (`noise_std`, off by
    default), for interactive exploration via the dashboard slider — the
    original model only stimulates explicitly chosen neurons.
    """

    # Reference model.py constants (mV, ms, Hz) — see class docstring.
    V_0 = -52.0
    V_RST = -52.0
    V_TH = -45.0
    T_MBR = 20.0     # membrane time constant, ms
    TAU_SYN = 5.0    # synaptic conductance decay time constant, ms
    T_RFC = 2.2      # refractory period, ms
    T_DLY = 1.8      # synaptic delay, ms
    W_SYN = 0.275    # weight per synapse, mV
    R_POI = 150.0    # default Poisson input rate, Hz
    F_POI = 250.0    # Poisson synapse scaling factor

    SUB_DT = 0.5     # internal integration step, ms (<< tau_syn=5ms for accurate Euler)

    def __init__(
        self,
        graph_path: str | Path,
        dt: float = 2e-3,
        sensory_drive: float = 1.0,
        noise_std: float = 0.0,
        dn_rate_tau: float = 0.08,
        seed: int = 0,
    ):
        # `dt` (seconds) is the outer cadence at which the bridge calls
        # step() — tied to the physics-coupling rate, not the brain's own
        # numerical accuracy. Internally we sub-step at SUB_DT (ms) for a
        # stable, accurate Euler integration of the ms-scale time constants
        # above, then report back once per outer call.
        outer_dt_ms = dt * 1000.0
        self.n_sub = max(1, round(outer_dt_ms / self.SUB_DT))
        self.sub_dt = outer_dt_ms / self.n_sub  # exact split of the outer step

        self.sensory_drive = sensory_drive   # multiplier on R_POI for sensory neurons
        self.noise_std = noise_std           # background Poisson rate (Hz) for ALL neurons — our addition
        self.dn_rate_alpha = dt / dn_rate_tau
        self.rng = np.random.default_rng(seed)

        with open(graph_path) as f:
            graph = json.load(f)

        nodes = graph["nodes"]
        edges = graph["edges"]
        self.body_ids = np.array([n["id"] for n in nodes], dtype=np.int64)
        self.n = len(nodes)
        id_to_idx = {bid: i for i, bid in enumerate(self.body_ids)}
        self.id_to_idx = id_to_idx

        superclass = np.array([n.get("superclass") or "" for n in nodes])
        self.is_dn = np.array([n.get("is_dn", False) for n in nodes], dtype=bool)
        self.is_sensory = np.isin(superclass, list(SENSORY_SUPERCLASSES))
        soma_side = np.array([n.get("soma_side") or "" for n in nodes])
        self.dn_left = self.is_dn & (soma_side == "L")
        self.dn_right = self.is_dn & (soma_side == "R")
        # DNs with unknown/missing soma side still count toward the
        # symmetric "forward drive" pool, just not toward the L/R split.
        self.dn_unsided = self.is_dn & ~(soma_side == "L") & ~(soma_side == "R")

        log.info(
            "BrainLIF: %d neurons | sensory=%d | DN=%d (L=%d R=%d unsided=%d) | "
            "%d sub-steps of %.3fms per %.3fms outer step",
            self.n, self.is_sensory.sum(), self.is_dn.sum(),
            self.dn_left.sum(), self.dn_right.sum(), self.dn_unsided.sum(),
            self.n_sub, self.sub_dt, outer_dt_ms,
        )

        rows, cols, vals = [], [], []
        nt_by_pre = {n["id"]: str(n.get("nt") or "").lower() for n in nodes}
        dropped = 0
        for e in edges:
            pre, post, w = e["source"], e["target"], e["weight"]
            if pre not in id_to_idx or post not in id_to_idx:
                dropped += 1
                continue
            sign = NT_SIGN.get(nt_by_pre.get(pre, ""), 1.0)
            rows.append(id_to_idx[post])   # post-synaptic row -> receives current
            cols.append(id_to_idx[pre])    # pre-synaptic column -> spike source
            # w_syn per raw synapse count, signed by neurotransmitter —
            # matches `Excitatory x Connectivity * w_syn` in the reference.
            vals.append(sign * w * self.W_SYN)
        if dropped:
            log.warning("BrainLIF: dropped %d edges referencing unknown bodyIds", dropped)

        self.W = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        log.info("BrainLIF: adjacency matrix %s, nnz=%d", self.W.shape, self.W.nnz)

        # Synaptic delay ring buffer: a spike fired now reaches postsynaptic
        # conductance g only after T_DLY ms (Paul et al.), not instantly.
        self.delay_steps = max(1, round(self.T_DLY / self.sub_dt))
        self._delay_buf = [np.zeros(self.n, dtype=np.float64) for _ in range(self.delay_steps)]
        self._delay_ptr = 0

        self.V = np.full(self.n, self.V_0)
        self.g = np.zeros(self.n)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.refrac_ms = np.zeros(self.n)  # ms remaining in refractory period
        self.dn_rate = np.zeros(self.n)

    def _substep(self) -> np.ndarray:
        dt = self.sub_dt
        active = self.refrac_ms <= 0.0  # (unless refractory): frozen v & g otherwise

        # 1) Threshold check + reset, on the state carried in from the
        #    previous sub-step (organic crossings from that sub-step's
        #    integration, or a manually forced V from force_spike()).
        self.spikes = active & (self.V >= self.V_TH)
        self.V[self.spikes] = self.V_RST
        self.g[self.spikes] = 0.0
        # Poisson-input (sensory) targets get no refractory period at all,
        # exactly as in the reference model (`rfc = 0*ms` for those
        # neurons) — everything else enters the 2.2ms refractory window.
        newly_refractory = self.spikes & ~self.is_sensory
        self.refrac_ms[newly_refractory] = self.T_RFC

        # 2) Delayed synaptic input: the spikes fired T_DLY ms ago arrive now.
        delayed_spikes = self._delay_buf[self._delay_ptr]
        self._delay_buf[self._delay_ptr] = self.spikes.astype(np.float64)
        self._delay_ptr = (self._delay_ptr + 1) % self.delay_steps
        self.g[active] += (self.W @ delayed_spikes)[active]

        # 3) External Poisson drive: independent per-neuron spike process.
        #    Sensory neurons at sensory_drive * R_POI Hz; everything else
        #    (our addition, off by default) at a small background rate —
        #    both add directly to v, not g, per the reference's `target_var='v'`.
        if self.sensory_drive > 0 and self.is_sensory.any():
            p = self.sensory_drive * self.R_POI * dt / 1000.0
            fires = self.is_sensory & (self.rng.random(self.n) < p)
            self.V[fires] += self.W_SYN * self.F_POI
        if self.noise_std > 0:
            p_bg = self.noise_std * self.R_POI * dt / 1000.0
            fires_bg = active & (self.rng.random(self.n) < p_bg)
            self.V[fires_bg] += self.W_SYN * self.F_POI

        # 4) Leaky integration (only for non-refractory neurons).
        dv = (dt / self.T_MBR) * (self.V_0 - self.V + self.g)
        dg = -(dt / self.TAU_SYN) * self.g
        self.V[active] += dv[active]
        self.g[active] += dg[active]

        self.refrac_ms[~active] = np.maximum(0.0, self.refrac_ms[~active] - dt)
        return self.spikes

    def step(self) -> np.ndarray:
        """Advance the network by one *outer* timestep (n_sub internal
        sub-steps). Returns the union of spikes across those sub-steps —
        used for the live spike-visualization stream. `dn_rate` (used for
        the DN->CPG readout) is updated by EMA once per outer call, on the
        union of spikes, so its time constant (dn_rate_tau) is unaffected
        by the internal sub-stepping."""
        union_spikes = np.zeros(self.n, dtype=bool)
        for _ in range(self.n_sub):
            union_spikes |= self._substep()

        self.dn_rate += self.dn_rate_alpha * (union_spikes.astype(np.float64) - self.dn_rate)
        self.spikes = union_spikes
        return self.spikes

    def dn_turning_command(
        self, base_amp: float = 0.9, speed_gain: float = 1.2, turn_gain: float = 3.0,
        amp_range: tuple[float, float] = (0.2, 1.5),
    ) -> np.ndarray:
        """Translate DN spike-rate readout directly into the 2-D
        [left_drive, right_drive] command consumed by HybridTurningFly's
        native CPG — no training involved."""
        pool = self.dn_left.sum() + self.dn_right.sum() + self.dn_unsided.sum()
        if pool == 0:
            return np.array([base_amp, base_amp])

        overall = self.dn_rate[self.is_dn].mean() if self.is_dn.any() else 0.0
        left_rate = self.dn_rate[self.dn_left].mean() if self.dn_left.any() else overall
        right_rate = self.dn_rate[self.dn_right].mean() if self.dn_right.any() else overall

        forward = base_amp + speed_gain * overall
        bias = turn_gain * (left_rate - right_rate)
        action = np.array([forward + bias, forward - bias])
        return np.clip(action, *amp_range)

    def spiking_body_ids(self) -> list[int]:
        return self.body_ids[self.spikes].tolist()

    def force_spike(self, body_ids: list[int]) -> int:
        """Manual stimulus injection: push the given neurons' membrane
        potential above threshold so they fire on the *next* sub-step
        (routed through the normal integration, not faked after the fact) —
        equivalent to a manual Poisson-input event. Returns how many of the
        requested bodyIds were actually found."""
        idx = [self.id_to_idx[b] for b in body_ids if b in self.id_to_idx]
        if idx:
            self.V[idx] = self.V_TH + 1.0
            self.refrac_ms[idx] = 0.0  # make sure it isn't masked by a stale refractory window
        return len(idx)

    def reset_state(self):
        self.V[:] = self.V_0
        self.g[:] = 0.0
        self.spikes[:] = False
        self.refrac_ms[:] = 0.0
        self.dn_rate[:] = 0.0
        for buf in self._delay_buf:
            buf[:] = 0.0
        self._delay_ptr = 0


# --------------------------------------------------------------------------
# Bridge: BrainLIF <-> flygym's native CPG-driven fly, in a complex arena
# --------------------------------------------------------------------------

@dataclass
class SimSnapshot:
    t: float
    fly_pos: list
    fly_quat: list
    dn_action: list
    spiking_ids: list = field(default_factory=list)
    frame: object | None = None  # numpy RGB array or None


class NeuroMechFlyBrainBridge:
    """Owns the connectome brain, flygym's native NeuroMechFly + CPG
    controller, and the arena. Advances physics every call to `.step()`;
    the brain (and therefore the DN-derived turning command) is only
    recomputed every `brain_every` physics steps, held constant in between
    (standard zero-order-hold rate coupling between a fast physics loop and
    a coarser neural update)."""

    def __init__(
        self,
        graph_path: str | Path,
        # Tried 5e-4 for a real ~4x speedup — a 1000-tick isolated test
        # looked stable, but it broke under real live usage (longer runs,
        # pause/resume/inject cycles): the fly went airborne and tumbled off
        # the arena. My test wasn't representative enough to catch it, most
        # likely HybridTurningFly's stumbling/retraction correction, whose
        # increment-per-step is calibrated relative to this timestep,
        # overshooting at 5x the step size. Reverted to 1e-4 — the value
        # every other test this session actually ran long, live sessions on
        # without incident.
        physics_timestep: float = 1e-4,
        brain_dt: float = 2e-3,
        render: bool = True,
        seed: int = 0,
    ):
        from flygym import SingleFlySimulation, YawOnlyCamera
        from flygym.arena import MixedTerrain
        from flygym.examples.locomotion import HybridTurningFly

        self.physics_timestep = physics_timestep
        self.brain_every = max(1, round(brain_dt / physics_timestep))
        self.brain = BrainLIF(graph_path, dt=self.brain_every * physics_timestep, seed=seed)

        # NOTE: tried trimming this to just the 3 segments the stumbling
        # rule reads (Tibia/Tarsus1/Tarsus2) — broke adhesion's own contact
        # lookup (`_adhesion_bodies_with_contact_sensors`), which indexes
        # into this same list expecting the full tarsal chain. Not a safe
        # optimization; reverted to the complete set flygym's own example
        # uses with enable_adhesion=True.
        contact_sensor_placements = [
            f"{leg}{segment}"
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]
        self.fly = HybridTurningFly(
            enable_adhesion=True,
            contact_sensor_placements=contact_sensor_placements,
            seed=seed,
            timestep=physics_timestep,
        )
        self.cameras = []
        self.cam = None
        if render:
            # play_speed is meant for offline video export (it compresses
            # *simulated* time when the saved file is played back later) —
            # flygym's own examples use 0.1. For a *live* stream that
            # semantics is wrong: at play_speed=0.1/fps=30, a new frame is
            # only captured every ~33 physics ticks. Since physics already
            # runs far slower than real time (~90-100Hz wall-clock, not
            # 10,000Hz), that meant a new frame reached the browser only
            # every ~1/3 of a real second — the actual cause of the choppy
            # video, not a rendering bug. Setting play_speed so the
            # interval equals one physics tick means every completed
            # physics step yields its own frame, capping live frame
            # delivery at the physics rate itself (the true, honest
            # ceiling) instead of an unrelated video-export setting.
            fps = 30
            self.cam = YawOnlyCamera(
                attachment_point=self.fly.model.worldbody,
                camera_name="camera_right",
                targeted_fly_names=self.fly.name,
                fps=fps,
                play_speed=fps * physics_timestep,
                # This value only makes sense as a label on an *exported*
                # video ("this file plays back at 0.1x"); burned into a
                # live stream it just reads as a confusingly tiny number
                # with no relation to the actual live framerate. Disabled.
                play_speed_text=False,
            )
            self.cameras = [self.cam]

        self.sim = SingleFlySimulation(
            fly=self.fly, cameras=self.cameras, timestep=physics_timestep,
            arena=MixedTerrain(),
        )
        self.seed = seed
        self.sim.reset(seed=seed)
        self._action = np.array([0.9, 0.9])
        self._step_count = 0

    def reset(self):
        """Reset both halves of the bridge: the fly's physics/pose and the
        brain's membrane potentials/spikes — used by the dashboard's Reset
        control."""
        self.sim.reset(seed=self.seed)
        self.brain.reset_state()
        self._action = np.array([0.9, 0.9])
        self._step_count = 0

    def step(self) -> SimSnapshot:
        if self._step_count % self.brain_every == 0:
            self.brain.step()
            self._action = self.brain.dn_turning_command()

        obs, reward, terminated, truncated, info = self.sim.step(self._action)
        frame = None
        if self.cam is not None:
            # Camera.render() internally throttles real frame capture to
            # play_speed/fps of *simulated* time (here, roughly 1 real frame
            # per 33 physics ticks) — but it still pays for update_colors()
            # and per-camera bookkeeping even on ticks it's going to skip.
            # Replicate its own gating check here so we only pay that cost
            # on ticks that will actually produce a frame.
            due = self.sim.curr_time >= len(self.cam._frames) * self.cam._eff_render_interval
            if due:
                frames = self.sim.render()
                if frames and frames[0] is not None:
                    frame = frames[0]
                    # `_frames` is meant for a final save_video() batch
                    # export and otherwise grows unbounded — fine for a
                    # short demo script, a real memory leak for a
                    # long-running server (each entry is a full ~900KB RGB
                    # frame). We only ever need the latest one; drop the
                    # big array but keep the list *length*, since the
                    # camera's own gating check above (`len(_frames) *
                    # interval`) depends on that count to know when the
                    # next frame is due.
                    self.cam._frames[-1] = None

        self._step_count += 1
        return SimSnapshot(
            t=self._step_count * self.physics_timestep,
            fly_pos=obs["fly"][0].tolist(),
            fly_quat=obs["fly"][2].tolist() if len(obs["fly"]) > 2 else [1, 0, 0, 0],
            dn_action=self._action.tolist(),
            spiking_ids=self.brain.spiking_body_ids(),
            frame=frame,
        )


# --------------------------------------------------------------------------
# CLI: standalone smoke-test / demo run
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="data/brain_graph.json")
    ap.add_argument("--steps", type=int, default=2000, help="physics steps to run")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out-video", default="data/demo.mp4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bridge = NeuroMechFlyBrainBridge(
        args.graph, render=not args.no_render, seed=args.seed,
    )
    log.info("Running %d physics steps (brain updates every %d steps)...",
              args.steps, bridge.brain_every)

    t0 = time.time()
    total_spikes = 0
    for i in range(args.steps):
        snap = bridge.step()
        total_spikes += len(snap.spiking_ids)
        if i % 500 == 0:
            log.info("t=%.3fs pos=%s dn_action=%s spiking=%d",
                      snap.t, [f"{v:.3f}" for v in snap.fly_pos],
                      [f"{v:.2f}" for v in snap.dn_action], len(snap.spiking_ids))

    elapsed = time.time() - t0
    log.info("Done: %d steps in %.1fs (%.0f steps/s), %d total spikes",
              args.steps, elapsed, args.steps / elapsed, total_spikes)
    log.info("Final fly position: %s", snap.fly_pos)

    if not args.no_render and bridge.cameras:
        out_path = Path(args.out_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bridge.cam.save_video(str(out_path))
        log.info("Saved demo video -> %s", out_path)


if __name__ == "__main__":
    main()
