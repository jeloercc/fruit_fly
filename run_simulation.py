#!/usr/bin/env python3
"""
run_simulation.py — connectome-driven LIF brain for a vision-first flight
navigation model. No flygym/MuJoCo body simulation: the fly's body and its
"park" environment are simulated kinematically in the browser (dashboard.jsx,
React Three Fiber), not here. This file owns only the brain:

  1. BrainLIF loads the connectome as a sparse adjacency matrix and runs the
     connectome-constrained LIF model (Shiu et al. 2023 parameters — see the
     class docstring) — no training, no learned weights.
  2. Optic-lobe sensory neurons (superclass 'ol_sensory', split L/R by soma
     side) are driven by live brightness samples the browser renders from
     the fly's-eye camera in the park scene — the actual compound-eye
     pathway, sent in over the WebSocket each tick (see set_visual_input()).
  3. Descending Neurons are read out (same L/R-rate readout as before) and
     turned into a (thrust, yaw_rate) flight command, sent back to the
     browser, which integrates the avatar's position kinematically — no
     physics engine, so no physics bottleneck either.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _hash_body_ids(body_ids: np.ndarray) -> str:
    """Stable identity fingerprint of a graph's node set, order-independent
    (sorted before hashing) — see BrainLIF.graph_fingerprint."""
    return hashlib.sha1(np.sort(body_ids).tobytes()).hexdigest()[:16]

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
        # Was 0.4, then dropped to 0.0 after the graph grew to the full
        # ~176k-neuron dataset (real APL recurrent inhibition made the old
        # tonic drive redundant for sustaining DN activity — see below).
        # But "DNs still fire" was the wrong metric: it doesn't show the
        # visual pathway is still *driving* behavior rather than just
        # ambient noise doing so. Measured directly (sweep set_visual_input
        # between (1,0) and (0,1), same seed, track yaw_rate delta against
        # its own trial-to-trial std as a noise floor, 100-step windows on
        # the full 176k graph):
        #   noise_std=0.0: coupling=-0.0006 (noise~0.0026) — BELOW its own
        #     noise floor, i.e. statistically indistinguishable from zero.
        #     0.0 silently killed vision->behavior coupling even though DN
        #     spike counts looked fine.
        #   noise_std=0.2: coupling=+0.0071 (noise~0.0051, ~1.4x SNR) —
        #     KC_active=0.174, real signal, better sparsity than 0.4.
        #   noise_std=0.4: coupling=+0.0048 (noise~0.0046, ~1.0x SNR) —
        #     barely above its own noise floor too; KC_active=0.209.
        # None of these are a clean signal — this coupling metric is noisy
        # at every setting tested on this graph — but 0.2 is the only point
        # where the signal clears its own noise floor by a real margin
        # while KC sparsity stays better than the old 0.4 default. Settled
        # on 0.2 as the measured trade-off, not picked by eye; the KC
        # sparsity numbers (94%->76% plastic-synapse selectivity) from the
        # 0.0 test are still real, just achieved at a value that also broke
        # the pathway plasticity is supposed to be shaping in the first
        # place.
        noise_std: float = 0.2,
        dn_rate_tau: float = 0.08,
        seed: int = 0,
        plasticity_enabled: bool = False,
        plasticity_eta: float = 0.15,
        # Swept 2000/1000/500/200/100ms against a realistic multi-pulse PPL
        # protocol (8 forced-aversive events over 150 outer steps, same as
        # a real learning episode would look like) and measured fraction of
        # the 44,042 plastic KC->MBON synapses moved >1%: 0.908, 0.907,
        # 0.908, 0.912, 0.918 — essentially flat. Shortening this window
        # does NOT buy selectivity here: with repeated dopamine pulses,
        # weights saturate against the same synapses regardless of how
        # fast eligibility decays between pulses, because decay resets
        # per-coincidence rather than gating which pairs ever coincide.
        # The real selectivity lever is which KC/MBON pairs spike together
        # at all, i.e. KC sparsity (APL, see noise_std above) — not this
        # time constant. Left at the literature-grounded 2000ms rather
        # than shortened on a mechanism that measurably doesn't help.
        tau_elig_ms: float = 2000.0,
        dopamine_rate_tau: float = 0.5,
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

        self.learning_enabled = plasticity_enabled
        self.plasticity_eta = plasticity_eta
        self.elig_decay_per_substep = np.exp(-self.SUB_DT / tau_elig_ms)
        self.dopamine_rate_alpha = dt / dopamine_rate_tau
        self.dopamine_rate = 0.0       # signed EMA: PAM(reward) - PPL(punishment)
        self.cumulative_reward = 0.0

        with open(graph_path) as f:
            graph = json.load(f)

        nodes = graph["nodes"]
        edges = graph["edges"]

        # The browser dashboard only ever reads `nodes` + `meta` (soma
        # positions, type/class/superclass for coloring, is_dn/soma_side for
        # DN picking) — it never touches `edges`. But on the full male-cns
        # dataset `edges` (10.6M synapse-count triples) is >90% of this
        # file's bytes, so shipping the raw graph_path to the browser means
        # `fetch().json()` has to buffer+parse ~540MB just to throw away
        # everything but a ~35MB slice of it, which is what froze the tab.
        # Write that slice out once per load as a sidecar the dashboard
        # fetches instead; regenerated every time (cheap relative to the
        # json.load above) so a graph swap (watch_and_swap.py) can't leave a
        # stale one behind.
        nodes_sidecar_path = Path(graph_path).with_name(Path(graph_path).stem + "_nodes.json")
        with open(nodes_sidecar_path, "w") as f:
            json.dump({"meta": graph.get("meta", {}), "nodes": nodes}, f)
        log.info("wrote nodes-only sidecar for dashboard: %s (%.1f MB, dropped %d edges)",
                  nodes_sidecar_path, nodes_sidecar_path.stat().st_size / 1e6, len(edges))
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

        # Vision-first pivot: sensory neurons are the optic-lobe input layer
        # (ol_sensory — real neuPrint superclass for the compound-eye-adjacent
        # photoreceptor/lamina neurons), split L/R by soma side since that's
        # the real anatomical eye split available in the connectome metadata
        # — we don't have per-ommatidium retinotopic coordinates, so this is
        # the coarsest honest mapping: left-eye brightness drives left-soma
        # sensory neurons, right-eye brightness drives right-soma ones.
        self.is_sensory_left = self.is_sensory & (soma_side == "L")
        self.is_sensory_right = self.is_sensory & (soma_side == "R")
        self.is_sensory_unsided = self.is_sensory & ~(soma_side == "L") & ~(soma_side == "R")
        self.visual_L = 0.5  # normalized brightness (0..1), set live by set_visual_input()
        self.visual_R = 0.5

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

        self._build_plasticity(nodes)

    def _build_plasticity(self, nodes):
        """Locates the real KC->MBON synapses in W and prepares a bounded
        eligibility-trace array for them — real neuPrint metadata, not
        assumed: `class == 'Kenyon_Cell'` (4,064 neurons in this dataset)
        and `class == 'MBON'` (97 neurons), confirmed present via a direct
        neuPrint query. No 'kenyon_cell'/'mushroom_body' *superclass* exists
        in male-cns:v1.0 — Kenyon cells and MBONs live one level down, in
        the finer-grained `class` field, alongside ~30k other cb_intrinsic
        neurons that field alone doesn't separate out. Dopaminergic neurons
        (class == 'DAN', 340 in this dataset) split into the real PAM
        (reward/appetitive, 316 neurons, 15 subtypes) and PPL
        (punishment/aversive, 24 neurons, 12 subtypes) clusters by `type`
        prefix — also confirmed against real data, matching the published
        Drosophila mushroom-body literature (Aso et al.), not assumed from
        it."""
        classes = np.array([n.get("class") or "" for n in nodes])
        types = np.array([str(n.get("type") or "") for n in nodes])

        self.is_kc = classes == "Kenyon_Cell"
        self.is_mbon = classes == "MBON"
        self.is_dan = classes == "DAN"
        self.is_pam = self.is_dan & np.char.startswith(types, "PAM")
        self.is_ppl = self.is_dan & np.char.startswith(types, "PPL")

        log.info(
            "plasticity: KC=%d MBON=%d DAN=%d (PAM/reward=%d PPL/punishment=%d)",
            self.is_kc.sum(), self.is_mbon.sum(), self.is_dan.sum(),
            self.is_pam.sum(), self.is_ppl.sum(),
        )

        if self.is_kc.sum() == 0 or self.is_mbon.sum() == 0:
            log.warning("plasticity: no KC or MBON neurons in this graph — "
                        "plastic subset is empty, learning is a documented no-op")
            self.plastic_pos = np.array([], dtype=np.int64)
            self.plastic_pre_idx = np.array([], dtype=np.int64)
            self.plastic_post_idx = np.array([], dtype=np.int64)
            self.eligibility = np.array([], dtype=np.float64)
            self.W0_plastic = np.array([], dtype=np.float64)
            self._w_sign_plastic = np.array([], dtype=np.float64)
            self._w_mag0_plastic = np.array([], dtype=np.float64)
            return

        kc_idx = set(np.where(self.is_kc)[0].tolist())
        mbon_idx = set(np.where(self.is_mbon)[0].tolist())

        # Find KC->MBON entries within W's actual CSR structure, not the
        # pre-construction edge list — scipy's csr_matrix constructor can
        # reorder (and sum duplicate) entries, so positions must be read
        # back off the real, canonicalized sparsity pattern.
        W = self.W
        W.sort_indices()
        plastic_pos, plastic_pre, plastic_post = [], [], []
        for post in mbon_idx:
            start, end = W.indptr[post], W.indptr[post + 1]
            for local_i, pre in enumerate(W.indices[start:end]):
                if pre in kc_idx:
                    plastic_pos.append(start + local_i)
                    plastic_pre.append(pre)
                    plastic_post.append(post)

        self.plastic_pos = np.array(plastic_pos, dtype=np.int64)
        self.plastic_pre_idx = np.array(plastic_pre, dtype=np.int64)
        self.plastic_post_idx = np.array(plastic_post, dtype=np.int64)
        self.eligibility = np.zeros(len(plastic_pos), dtype=np.float64)
        self.W0_plastic = self.W.data[self.plastic_pos].copy()
        self._w_sign_plastic = np.sign(self.W0_plastic)
        self._w_mag0_plastic = np.abs(self.W0_plastic)

        # Integrity check, not decoration: plastic_pos/pre_idx/post_idx are
        # derived fresh from real bodyId identity (is_kc/is_mbon, computed
        # from THIS graph's own `class` field) every time _build_plasticity
        # runs, never cached across a graph swap — the same class of risk
        # already fixed once in fetch_brain.py's edge checkpoints (a
        # positional key silently surviving a scope change). There is
        # currently no cross-process state of any kind for W or the
        # plastic subset (grep confirms no pickle/np.save/joblib anywhere
        # in this project) — watch_and_swap.py's graph swap works by
        # killing the whole server process and starting a fresh one, which
        # rebuilds BrainLIF from scratch against whatever's on disk at that
        # moment. This assertion exists so that if a *future* change ever
        # adds any form of persisted/restored weight state, a mismatch
        # between that state and the currently-loaded graph fails loudly
        # here instead of silently mutating the wrong synapses.
        if len(self.plastic_pos):
            assert np.array_equal(self.W.indices[self.plastic_pos], self.plastic_pre_idx), (
                "plasticity integrity check failed: plastic_pos does not point to "
                "plastic_pre_idx's columns in W — refusing to continue, since silently "
                "mutating the wrong synapses is exactly the bug this check exists to catch."
            )
            assert self.is_kc[self.plastic_pre_idx].all() and self.is_mbon[self.plastic_post_idx].all(), (
                "plasticity integrity check failed: plastic subset contains a non-KC "
                "presynaptic or non-MBON postsynaptic neuron."
            )

        # A fingerprint of the graph this plastic subset was built against —
        # not consulted anywhere yet (nothing persists weight state across
        # restarts today), but here so that the day something *does* save/
        # restore learned weights, it has an immediate, real way to verify
        # "is this checkpoint for the graph I just loaded" before touching
        # W.data, rather than needing to invent one under pressure then.
        self.graph_fingerprint = {
            "node_count": self.n,
            "edge_count": int(self.W.nnz),
            "kc_count": int(self.is_kc.sum()),
            "mbon_count": int(self.is_mbon.sum()),
            "plastic_count": int(len(self.plastic_pos)),
            "body_id_hash": _hash_body_ids(self.body_ids),
        }

        log.info("plasticity: %d real KC->MBON synapses made plastic (of %d W nonzeros total)",
                  len(plastic_pos), self.W.nnz)

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

        # 1.5) Three-factor plasticity on the real KC->MBON subset only
        #      (never the full ~246k-edge W — see _build_plasticity). Two
        #      factors happen every sub-step regardless of the learning
        #      toggle (eligibility is a physical trace of recent Hebbian
        #      coincidence, not a "training mode" switch): decay the
        #      existing trace, then bump it wherever a KC and its MBON
        #      partner spiked in the *same* sub-step (direct coincidence
        #      detector — a standard simplification of the eligibility term
        #      in three-factor rules; see Fremaux & Gerstner 2016). The
        #      third factor (self.dopamine_rate, the signed PAM-minus-PPL
        #      EMA computed once per outer step()) only actually moves
        #      weights when learning is enabled.
        if len(self.plastic_pos):
            self.eligibility *= self.elig_decay_per_substep
            coincident = self.spikes[self.plastic_pre_idx] & self.spikes[self.plastic_post_idx]
            self.eligibility[coincident] += 1.0
            if self.learning_enabled and self.dopamine_rate != 0.0:
                # delta >= 0 in the direction of dopamine_rate's sign;
                # applying it *along the synapse's own sign* (sign0) means
                # reward (dopamine_rate>0) always potentiates (pushes |w|
                # up) and punishment (dopamine_rate<0) always depresses
                # (pushes |w| down) — for both excitatory and inhibitory
                # synapses alike, since "stronger" means "further from
                # zero in its own direction" either way.
                delta = self.plasticity_eta * self.eligibility * self.dopamine_rate * dt
                new_raw = self.W.data[self.plastic_pos] + self._w_sign_plastic * delta
                new_mag = np.clip(np.abs(new_raw), 0.1 * self._w_mag0_plastic, 3.0 * self._w_mag0_plastic)
                self.W.data[self.plastic_pos] = self._w_sign_plastic * new_mag

        # 2) Delayed synaptic input: the spikes fired T_DLY ms ago arrive now.
        delayed_spikes = self._delay_buf[self._delay_ptr]
        self._delay_buf[self._delay_ptr] = self.spikes.astype(np.float64)
        self._delay_ptr = (self._delay_ptr + 1) % self.delay_steps
        self.g[active] += (self.W @ delayed_spikes)[active]

        # 3) External Poisson drive: independent per-neuron spike process.
        #    Sensory (optic-lobe) neurons fire at a rate proportional to the
        #    live L/R eye brightness from the browser-rendered park scene —
        #    the actual compound-eye pathway, not a constant drive. Plus a
        #    small background rate on everything else (our addition, off by
        #    default) — both add directly to v, not g, per the reference's
        #    `target_var='v'`.
        if self.sensory_drive > 0 and self.is_sensory.any():
            rate_L = self.sensory_drive * self.R_POI * self.visual_L
            rate_R = self.sensory_drive * self.R_POI * self.visual_R
            rate_unsided = self.sensory_drive * self.R_POI * 0.5 * (self.visual_L + self.visual_R)
            fires = (
                (self.is_sensory_left & (self.rng.random(self.n) < rate_L * dt / 1000.0))
                | (self.is_sensory_right & (self.rng.random(self.n) < rate_R * dt / 1000.0))
                | (self.is_sensory_unsided & (self.rng.random(self.n) < rate_unsided * dt / 1000.0))
            )
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

        # Third factor: signed dopamine EMA, PAM (reward) minus PPL
        # (punishment) instantaneous population spike fraction — the real
        # reward signal driving the weight update in _substep. Computed
        # once per outer step (not per sub-step) since it's a slow-ish
        # behavioral-timescale signal, not something that needs sub-ms
        # resolution.
        pam_rate = union_spikes[self.is_pam].mean() if self.is_pam.any() else 0.0
        ppl_rate = union_spikes[self.is_ppl].mean() if self.is_ppl.any() else 0.0
        target = pam_rate - ppl_rate
        self.dopamine_rate += self.dopamine_rate_alpha * (target - self.dopamine_rate)
        self.cumulative_reward += self.dopamine_rate * self.n_sub * self.sub_dt / 1000.0

        return self.spikes

    def set_learning(self, enabled: bool):
        self.learning_enabled = bool(enabled)

    def weight_change_ratio(self) -> np.ndarray:
        """Per-plastic-synapse |current_weight| / |initial_weight| - 1, for
        the dashboard to color the plastic subset by how much each synapse
        has actually moved since startup (0 = unchanged)."""
        if len(self.plastic_pos) == 0:
            return np.array([])
        current_mag = np.abs(self.W.data[self.plastic_pos])
        return current_mag / np.maximum(self._w_mag0_plastic, 1e-9) - 1.0

    def plastic_node_weight_changes(self) -> dict:
        """Per-*neuron* (not per-synapse — 44k synapse-level values every
        broadcast tick would be tens of MB/s) mean |Δw/w0| across each KC's
        outgoing, or each MBON's incoming, plastic synapses. Compact enough
        (KC+MBON = 4,161 neurons here) to send periodically over the
        WebSocket for the dashboard to recolor the plastic subset's soma
        points by."""
        if len(self.plastic_pos) == 0:
            return {}
        change = np.abs(self.weight_change_ratio())
        out: dict[int, float] = {}
        for idx_array, label in ((self.plastic_pre_idx, "pre"), (self.plastic_post_idx, "post")):
            sums = np.bincount(idx_array, weights=change, minlength=self.n)
            counts = np.bincount(idx_array, minlength=self.n)
            nz = counts > 0
            for i in np.where(nz)[0]:
                out[int(self.body_ids[i])] = float(sums[i] / counts[i])
        return out

    def set_visual_input(self, left: float, right: float):
        """Live compound-eye input: normalized brightness (0..1) sampled
        from the browser-rendered park scene, one value per eye. Drives the
        optic-lobe sensory neurons' Poisson rate directly (see _substep)."""
        self.visual_L = float(np.clip(left, 0.0, 1.0))
        self.visual_R = float(np.clip(right, 0.0, 1.0))

    def flight_command(
        self, base_thrust: float = 0.5, speed_gain: float = 1.2, yaw_gain: float = 4.0,
        thrust_range: tuple[float, float] = (0.0, 1.0), yaw_range: tuple[float, float] = (-2.0, 2.0),
    ) -> tuple[float, float]:
        """Translate DN spike-rate readout directly into a (thrust, yaw_rate)
        flight command — no training involved. Same L/R-DN-rate readout as
        the walking-controller version this replaced, just interpreted as
        flight kinematics instead of a CPG drive: overall DN activity ->
        forward thrust, left/right DN imbalance -> yaw."""
        overall = self.dn_rate[self.is_dn].mean() if self.is_dn.any() else 0.0
        left_rate = self.dn_rate[self.dn_left].mean() if self.dn_left.any() else overall
        right_rate = self.dn_rate[self.dn_right].mean() if self.dn_right.any() else overall

        thrust = np.clip(base_thrust + speed_gain * overall, *thrust_range)
        yaw = np.clip(yaw_gain * (left_rate - right_rate), *yaw_range)
        return float(thrust), float(yaw)

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
        # Eligibility is transient brain state (a ~2s trace, same category
        # as V/g/spikes) and gets cleared here. Learned *weights* and
        # cumulative_reward are deliberately left untouched — "reset
        # flight" clears the fly's pose and neural activity, not what the
        # mushroom body has learned so far.
        self.eligibility[:] = 0.0
        self.dopamine_rate = 0.0


# --------------------------------------------------------------------------
# Bridge: BrainLIF <-> browser-side kinematic flight (no physics engine)
# --------------------------------------------------------------------------

@dataclass
class BrainSnapshot:
    t: float
    thrust: float
    yaw_rate: float
    spiking_ids: list = field(default_factory=list)
    reward_signal: float = 0.0
    cumulative_reward: float = 0.0
    learning_enabled: bool = False


class VisionFlightBridge:
    """Owns the connectome brain only. The fly's body and the park it flies
    through are a lightweight kinematic model in the browser
    (dashboard.jsx) — there is no physics engine here, and therefore no
    physics-timestep bottleneck: BrainLIF.step() alone runs in the
    thousands of steps/sec (see the CLI benchmark below), nowhere near a
    constraint the way MuJoCo's contact solver was."""

    def __init__(self, graph_path: str | Path, brain_dt: float = 2e-3, seed: int = 0):
        self.brain = BrainLIF(graph_path, dt=brain_dt, seed=seed)
        self.brain_dt = brain_dt
        self._t = 0.0

    def set_visual_input(self, left: float, right: float):
        self.brain.set_visual_input(left, right)

    def set_learning(self, enabled: bool):
        self.brain.set_learning(enabled)

    def reset(self):
        self.brain.reset_state()
        self._t = 0.0

    def step(self) -> BrainSnapshot:
        self.brain.step()
        thrust, yaw = self.brain.flight_command()
        self._t += self.brain_dt
        return BrainSnapshot(
            t=self._t, thrust=thrust, yaw_rate=yaw,
            spiking_ids=self.brain.spiking_body_ids(),
            reward_signal=self.brain.dopamine_rate,
            cumulative_reward=self.brain.cumulative_reward,
            learning_enabled=self.brain.learning_enabled,
        )


# --------------------------------------------------------------------------
# CLI: standalone smoke-test / benchmark
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", default="data/brain_graph.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bridge = VisionFlightBridge(args.graph, seed=args.seed)
    log.info("Running %d brain steps (no physics engine involved)...", args.steps)

    t0 = time.time()
    total_spikes = 0
    for i in range(args.steps):
        # Feed a slowly-varying synthetic L/R visual signal so the smoke
        # test actually exercises set_visual_input() the way the browser will.
        left = 0.5 + 0.4 * np.sin(i * 0.01)
        right = 0.5 + 0.4 * np.sin(i * 0.01 + 1.0)
        bridge.set_visual_input(left, right)
        snap = bridge.step()
        total_spikes += len(snap.spiking_ids)
        if i % 500 == 0:
            log.info("t=%.3fs thrust=%.2f yaw=%.2f spiking=%d",
                      snap.t, snap.thrust, snap.yaw_rate, len(snap.spiking_ids))

    elapsed = time.time() - t0
    log.info("Done: %d steps in %.2fs (%.0f steps/s), %d total spikes",
              args.steps, elapsed, args.steps / elapsed, total_spikes)


if __name__ == "__main__":
    main()
