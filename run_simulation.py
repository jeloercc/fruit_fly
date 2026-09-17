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
        # Was 0.15. With sustained dopamine that drove every plastic synapse
        # in both compartments into the -90% clip floor within ~240 steps
        # (measured: -85.0% / -85.9%), which is saturation, not learning —
        # a saturated synapse cannot encode anything. Baseline subtraction
        # (see pam_baseline/ppl_baseline) already cuts the effective
        # dopamine term several-fold. A first attempt at 0.05 was still
        # far too high — 19,283 of 21,573 PAM-compartment synapses still
        # hit the -90% clip floor. 0.005 also saturated (-87%) once the
        # dopamine EMAs were properly warmed to steady state, which raises
        # the effective drive ~3.4x (d_PAM 0.067 -> 0.23) relative to the
        # earlier cold-start tests. Sized arithmetically instead of by
        # guessing: cumulative depression over one 240-step episode is
        # ~eta * elig * d_dopa * sub_dt * 960 substeps; with elig~50 and
        # d_dopa~0.22 that gives ~26x the weight at 0.005 (hence the floor)
        # and ~0.26 — a ~25% depression, the intended non-saturating range —
        # at 5e-5.
        plasticity_eta: float = 5e-5,
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
        # Was 0.5s. At the tunnel's 0.3s appetitive-burst cadence (see
        # dashboard.jsx TunnelRig), a 0.5s EMA mostly decayed back toward
        # baseline between pulses before the next one landed — sustained
        # full-PAM stimulation over 5s measured live only reached -0.156
        # from a -0.20 baseline (see AGENTS.md-style reasoning in the
        # dopamine_baseline comment below). A longer time constant lets
        # bursts arriving faster than it decays actually compound instead
        # of mostly resetting each cycle.
        dopamine_rate_tau: float = 1.5,
        # First-order muscle-activation (calcium-dynamics-style) filter
        # time constant for the VNC efferent pathway — see
        # motor_activation()/step() below. Not neuPrint-derived (there's no
        # calcium-imaging data in this connectome), a literature-typical
        # insect-muscle activation timescale used as a first-pass constant,
        # same status as DN_RATE_TO_DRIVE_GAIN-style gains elsewhere.
        muscle_tau_s: float = 0.05,
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
        self.dt = dt  # outer timestep (s) — needed to express rates in Hz
        self.dn_rate_alpha = dt / dn_rate_tau
        self.rng = np.random.default_rng(seed)

        self.learning_enabled = plasticity_enabled
        self.plasticity_eta = plasticity_eta
        self.elig_decay_per_substep = np.exp(-self.SUB_DT / tau_elig_ms)
        self.dopamine_rate_alpha = dt / dopamine_rate_tau
        self.dopamine_rate = 0.0       # signed EMA: PAM(reward) - PPL(punishment)
        # Empirically measured resting-state value of dopamine_rate with
        # zero stimulation, purely from ambient noise/sensory_drive (-0.20
        # to -0.21 across repeated live measurements). This is a real
        # network property, not fabricated — but it's a normalization
        # artifact, not evidence that PPL is "more excitable": PPL is only
        # 24 neurons vs PAM's 316, so the same handful of noise-driven
        # spikes produces a far larger *fractional* population rate for
        # PPL than for PAM (mean-fraction over a small N is a
        # high-variance, upward-biased estimator). dopamine_rate itself
        # (used for plasticity below, real biological signal) is left
        # exactly as-is; this constant only recenters what gets *reported*
        # as reward — see reward_signal/cumulative_reward in step().
        self.dopamine_baseline = -0.20
        self.cumulative_reward = 0.0
        # Unsigned per-compartment dopamine levels — the third factor for
        # the compartment-specific depression rule (see step()/_substep).
        self.pam_level = 0.0
        self.ppl_level = 0.0
        # Empirically measured resting levels of the two EMAs above, with
        # zero stimulation and learning frozen (400 steps, mean of the last
        # 250): PAM 0.04632 +- 0.01033, PPL 0.10821 +- 0.02400. The 2.34x
        # ratio is the small-N artifact, not higher PPL excitability — PPL
        # is 24 cells vs PAM's 316, so identical noise yields a much larger
        # population fraction. Subtracted in _substep so an unstimulated
        # network produces exactly zero weight change (dopamine
        # reuptake/clearance, functionally).
        self.pam_baseline = 0.046
        self.ppl_baseline = 0.108
        # SYNTHETIC diagnostic toggle — see the clamp block in _substep.
        # Default OFF: the honest model lets the two DAN populations
        # cross-talk, because that is what the connectome does.
        self.dan_clamp = False
        # By-intention clamp state. When dan_clamp is ON, the valence the
        # USER most recently injected (not the measured EMAs) decides which
        # compartment may learn. Latched for dan_intent_window outer steps,
        # sized to the dopamine EMA's own time constant so the window covers
        # the period over which that volley's dopamine is actually elevated.
        self.dan_intent = None          # 'PAM' | 'PPL' | None
        self.dan_intent_left = 0        # outer steps remaining
        self.dan_intent_window = int(round(dopamine_rate_tau / dt))

        # VNC efferent pathway: real leg motor-neuron output, filtered
        # through a muscle-activation leaky integrator (see step()). Kept
        # as brain state (not physics_worker.py's) since it's the SNN's own
        # readout being smoothed, same category as dn_rate/dopamine_rate.
        self.muscle_alpha = dt / muscle_tau_s
        self.muscle_activation_left = 0.0
        self.muscle_activation_right = 0.0

        # Stamina/fatigue effect (frontend-driven, see dashboard.jsx
        # FlightRig): an additive mV offset applied only to DN (motor
        # output) neurons' spike threshold, not the model's global V_TH.
        # Real neuromuscular fatigue reduces motor-neuron excitability
        # without touching sensory/interneuron thresholds — this is that
        # same scoped effect, applied here rather than as a fake
        # frontend-only speed cap so a fatigued fly's DNs measurably fire
        # less, not just get their output clamped downstream.
        self.motor_threshold_boost = 0.0

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
        # Written via a temp file + atomic rename, not in place: the
        # frontend's static-file GET for this exact path can land while a
        # fresh backend is still mid-json.dump() (a 37MB write isn't
        # instantaneous), and StaticFiles computes Content-Length from an
        # early stat() of the file — if the file keeps growing under it
        # while streaming, uvicorn raises "Response content longer than
        # Content-Length". Path.replace() is an atomic rename on POSIX, so
        # a concurrent reader only ever sees the old complete file or the
        # new complete file, never a partially-written one.
        nodes_sidecar_path = Path(graph_path).with_name(Path(graph_path).stem + "_nodes.json")
        tmp_sidecar_path = nodes_sidecar_path.with_suffix(".json.tmp")
        with open(tmp_sidecar_path, "w") as f:
            json.dump({"meta": graph.get("meta", {}), "nodes": nodes}, f)
        tmp_sidecar_path.replace(nodes_sidecar_path)
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

        # VNC efferent pathway: real leg motor neurons (superclass
        # 'vnc_motor', 708 neurons — genuine neuPrint population, types
        # like "Ti flexor MN"/"Fe reductor MN"/"Ta levator MN"/etc., named
        # after the real muscle/joint they act on). male-cns:v1.0 has no
        # per-leg (T1/T2/T3, i.e. fore/mid/hind) identity field on these —
        # only soma_side — so L/R pooled firing rate is the finest honest
        # split available; see motor_activation()/step() for the filter
        # that turns this into the physics process's actual drive signal.
        # Defined once here — used by the DN/photoreceptor/sensory-channel
        # masks throughout the rest of __init__.
        type_str = np.array([(n.get("type") or "") for n in nodes])

        is_vnc_motor = superclass == "vnc_motor"
        self.is_vnc_motor = is_vnc_motor
        self.mn_left = is_vnc_motor & (soma_side == "L")
        self.mn_right = is_vnc_motor & (soma_side == "R")

        # Named locomotor DNs from the literature, confirmed present in
        # this dataset by direct query (counts are small because these are
        # individually-identified cell types, not populations): DNb01 (2)
        # and DNb02 (4) are the forward-drive pair; DNp09 (2) is the
        # stopping/freezing DN. All three verified to synapse onto real
        # vnc_motor neurons here (651 / 337 / 77 synapses respectively), so
        # they are genuinely part of the efferent path, not just present.
        # With only 2-4 cells each their population rate is a very
        # high-variance estimator (the same small-N problem documented for
        # PPL in dopamine_baseline above) — reported for observability,
        # NOT used as the motor drive signal. See dn_premotor below.
        self.dn_named = {
            "DNb01": self.is_dn & (type_str == "DNb01"),
            "DNb02": self.is_dn & (type_str == "DNb02"),
            "DNp09": self.is_dn & (type_str == "DNp09"),
        }

        # Vision-first pivot: sensory neurons are the optic-lobe input layer
        # (ol_sensory — real neuPrint superclass for the compound-eye-adjacent
        # photoreceptor/lamina neurons), split L/R by soma side since that's
        # the real anatomical eye split available in the connectome metadata
        # — we don't have per-ommatidium retinotopic coordinates, so this is
        # the coarsest honest mapping: left-eye brightness drives left-soma
        # sensory neurons, right-eye brightness drives right-soma ones.
        # Eye side comes from `instance`, NOT `soma_side`. Measured on this
        # dataset: of the 6,098 ol_sensory photoreceptors, soma_side is
        # null for 6,062 of them (only 23 L / 13 R annotated) — so keying
        # the visual split on soma_side silently routed 99.4% of the
        # retina into the "unsided" bucket, where it received the MEAN of
        # both eyes and could never encode a left/right difference at all.
        # `instance` carries the real side as a suffix ("R1-R6_L",
        # "R7y_R", ...) and covers 100% of them: L=2,349, R=3,749, 0
        # unsided. Same suffix convention is used across the other sensory
        # superclasses, so this is applied to is_sensory as a whole.
        instance = np.array([n.get("instance") or "" for n in nodes])
        side_L = np.char.endswith(instance, "_L") | (soma_side == "L")
        side_R = np.char.endswith(instance, "_R") | (soma_side == "R")
        self.is_sensory_left = self.is_sensory & side_L
        self.is_sensory_right = self.is_sensory & side_R
        self.is_sensory_unsided = self.is_sensory & ~side_L & ~side_R
        self.visual_L = 0.5  # normalized brightness (0..1), set live by set_visual_input()
        self.visual_R = 0.5

        # Real photoreceptor classes, straight off the connectome's own
        # `type` field within ol_sensory (counts measured on this dataset):
        #   R1-R6          3,377 — broadband outer photoreceptors
        #                          (motion/luminance)
        #   pale R7/R8       662 — one spectral subtype
        #   yellow R7/R8     963 — the other spectral subtype
        #   unclear R7/R8  1,096 — annotated R7/R8 but subtype not resolved
        # flygym's retina reports the SAME pale/yellow distinction via its
        # own pale_type_mask (216 pale / 505 yellow of 721 ommatidia), so
        # the visual afference below is class-matched on both ends rather
        # than collapsed to a single brightness scalar. Per-ommatidium
        # retinotopy is NOT claimed: male-cns:v1.0 gives these neurons no
        # coordinates (soma is [None,None,None] for all of them), so eye
        # side + spectral class is the finest honest split available.
        type_ol = np.array([(n.get("type") or "") for n in nodes])
        is_ol_sensory = superclass == "ol_sensory"
        is_r16 = is_ol_sensory & np.char.startswith(type_ol, "R1-R6")
        is_r78 = is_ol_sensory & (np.char.startswith(type_ol, "R7") | np.char.startswith(type_ol, "R8"))
        # Subtype suffix: "R7p"/"R8p" = pale, "R7y"/"R8y" = yellow. The
        # "_unclear" ones match neither and are deliberately left out of
        # both spectral groups rather than guessed into one.
        is_pale = is_r78 & np.char.endswith(type_ol, "p")
        is_yellow = is_r78 & np.char.endswith(type_ol, "y")
        self.photoreceptor_masks = {
            "broadband_L": is_r16 & side_L,
            "broadband_R": is_r16 & side_R,
            "pale_L": is_pale & side_L,
            "pale_R": is_pale & side_R,
            "yellow_L": is_yellow & side_L,
            "yellow_R": is_yellow & side_R,
        }
        # Per-class drive levels (0..1 luminance), written by
        # set_visual_afference() and injected as Poisson current in
        # _substep — the real retinal input path.
        self.visual_afference = {k: 0.0 for k in self.photoreceptor_masks}

        # Neural Sandbox: manually-controllable sensory channels, each a real
        # neuPrint population (not invented categories) identified by `type`
        # within the existing sensory superclasses:
        #   optic_L/R    — ol_sensory (compound-eye photoreceptors), same
        #                  population set_visual_input already drives, split
        #                  by soma side, exposed here for manual override too
        #   antennal     — cb_sensory neurons typed "JO-*": real Johnston's
        #                  Organ units, the fly's actual antennal
        #                  mechanoreceptor/near-field-sound organ
        #   olfactory    — cb_sensory neurons typed "ORN": real Olfactory
        #                  Receptor Neurons (antennal odor input)
        #   leg_body     — the whole vnc_sensory superclass: real leg/body
        #                  touch and proprioceptive afferents entering via
        #                  the ventral nerve cord
        # group_drive holds a continuous Poisson-rate multiplier per channel
        # (0 = off), applied additively in _substep — independent of
        # sensory_drive/visual_L/R, which remain the camera-driven pathway.
        is_ol = superclass == "ol_sensory"
        # tarsal_contact/chordotonal: real sub-populations within
        # vnc_sensory, identified by their actual neuPrint `type` prefix —
        # SNta* (2,573 neurons — "sensory neuron, tarsal", genuine
        # touch/ground-contact afferents) and SNch* (124 neurons — genuine
        # chordotonal-organ joint-stretch proprioceptors). Distinct from the
        # existing "leg_body" manual-slider channel (the whole vnc_sensory
        # superclass) so real physical telemetry from physics_worker.py
        # (see telemetry_server.py's SimWorker._run) never fights with the
        # user's own manual sensory-injection slider on the same group.
        is_vnc_sensory = superclass == "vnc_sensory"
        self._group_masks = {
            # Eye side from `instance`, not soma_side — see the eye-split
            # note above (soma_side is null for 99.4% of photoreceptors).
            "optic_L": is_ol & side_L,
            "optic_R": is_ol & side_R,
            "antennal": np.char.startswith(type_str, "JO"),
            # Real ORNs carry their target antennal-lobe glomerulus in the
            # type ("ORN_DA1", "ORN_VA1d", ... — 2,635 of them). NO neuron
            # in male-cns:v1.0 is typed bare "ORN", so the previous exact
            # match `type_str == "ORN"` selected an EMPTY set and this
            # channel silently injected nothing at all.
            "olfactory": np.char.startswith(type_str, "ORN"),
            "leg_body": is_vnc_sensory,
            "tarsal_contact": is_vnc_sensory & np.char.startswith(type_str, "SNta"),
            "chordotonal": is_vnc_sensory & np.char.startswith(type_str, "SNch"),
        }

        # ---- Valence-specific, lateralized stimulus channels -------------
        # All 53 real antennal-lobe glomeruli are present in this dataset as
        # ORN_<glomerulus> types, so a stimulus can be aimed at a specific
        # olfactory channel instead of the whole receptor population.
        #
        # FOOD (attractive): the canonical attractive food-odor glomeruli —
        #   DM1 (Or42b, apple cider vinegar), DM2 (Or22a), DM4 (Or59b),
        #   VA2 (Or92a), VM2 (Or43b).
        #   NOTE: VA1v is deliberately NOT included here. VA1v is the Or47b
        #   channel, a PHEROMONE/courtship glomerulus, not a food-odor one —
        #   including it would mislabel a mating-circuit stimulus as feeding.
        # AVERSIVE: V is the real CO2 glomerulus (Gr21a/Gr63a); DA2 is the
        #   geosmin channel (Or56a, microbial-danger avoidance); DL5 (Or7a).
        #
        # WIND/FREEZE: Johnston's Organ subtypes C/D/E — the static-
        #   deflection units that encode wind and gravity — while A/B (sound
        #   /vibration, 138 cells here) are excluded. JO afferents ARE the
        #   AMMC input: this graph carries no neuropil field, so AMMC is
        #   addressed through its afferent population, not a region label.
        food_glom = ("ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VA2", "ORN_VM2")
        averse_glom = ("ORN_V", "ORN_DA2", "ORN_DL5")
        is_food = np.isin(type_str, food_glom)
        is_averse = np.isin(type_str, averse_glom)
        is_wind = np.char.startswith(type_str, "JO-C") | np.char.startswith(type_str, "JO-D") \
            | np.char.startswith(type_str, "JO-E")
        # Lateralized so asymmetric injection can produce a real left/right
        # motor difference. Neurons whose side is unannotated are excluded
        # from both sides rather than duplicated into each (food 15/284,
        # aversive 8/146, wind 0/343 are unsided here).
        for name, mask in (("food", is_food), ("co2", is_averse), ("wind", is_wind)):
            self._group_masks[f"{name}_L"] = mask & side_L
            self._group_masks[f"{name}_R"] = mask & side_R
        self.group_drive = {name: 0.0 for name in self._group_masks}

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

        # Premotor DN population, derived from REAL connectivity rather
        # than from a name list: every DN with at least one direct synapse
        # onto a vnc_motor neuron. W is [post, pre], so the presynaptic
        # partners of the motor rows are exactly the column indices present
        # in those rows. Measured on this dataset: 981 of the 1,316 DNs
        # qualify — a population large enough to give a stable rate, unlike
        # the 2-4 cell named types in dn_named. This is the "efferent
        # bottleneck" the motor pathway actually flows through.
        motor_rows = self.W[self.is_vnc_motor]
        premotor_pre = np.unique(motor_rows.indices)
        self.dn_premotor = np.zeros(self.n, dtype=bool)
        self.dn_premotor[premotor_pre] = True
        self.dn_premotor &= self.is_dn
        log.info("BrainLIF: premotor DNs (direct DN->vnc_motor synapse): %d of %d DNs",
                 self.dn_premotor.sum(), self.is_dn.sum())

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

        # ---- Compartment assignment, derived from real DAN->MBON wiring --
        # In Drosophila the mushroom body is organised into compartments:
        # each dopaminergic population innervates a specific set of MBONs,
        # and DAN activity DEPRESSES the KC->MBON synapses in *its own*
        # compartment only. That is what makes valence learning specific
        # rather than global.
        #
        # Rather than assume which MBONs belong to which compartment, this
        # reads it out of the connectome: total synaptic input onto each
        # MBON from PAM vs PPL neurons, requiring a 2x dominance margin so
        # ambiguous MBONs are assigned to neither. Measured on this dataset:
        # 38 PAM-dominant, 57 PPL-dominant of 97 MBONs (PAM contacts 49
        # distinct MBONs with 27,090 synapses; PPL contacts 85 with 10,882).
        Wabs = abs(self.W)
        pam_in = np.asarray(Wabs[:, self.is_pam].sum(axis=1)).ravel()
        ppl_in = np.asarray(Wabs[:, self.is_ppl].sum(axis=1)).ravel()
        mbon_pam_dom = self.is_mbon & (pam_in > 2.0 * ppl_in)
        mbon_ppl_dom = self.is_mbon & (ppl_in > 2.0 * pam_in)
        # Per-plastic-synapse compartment membership, via its postsynaptic MBON.
        self.plastic_in_pam_comp = mbon_pam_dom[self.plastic_post_idx]
        self.plastic_in_ppl_comp = mbon_ppl_dom[self.plastic_post_idx]
        log.info("plasticity: MB compartments from real DAN->MBON wiring — "
                 "PAM-dominant MBONs=%d PPL-dominant=%d | plastic synapses: "
                 "PAM-comp=%d PPL-comp=%d unassigned=%d",
                 int(mbon_pam_dom.sum()), int(mbon_ppl_dom.sum()),
                 int(self.plastic_in_pam_comp.sum()), int(self.plastic_in_ppl_comp.sum()),
                 int(len(self.plastic_pos) - self.plastic_in_pam_comp.sum()
                     - self.plastic_in_ppl_comp.sum()))

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
        if self.motor_threshold_boost != 0.0:
            # Only DNs pay the fatigue cost — sensory/interneuron thresholds
            # are untouched. np.where allocates a temp array; skipped
            # entirely (falls through to the plain scalar compare) whenever
            # stamina is full, which is the common case.
            eff_th = np.where(self.is_dn, self.V_TH + self.motor_threshold_boost, self.V_TH)
            self.spikes = active & (self.V >= eff_th)
        else:
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
            if self.learning_enabled and (self.pam_level > 0.0 or self.ppl_level > 0.0):
                # DOPAMINE-INDUCED DEPRESSION, compartment-specific.
                #
                # This replaces an earlier rule that potentiated on reward
                # and depressed on punishment. That had the sign backwards:
                # in Drosophila, pairing an odour with DAN activity
                # DEPRESSES the KC->MBON synapses carrying that odour in the
                # innervated compartment (Hige et al. 2015; Cohn et al.
                # 2015; Owald & Waddell 2015). Learning works by *removing*
                # one MBON's vote rather than by strengthening it: an MBON
                # driving avoidance goes quiet for the rewarded odour, so
                # the opposing pathway wins and behaviour shifts.
                #
                # It is also compartment-specific, not global: PAM activity
                # depresses only PAM-compartment synapses and PPL only
                # PPL-compartment ones (membership read from real DAN->MBON
                # wiring in _build_plasticity). Using the SIGNED
                # PAM-minus-PPL difference here would be wrong twice over —
                # it lets the two populations cancel, and it applies one
                # valence's dopamine to the other's synapses.
                # Noise-floor subtraction per population, the functional
                # analogue of dopamine reuptake/clearance: trace basal
                # dopamine must not drive continuous depression.
                #
                # This is NOT cosmetic. PPL has 24 cells against PAM's 316,
                # so the same handful of noise-driven spikes yields a far
                # larger population FRACTION for PPL (the same small-N
                # artifact documented at dopamine_baseline). The earlier
                # signed PAM-minus-PPL form accidentally hid this; moving to
                # unsigned per-compartment levels exposed it, and without
                # these baselines the PPL compartment depressed continuously
                # at rest — measured ppl_level ~0.10 with zero punishment.
                # Subtracting each population's own resting level restores
                # dW = 0 for an unstimulated network.
                d_pam = max(0.0, self.pam_level - self.pam_baseline)
                d_ppl = max(0.0, self.ppl_level - self.ppl_baseline)
                if self.dan_clamp and self.dan_intent is not None:
                    # SYNTHETIC — NOT BIOLOGY. Winner-take-all mutual
                    # inhibition between the two DAN populations, so only
                    # the more strongly driven valence writes to its
                    # compartment. This has NO structural correlate in
                    # male-cns:v1.0: measured directly, PAM->PPL is 192
                    # synapses and PPL->PAM is 453, and ALL 645 of them are
                    # EXCITATORY — zero inhibitory. The real populations
                    # drag each other up (D_PAM/D_PPL ratio 1.09 when PAM
                    # alone is stimulated), which is why unclamped learning
                    # is generalised rather than associative. Off by
                    # default; exposed in the UI purely as a diagnostic for
                    # observing isolated three-factor STDP dynamics.
                    # BY INTENTION, not by measurement. An earlier version
                    # compared the two measured EMAs winner-take-all, but
                    # the crosstalk equalises them (d_PAM 0.2264 vs d_PPL
                    # 0.2284, a 0.9% gap), so the winner flipped randomly
                    # per sub-step and the net specificity was +0.50pp —
                    # nothing. Keying off the valence the user actually
                    # injected is the only version that discriminates. This
                    # is MORE synthetic, not less: it ignores network state
                    # entirely and obeys the button.
                    if self.dan_intent == "PAM":
                        d_ppl = 0.0
                    else:
                        d_pam = 0.0
                dopa = (self.plastic_in_pam_comp * d_pam
                        + self.plastic_in_ppl_comp * d_ppl)
                # Negative delta => magnitude shrinks (see the sign0 factor
                # below), i.e. depression, for excitatory and inhibitory
                # synapses alike.
                delta = -self.plasticity_eta * self.eligibility * dopa * dt
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

        # Real retinal afference: per-photoreceptor-class Poisson current
        # from the live MuJoCo compound-eye render (physics_worker.py ->
        # set_visual_afference). Same injection mechanism as every other
        # sensory pathway here — additive to v, not g — but keyed to the
        # actual R1-R6 / pale-R7R8 / yellow-R7R8 populations per eye
        # instead of one global brightness scalar.
        for _cls, _lum in self.visual_afference.items():
            if _lum <= 0.0:
                continue
            _mask = self.photoreceptor_masks[_cls]
            _fires = _mask & (self.rng.random(self.n) < self.sensory_drive * self.R_POI * _lum * dt / 1000.0)
            self.V[_fires] += self.W_SYN * self.F_POI

        # Neural Sandbox manual channel drive — independent of the camera
        # pathway above, additive continuous Poisson current per channel.
        for name, rate in self.group_drive.items():
            if rate <= 0.0:
                continue
            mask = self._group_masks[name]
            fires_group = mask & (self.rng.random(self.n) < rate * self.R_POI * dt / 1000.0)
            self.V[fires_group] += self.W_SYN * self.F_POI

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
        # Per-compartment UNSIGNED dopamine level. The signed difference
        # above is the right thing to REPORT as reward, but it is the wrong
        # thing to learn from: PAM and PPL innervate different compartments,
        # so each depresses its own synapses independently and they must not
        # cancel. These two EMAs are the third factor actually used by the
        # plasticity rule in _substep.
        if self.dan_intent_left > 0:
            self.dan_intent_left -= 1
            if self.dan_intent_left == 0:
                self.dan_intent = None
        self.pam_level += self.dopamine_rate_alpha * (pam_rate - self.pam_level)
        self.ppl_level += self.dopamine_rate_alpha * (ppl_rate - self.ppl_level)
        # cumulative_reward integrates deviation-from-rest, not the raw
        # signal — "reward" in the RL sense this dashboard is telling a
        # story about means better-or-worse-than-doing-nothing, and the
        # raw signal's resting value isn't zero (see dopamine_baseline
        # above). Plasticity below still uses raw self.dopamine_rate
        # unchanged — real biological weight updates aren't recentered.
        self.cumulative_reward += (self.dopamine_rate - self.dopamine_baseline) * self.n_sub * self.sub_dt / 1000.0

        # VNC efferent pathway: real vnc_motor L/R pooled firing rate (dn_rate
        # is a whole-population EMA array despite its name — see __init__ —
        # so indexing it with mn_left/mn_right reuses that same smoothing
        # unchanged) through a SECOND, distinct low-pass stage modeling
        # muscle calcium-activation dynamics (tau*da/dt = -a+u) — this is
        # what set_forward_drive actually sends to the physics process, not
        # the raw dn_rate readout.
        mn_left_rate = self.dn_rate[self.mn_left].mean() if self.mn_left.any() else 0.0
        mn_right_rate = self.dn_rate[self.mn_right].mean() if self.mn_right.any() else 0.0
        self.muscle_activation_left += self.muscle_alpha * (mn_left_rate - self.muscle_activation_left)
        self.muscle_activation_right += self.muscle_alpha * (mn_right_rate - self.muscle_activation_right)

        return self.spikes

    def dn_readout(self) -> dict:
        """Firing-rate readout of the efferent bottleneck: the broad
        connectivity-derived premotor DN population (stable, 981 cells)
        plus the individually-named literature locomotor DNs (DNb01/DNb02
        forward, DNp09 stop — 2-4 cells each, high variance, reported for
        observability only). dn_rate is the whole-network spike-rate EMA
        (see step()), so these are all the same smoothed measure.

        Values are returned in Hz. dn_rate itself is an EMA of a 0/1
        per-outer-step spike indicator (i.e. spikes per step, ~0..1), so
        dividing by the outer timestep converts it to spikes/second —
        1/0.002s = a 500x factor here. Reporting the raw indicator as "Hz"
        would understate the real firing rate by that same factor."""
        to_hz = 1.0 / self.dt
        out = {"premotor": float(self.dn_rate[self.dn_premotor].mean()) * to_hz if self.dn_premotor.any() else 0.0}
        for name, mask in self.dn_named.items():
            out[name] = float(self.dn_rate[mask].mean()) * to_hz if mask.any() else 0.0
        return out

    def motor_activation(self) -> tuple[float, float]:
        """The VNC efferent readout physics_worker.py's CPG is actually
        driven by — see the muscle-activation filter in step()."""
        return float(self.muscle_activation_left), float(self.muscle_activation_right)

    def set_learning(self, enabled: bool):
        self.learning_enabled = bool(enabled)

    def set_dan_clamp(self, enabled: bool):
        """Enable/disable the SYNTHETIC winner-take-all clamp between the
        PAM and PPL dopamine populations (see _substep). This is a
        diagnostic override with no structural correlate in the connectome
        — all 645 real PAM<->PPL synapses are excitatory. It exists so the
        isolated three-factor STDP dynamics can be observed; it does not
        make the model more biological."""
        self.dan_clamp = bool(enabled)

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

    def set_visual_afference(self, broadband: tuple[float, float], pale: tuple[float, float],
                              yellow: tuple[float, float]):
        """Live compound-eye input from the real MuJoCo retina render, one
        normalized luminance (0..1) per eye per photoreceptor class — see
        photoreceptor_masks in __init__ for how each maps onto real
        connectome populations. Each tuple is (left_eye, right_eye),
        matching obs["vision"]'s (2, 721, 2) eye-major layout."""
        v = self.visual_afference
        v["broadband_L"], v["broadband_R"] = float(np.clip(broadband[0], 0, 1)), float(np.clip(broadband[1], 0, 1))
        v["pale_L"], v["pale_R"] = float(np.clip(pale[0], 0, 1)), float(np.clip(pale[1], 0, 1))
        v["yellow_L"], v["yellow_R"] = float(np.clip(yellow[0], 0, 1)), float(np.clip(yellow[1], 0, 1))

    def set_group_drive(self, name: str, rate: float):
        """Neural Sandbox: continuous manual drive on one real sensory
        channel (see _group_masks), independent of the camera pathway.
        `rate` is a sensory_drive-style multiplier on R_POI, clamped to a
        sane range so a UI slider can't push a channel to a pathological
        firing rate."""
        if name not in self._group_masks:
            raise ValueError(f"unknown sensory group {name!r}; have {list(self._group_masks)}")
        self.group_drive[name] = float(np.clip(rate, 0.0, 5.0))

    def group_body_ids(self, name: str) -> list[int]:
        """bodyIds for one real sensory channel — for discrete force_spike()
        injection from the UI, same mechanism the PPL/PAM buttons use."""
        if name not in self._group_masks:
            raise ValueError(f"unknown sensory group {name!r}; have {list(self._group_masks)}")
        return self.body_ids[self._group_masks[name]].tolist()

    def flight_command(
        self, base_thrust: float = 0.5, speed_gain: float = 1.2, yaw_gain: float = 4.0,
        thrust_range: tuple[float, float] = (0.0, 1.0), yaw_range: tuple[float, float] = (-2.0, 2.0),
    ) -> tuple[float, float, float, float]:
        """Translate DN spike-rate readout directly into a (thrust, yaw_rate)
        flight command — no training involved. Same L/R-DN-rate readout as
        the walking-controller version this replaced, just interpreted as
        flight kinematics instead of a CPG drive: overall DN activity ->
        forward thrust, left/right DN imbalance -> yaw. Also returns the raw
        per-side rates (unclipped, un-gained) so callers can broadcast the
        actual DN readout the motor hook is built on, not just the derived
        thrust/yaw it collapses down to."""
        overall = self.dn_rate[self.is_dn].mean() if self.is_dn.any() else 0.0
        left_rate = self.dn_rate[self.dn_left].mean() if self.dn_left.any() else overall
        right_rate = self.dn_rate[self.dn_right].mean() if self.dn_right.any() else overall

        thrust = np.clip(base_thrust + speed_gain * overall, *thrust_range)
        yaw = np.clip(yaw_gain * (left_rate - right_rate), *yaw_range)
        return float(thrust), float(yaw), float(left_rate), float(right_rate)

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
            # Latch which valence the user just injected, for the
            # by-intention clamp (see _substep). Decided by which DAN
            # population the injected set actually overlaps — no separate
            # UI command needed, and it stays correct if the UI changes
            # which neurons it samples.
            sel = np.zeros(self.n, dtype=bool)
            sel[idx] = True
            n_pam = int((sel & self.is_pam).sum())
            n_ppl = int((sel & self.is_ppl).sum())
            if n_pam or n_ppl:
                self.dan_intent = "PAM" if n_pam >= n_ppl else "PPL"
                self.dan_intent_left = self.dan_intent_window
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
        self.dan_intent = None
        self.dan_intent_left = 0
        self.pam_level = 0.0
        self.ppl_level = 0.0
        self.muscle_activation_left = 0.0
        self.muscle_activation_right = 0.0


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
    dn_left_rate: float = 0.0
    dn_right_rate: float = 0.0
    mn_activation_left: float = 0.0
    mn_activation_right: float = 0.0
    dn_premotor_rate: float = 0.0
    dn_named_rates: dict = field(default_factory=dict)


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
        thrust, yaw, dn_left_rate, dn_right_rate = self.brain.flight_command()
        mn_activation_left, mn_activation_right = self.brain.motor_activation()
        dn_rates = self.brain.dn_readout()
        self._t += self.brain_dt
        return BrainSnapshot(
            t=self._t, thrust=thrust, yaw_rate=yaw,
            spiking_ids=self.brain.spiking_body_ids(),
            # Recentered on the resting baseline (see dopamine_baseline in
            # BrainLIF.__init__) so the dashboard reports reward relative to
            # doing-nothing, not the raw PAM-minus-PPL population fraction.
            reward_signal=self.brain.dopamine_rate - self.brain.dopamine_baseline,
            cumulative_reward=self.brain.cumulative_reward,
            learning_enabled=self.brain.learning_enabled,
            dn_left_rate=dn_left_rate, dn_right_rate=dn_right_rate,
            mn_activation_left=mn_activation_left, mn_activation_right=mn_activation_right,
            dn_premotor_rate=dn_rates.pop("premotor"), dn_named_rates=dn_rates,
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
