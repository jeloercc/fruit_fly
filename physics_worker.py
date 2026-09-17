#!/usr/bin/env python3
"""
physics_worker.py — live flygym/MuJoCo body simulation in its own OS
PROCESS (multiprocessing, not threading).

Why a process and not a thread: measured directly in this project — a
15-line reproduction (nothing but `import flygym` on a background
`threading.Thread`) triggers `*** Assertion failure in -[NSMenu
_setMenuName:]` and hangs the process indefinitely on this Mac. flygym/
dm_control's rendering-context setup touches Cocoa/AppKit at IMPORT TIME,
which macOS requires to happen on a process's actual main thread — a
background thread within the SAME process never qualifies, no matter how
little work it does. A separate `multiprocessing.Process` gets its own
real main thread, which sidesteps this entirely. `import flygym`/`import
mujoco` happen INSIDE physics_worker_process() (not at module level) so
they execute strictly after the child process's own main thread exists,
never in the parent's import graph or bootstrap sequence.

Still isolated from real-time: MuJoCo's own per-step cost benchmarked at
~3.9ms (0.03x real-time — see biomechanics_env.py). The brain process
keeps running at full speed regardless; this process produces a new pose
whenever it finishes its next step, which the broadcaster in
telemetry_server.py reads non-blockingly, repeating the last known pose
when nothing new has arrived yet.

IPC: two multiprocessing.Queue objects, not shared memory — drive_queue
(main -> physics, carries the latest efferent drive scalar: real VNC
motor-neuron activation, already folding in the real dopamine signal as an
arousal multiplier — see PhysicsBridge.set_forward_drive) and
telemetry_queue (physics -> main, carries {"pos": [...], "quat": [...],
"fatigue": float, "resting": bool, "contact_magnitude": float,
"joint_velocity_rms": float}). Both are drained to their LATEST item on
every read (see PhysicsBridge below) rather than consumed one-at-a-time,
since the two sides run at wildly different rates (the brain loop is far
faster than 3.9ms/step) and neither side should ever process a backlog of
stale values.

Efferent pathway (brain -> body): run_simulation.py's BrainLIF now pools
the real firing rate of the actual VNC leg motor neurons (superclass
'vnc_motor' — genuine neuPrint population, types like "Ti flexor MN"/"Fe
reductor MN"/etc, split by soma_side since male-cns:v1.0 has no per-leg
T1/T2/T3 identity to split further) and passes it through a first-order
muscle-activation filter (calcium-dynamics-style low-pass). That filtered
activation — not raw DN Hz — is what set_forward_drive turns into the
CPG's driving scalar here, dopamine-arousal-scaled as before.

Afferent pathway (body -> brain): this process also measures real
mechanosensory/proprioceptive signals off obs["contact_forces"] (tarsal
ground-contact force magnitude) and obs["joints"] (row 1 = joint angular
velocity, RMS'd) every step, sent back over telemetry_queue. The main
process feeds these into two more real, distinct neuPrint populations
within vnc_sensory: SNta* (tarsal sensory neurons — genuine touch/contact
afferents) for contact, and SNch* (chordotonal organs — genuine
joint-stretch proprioceptors) for joint velocity. See BrainLIF._group_masks
in run_simulation.py.

Fatigue: a body-side homeostatic scalar, NOT tied to any specific
connectome neuron population (there's no annotated "tiredness" class in
male-cns:v1.0 to hang it on honestly): it accumulates in the physics
process itself in proportion to how hard the legs are being driven, and
once it crosses FATIGUE_REST_THRESHOLD the process ignores incoming drive
entirely and forces idle stepping until fatigue decays back below
FATIGUE_RESUME_THRESHOLD. This models muscular/motor fatigue, deliberately
kept separate from the brain's own dopamine/VNC-motor circuitry.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue

log = logging.getLogger("physics_worker")

BASE_FREQ = 4.0    # Hz, idle tripod stepping frequency at effective_drive=0
FREQ_GAIN = 8.0     # Hz added per unit of effective_drive
# muscle_activation (from BrainLIF's leaky-integrator filter over real
# vnc_motor firing rate) sits in the same small-fraction range as any other
# EMA'd 0/1 spike readout here (~0.05-0.15 typical), not literal Hz — this
# gain is a first-pass scaling to land in the CPG's ~0-1.5 drive range, not
# a calibrated biomechanical constant.
MN_ACTIVATION_TO_DRIVE_GAIN = 12.0

# Dopamine (reward_signal, already recentered on resting baseline in
# run_simulation.py) scales the VNC-derived drive as an arousal term:
# reward_signal=0 (resting) -> arousal=1.0 (no change), positive/appetitive
# -> more vigorous stepping, negative/aversive -> more subdued. Clamped so
# neither a huge reward burst nor a hard aversive one can zero out or blow
# up the drive outright.
DOPAMINE_AROUSAL_GAIN = 3.0
MIN_AROUSAL = 0.15
MAX_AROUSAL = 2.5

# Fatigue: a body-side (not brain-side) homeostatic scalar, accumulated in
# physics_worker_process in proportion to effective drive. ~20s of
# sustained strong driving reaches FATIGUE_REST_THRESHOLD; forced rest then
# decays it back below FATIGUE_RESUME_THRESHOLD in ~4-5s.
FATIGUE_RATE = 0.0002
FATIGUE_RECOVERY_RATE = 0.0006
FATIGUE_REST_THRESHOLD = 1.0
FATIGUE_RESUME_THRESHOLD = 0.3


def physics_worker_process(drive_queue: mp.Queue, telemetry_queue: mp.Queue, stop_event):
    """Runs in the CHILD process, on ITS main thread. flygym/mujoco are
    imported here, not at module level, so their Cocoa/AppKit-touching
    setup only ever happens after this process's own main thread exists —
    never during the parent's import graph, never on a thread."""
    import numpy as np

    import flygym
    from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps

    log.info("physics_worker_process: initializing live flygym/MuJoCo in a dedicated "
              "OS process (real, ~0.03x-real-time engine, not baked)...")

    timestep = 1e-4
    phase_biases = np.pi * np.array([
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ])
    coupling_weights = (phase_biases > 0) * 10
    cpg = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * BASE_FREQ,
        intrinsic_amps=np.ones(6) * 1.0,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=np.ones(6) * 20,
    )
    steps = PreprogrammedSteps()
    # enable_vision=True turns on MuJoCo's two real compound-eye cameras.
    # Measured cost on this machine: 5.29ms -> 6.69ms per step (+26%,
    # 0.019x -> 0.015x real-time). Accepted deliberately — this is the
    # actual retinal input, not a proxy.
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position",
                     enable_vision=True)
    sim = flygym.SingleFlySimulation(fly=fly, timestep=timestep)  # flygym.NeuroMechFly doesn't exist in this version
    # Pose is reported for the FlyBody frame — NOT the thorax and NOT
    # obs["fly"][0] — because that is the frame dashboard.jsx's
    # fly_rigged.glb is rooted at (build_fly_anatomical_mesh.py walks the
    # body tree from FlyBody). Measured at rest: FlyBody sits at z=-0.20
    # while obs["fly"][0] reads z=1.07 and the thorax z=1.04, so feeding
    # either of the latter to a FlyBody-rooted mesh lifted the whole model
    # ~1.27 units — the fly visibly floating above the floor. FlyBody ->
    # Thorax is a rigid offset with an identity local quaternion (verified
    # numerically to 2e-16), so this single frame's pos+quat pins the
    # entire chassis exactly, and the feet land on the floor at z~0.02.
    root_id = next(b for b in fly.model.find_all("body") if b.name == "FlyBody").full_identifier

    # --- Retinotopic bookkeeping, computed once ---------------------------
    # fly.retina.ommatidia_id_map is a real (512,450) label image: each
    # pixel carries the 1-based id of the ommatidium that samples it (0 =
    # background). Averaging pixel coordinates per label gives each of the
    # 721 ommatidia its true position on the eye — which is what makes the
    # dashboard's L/R ommatidia grid a genuine spatial downsample of the
    # real retina rather than an arbitrary reshape.
    retina = fly.retina
    _id_map = retina.ommatidia_id_map
    _flat = _id_map.ravel()
    _valid = _flat > 0
    _idx = _flat[_valid] - 1
    _yy, _xx = np.mgrid[0:_id_map.shape[0], 0:_id_map.shape[1]]
    _counts = np.bincount(_idx, minlength=retina.num_ommatidia_per_eye).astype(float)
    _counts[_counts == 0] = 1.0
    _cy = np.bincount(_idx, weights=_yy.ravel()[_valid], minlength=retina.num_ommatidia_per_eye) / _counts
    _cx = np.bincount(_idx, weights=_xx.ravel()[_valid], minlength=retina.num_ommatidia_per_eye) / _counts

    def _norm_bin(vals, n_bins):
        lo, hi = vals.min(), vals.max()
        b = ((vals - lo) / max(hi - lo, 1e-9) * n_bins).astype(int)
        return np.clip(b, 0, n_bins - 1)

    # Flattened 9x9 cell index per ommatidium (matches OMMATIDIA_ROWS/COLS
    # in dashboard.jsx).
    VISION_GRID_ROWS = VISION_GRID_COLS = 9
    _cell = _norm_bin(_cy, VISION_GRID_ROWS) * VISION_GRID_COLS + _norm_bin(_cx, VISION_GRID_COLS)
    _cell_counts = np.bincount(_cell, minlength=VISION_GRID_ROWS * VISION_GRID_COLS).astype(float)
    _cell_counts[_cell_counts == 0] = 1.0

    # Real spectral classes: flygym's own pale_type_mask (216 pale / 505
    # yellow of 721) — the same pale/yellow R7-R8 distinction the
    # connectome annotates on its own photoreceptors (see
    # run_simulation.py's _photoreceptor_masks).
    _pale = retina.pale_type_mask.astype(bool)
    _yellow = ~_pale

    obs, info = sim.reset(seed=0)
    log.info("physics_worker_process: flygym ready, entering the real (slow) step loop.")

    forward_drive = 0.0
    fatigue = 0.0
    resting = False
    vision_grid = None
    vision_broadband = vision_pale = vision_yellow = [0.0, 0.0]
    while not stop_event.is_set():
        # Drain to the latest drive value — the brain process enqueues far
        # faster than this loop can consume.
        while True:
            try:
                forward_drive = drive_queue.get_nowait()
            except queue.Empty:
                break

        # Fatigue FSM: while resting, incoming drive is ignored outright
        # (effective_drive=0, legs idle at BASE_FREQ) and fatigue decays;
        # once it's low enough, normal DN-driven control resumes. This is
        # hysteresis (rest/resume thresholds differ) so it doesn't chatter
        # right at the boundary.
        if resting:
            fatigue = max(0.0, fatigue - FATIGUE_RECOVERY_RATE)
            effective_drive = 0.0
            if fatigue <= FATIGUE_RESUME_THRESHOLD:
                resting = False
        else:
            effective_drive = max(0.0, forward_drive)
            fatigue += effective_drive * FATIGUE_RATE
            if fatigue >= FATIGUE_REST_THRESHOLD:
                resting = True

        cpg.intrinsic_freqs[:] = BASE_FREQ + effective_drive * FREQ_GAIN
        cpg.step()

        joints_angles, adhesion_onoff = [], []
        for i, leg in enumerate(steps.legs):
            joints_angles.append(steps.get_joint_angles(leg, cpg.curr_phases[i], cpg.curr_magnitudes[i]))
            adhesion_onoff.append(steps.get_adhesion_onoff(leg, cpg.curr_phases[i]))
        action = {
            "joints": np.concatenate(joints_angles),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

        try:
            obs, reward, terminated, truncated, info = sim.step(action)
        except Exception:
            log.exception("physics_worker_process: sim.step failed — exiting")
            return

        if terminated or truncated:
            log.warning("physics_worker_process: episode ended (terminated=%s truncated=%s) — resetting",
                        terminated, truncated)
            obs, info = sim.reset(seed=0)
            continue

        pos = sim.physics.named.data.xpos[root_id]
        quat = sim.physics.named.data.xquat[root_id]
        # Afferent readout, straight off the real observation space (no
        # separate sensor model): total tarsal ground-contact force
        # magnitude (obs["contact_forces"] is (30,3), one 3D vector per
        # tarsus contact sensor) and RMS joint angular velocity
        # (obs["joints"][1] is the velocity row of the (3,42) array) — see
        # this module's docstring for which real sensory neurons these feed.
        contact_magnitude = float(np.sum(np.linalg.norm(obs["contact_forces"], axis=1)))
        joint_velocity_rms = float(np.sqrt(np.mean(obs["joints"][1] ** 2)))
        # Real per-joint angles (row 0 of the (3,42) array — see this
        # module's docstring), same order as fly.actuated_joints, for live
        # skeletal articulation in dashboard.jsx (fly_rigged.glb's joint
        # nodes are named exactly after these, e.g. "joint_LFTibia").
        joint_angles = [float(x) for x in obs["joints"][0]]

        # --- Real visual afference -------------------------------------
        # obs["vision"] is (2 eyes, 721 ommatidia, 2 photoreceptor
        # channels), values already normalized 0..1 by flygym's own retina
        # model. The retina only re-renders at vision_refresh_rate (500Hz
        # of SIM time, i.e. every ~20 physics steps), so recompute the
        # derived quantities only when flygym says it actually updated —
        # otherwise reuse the previous ones rather than burning CPU
        # re-reducing an identical array.
        if info.get("vision_updated", True) or vision_grid is None:
            vis = obs["vision"]              # (2, 721, 2)
            per_om = vis.mean(axis=2)        # (2, 721) mean across the 2 channels
            # Population means feeding the SNN's real photoreceptor
            # classes (see telemetry_server.py -> set_visual_afference):
            # broadband R1-R6 gets whole-eye luminance; pale/yellow R7-R8
            # get only their own ommatidia, per flygym's pale_type_mask.
            vision_broadband = [float(per_om[0].mean()), float(per_om[1].mean())]
            vision_pale = [float(per_om[0][_pale].mean()), float(per_om[1][_pale].mean())]
            vision_yellow = [float(per_om[0][_yellow].mean()), float(per_om[1][_yellow].mean())]
            # True spatial 9x9 downsample per eye, binned by each
            # ommatidium's real position on the retina (see _cell above).
            vision_grid = [
                (np.bincount(_cell, weights=per_om[e], minlength=_cell_counts.size) / _cell_counts).round(4).tolist()
                for e in (0, 1)
            ]

        try:
            telemetry_queue.put_nowait({
                "pos": [float(x) for x in pos],
                "quat": [float(x) for x in quat],
                "fatigue": float(fatigue),
                "resting": resting,
                "contact_magnitude": contact_magnitude,
                "joint_velocity_rms": joint_velocity_rms,
                "joint_angles": joint_angles,
                "vision_grid": vision_grid,
                "vision_broadband": vision_broadband,
                "vision_pale": vision_pale,
                "vision_yellow": vision_yellow,
            })
        except queue.Full:
            pass  # main process hasn't drained yet; this step's pose is superseded by the next anyway

    log.info("physics_worker_process: stop_event set, exiting.")


class PhysicsBridge:
    """Owned by the MAIN process. Spawns physics_worker_process as a real
    OS process and exposes the same set_forward_drive()/latest() interface
    the rest of telemetry_server.py already expects — the multiprocessing
    plumbing is entirely internal to this class."""

    def __init__(self):
        ctx = mp.get_context("spawn")
        self._drive_queue = ctx.Queue()
        self._telemetry_queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._process: mp.Process | None = None
        self._latest: dict | None = None

    def start(self):
        ctx = mp.get_context("spawn")
        self._process = ctx.Process(
            target=physics_worker_process,
            args=(self._drive_queue, self._telemetry_queue, self._stop_event),
            daemon=True,
        )
        self._process.start()
        log.info("PhysicsBridge: spawned physics_worker_process (pid=%s)", self._process.pid)

    def stop(self):
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()

    def set_forward_drive(self, mn_activation_left: float, mn_activation_right: float, reward_signal: float = 0.0):
        """Called from the BRAIN thread every brain tick. Never blocks:
        a full queue just means the physics process hasn't read the
        previous value yet, so this one's fine to drop — a fresher one
        will be along on the very next brain tick anyway.

        mn_activation_left/right are BrainLIF's muscle-activation-filtered
        readout of the real VNC leg motor-neuron population (see this
        module's docstring) — the efferent drive itself, not DN Hz.
        reward_signal is the real dopamine readout (already recentered on
        its resting baseline in run_simulation.py, so 0 == rest) — folded
        in here as an arousal multiplier on top of that drive.

        SYMMETRIC FORWARD-DRIVE LOCK (deliberate, do not "fix"):
        the two motor-pool activations are AVERAGED into one scalar, and
        physics_worker_process applies that single value uniformly to all
        six legs (cpg.intrinsic_freqs[:]). There is therefore no yaw path
        anywhere in this file, by design.

        This is not an oversight — it is the measured result. Three
        controlled experiments (sham_causality_test.py,
        lateralization_test.py, and the latter re-run on a fully silenced
        network with a verified 0-spike baseline) all found no usable
        lateral signal: unilateral odour drives MN_L and MN_R within ~4% of
        each other, and the steering statistic is null with inconsistent
        sign across cycles. ei_balance_trace.py explains why — the stimulus
        reaches 90% of the 176,422 neurons within four hops, so left/right
        identity is destroyed by fan-out rather than by inhibitory gating.
        Reintroducing a (mn_left - mn_right) yaw term would therefore be
        amplifying measurement noise into fake steering. Overall network
        excitation -> forward speed is the part that genuinely works."""
        base_drive = max(0.0, (mn_activation_left + mn_activation_right) * 0.5 * MN_ACTIVATION_TO_DRIVE_GAIN)
        arousal = min(MAX_AROUSAL, max(MIN_AROUSAL, 1.0 + DOPAMINE_AROUSAL_GAIN * reward_signal))
        drive = base_drive * arousal
        try:
            self._drive_queue.put_nowait(drive)
        except queue.Full:
            pass

    def latest(self) -> dict | None:
        """Drains telemetry_queue to its newest item (if any) and caches
        it — this is what makes "broadcast the last known position when
        nothing new has arrived" work: an empty queue just means keep
        returning the cached value."""
        while True:
            try:
                self._latest = self._telemetry_queue.get_nowait()
            except queue.Empty:
                break
        return self._latest
