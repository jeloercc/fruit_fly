#!/usr/bin/env python3
"""
telemetry_server.py — FastAPI + WebSocket server driving the connectome
LIF brain (SimWorker, a background thread, via
run_simulation.VisionFlightBridge) and bridging to a live flygym/MuJoCo
body simulation running in its own OS PROCESS (PhysicsBridge, spawning
physics_worker.physics_worker_process — see that file's docstring for why
this had to be a process and not a thread: `import flygym` on any
background thread of THIS process reproducibly hangs it on macOS with an
AppKit assertion failure, confirmed with a minimal 15-line repro in this
project's own history — a separate process gets its own real main thread,
which sidesteps that entirely). MuJoCo's own real per-step cost
benchmarked at 0.03x real-time (biomechanics_env.py), so the 60Hz
broadcast below just reads whatever pose PhysicsBridge has most recently
finished computing — repeating it across ticks when the physics process
hasn't produced a new one yet, deliberately (see broadcaster()).

  server -> client, ~60Hz:
    - which neurons just spiked (bodyIds, sparse)
    - the DN-derived (thrust, yaw_rate) flight command
    - fly_pos/fly_quat: the live (not baked) MuJoCo torso pose, whenever
      PhysicsBridge has one
    - fly_fatigue/fly_resting: the body-side fatigue scalar and forced-rest
      flag computed in physics_worker.py (real dopamine/reward_signal feeds
      in as an arousal multiplier on the VNC motor drive there, not here)
    - fly_joints: real per-joint angles (42, same order as
      flygym.Fly().actuated_joints), for live skeletal articulation of
      dashboard.jsx's fly_rigged.glb

  Closed sensorimotor loop (both directions, both real neuPrint
  populations, see run_simulation.py/physics_worker.py docstrings for the
  actual neuron types/obs keys involved):
    - efferent (brain -> body): real vnc_motor leg-motor-neuron firing rate,
      muscle-activation-filtered, drives the physics process's CPG —
      replacing what used to be a raw DN-Hz-derived drive.
    - afferent (body -> brain): real tarsal contact-force / joint-velocity
      readouts from the physics process feed the real SNta*/SNch* sensory
      neuron populations every brain tick (see SimWorker._run below).
  client -> server, whenever the browser has a new sample:
    - {"cmd": "visual_input", "left": 0..1, "right": 0..1} — brightness
      sampled from the fly's-eye camera in the browser-rendered park scene,
      the actual compound-eye input to the optic-lobe sensory neurons.

Run:
    python3 telemetry_server.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from physics_worker import PhysicsBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("telemetry_server")

BROADCAST_HZ = 60.0
GRAPH_PATH = Path("data/brain_graph.json")

# Afferent pathway gains: real obs["contact_forces"]/obs["joints"] velocity
# magnitudes (measured directly off a live flygym walk in this project,
# not guessed — typical p50/p90 of ~19/44 for contact-force magnitude and
# ~13/18 for joint-velocity RMS) scaled onto set_group_drive's expected
# 0..5 range, landing "typically active" walking around 1-3 with headroom
# for real spikes to clip at 5 rather than blow the channel out.
TARSAL_CONTACT_GAIN = 0.07
CHORDOTONAL_GAIN = 0.17


class SimWorker:
    """Owns the VisionFlightBridge and steps it continuously in a background
    thread, exposing the latest snapshot under a lock."""

    def __init__(self, graph_path: Path, seed: int):
        self.graph_path = graph_path
        self.seed = seed
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._cmd_queue: queue.Queue = queue.Queue()
        self.paused = False
        self.bridge = None  # set once the bridge finishes initializing in _run()

    def submit(self, cmd: dict):
        """Thread-safe: called from the asyncio/websocket side to enqueue a
        control command, applied on the sim loop's own thread."""
        self._cmd_queue.put(cmd)

    def _apply_command(self, cmd: dict):
        kind = cmd.get("cmd")
        try:
            if kind == "set_params":
                if "sensory_drive" in cmd:
                    self.bridge.brain.sensory_drive = float(cmd["sensory_drive"])
                if "noise_std" in cmd:
                    self.bridge.brain.noise_std = float(cmd["noise_std"])
                if "motor_threshold_boost" in cmd:
                    self.bridge.brain.motor_threshold_boost = float(cmd["motor_threshold_boost"])
            elif kind == "set_group_drive":
                self.bridge.brain.set_group_drive(cmd["group"], float(cmd["rate"]))
            elif kind == "visual_input":
                self.bridge.set_visual_input(float(cmd.get("left", 0.5)), float(cmd.get("right", 0.5)))
            elif kind == "inject_spike":
                body_ids = [int(b) for b in cmd.get("body_ids", [])]
                n = self.bridge.brain.force_spike(body_ids)
                log.info("inject_spike: %d/%d bodyIds matched", n, len(body_ids))
            elif kind == "set_dan_clamp":
                self.bridge.brain.set_dan_clamp(bool(cmd.get("enabled", False)))
                log.info("SYNTHETIC DAN clamp %s", "ON" if cmd.get("enabled") else "OFF")
            elif kind == "set_learning":
                self.bridge.set_learning(bool(cmd.get("enabled", False)))
                log.info("learning %s", "enabled" if cmd.get("enabled") else "disabled")
            elif kind == "pause":
                self.paused = True
            elif kind == "resume":
                self.paused = False
            elif kind == "reset":
                self.bridge.reset()
                log.info("reset: brain state cleared")
            else:
                log.warning("unknown control command: %r", cmd)
        except Exception:
            log.exception("failed to apply command %r", cmd)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        from run_simulation import VisionFlightBridge

        log.info("SimWorker: initializing connectome brain (graph=%s)...", self.graph_path)
        bridge = VisionFlightBridge(self.graph_path, seed=self.seed)
        self.bridge = bridge
        log.info("SimWorker: brain ready, entering step loop.")

        while self._running:
            while True:
                try:
                    cmd = self._cmd_queue.get_nowait()
                except queue.Empty:
                    break
                self._apply_command(cmd)

            if self.paused:
                time.sleep(0.02)
                with self._lock:
                    if self._latest is not None:
                        self._latest["paused"] = True
                continue

            snap = bridge.step()
            self._tick_count = getattr(self, "_tick_count", 0) + 1

            # Efferent: feed the real VNC motor-neuron activation readout to
            # the physics process — a non-blocking queue put (see
            # PhysicsBridge.set_forward_drive), not a physics call, so this
            # fast brain loop is never blocked by MuJoCo's real ~3.9ms/step
            # cost in the other process.
            if physics_worker is not None:
                physics_worker.set_forward_drive(
                    snap.mn_activation_left, snap.mn_activation_right, snap.reward_signal,
                )
                # Afferent: route the physics process's own most recent
                # real contact-force/joint-velocity readout back into the
                # SNN's genuine tarsal/chordotonal sensory populations —
                # closes the loop the other direction. physics_worker's
                # telemetry_queue is also read by broadcaster() below on the
                # asyncio thread; PhysicsBridge._latest is only ever
                # wholesale-replaced (never mutated in place), so reading it
                # from this thread too is safe under the GIL without a lock.
                phys = physics_worker.latest()
                if phys is not None:
                    bridge.brain.set_group_drive(
                        "tarsal_contact", phys.get("contact_magnitude", 0.0) * TARSAL_CONTACT_GAIN,
                    )
                    bridge.brain.set_group_drive(
                        "chordotonal", phys.get("joint_velocity_rms", 0.0) * CHORDOTONAL_GAIN,
                    )
                    # Real retinal afference: MuJoCo's two compound-eye
                    # renders, reduced per eye per photoreceptor class in
                    # physics_worker.py, injected as Poisson current into
                    # the matching real connectome populations.
                    if phys.get("vision_broadband") is not None:
                        bridge.brain.set_visual_afference(
                            phys["vision_broadband"], phys["vision_pale"], phys["vision_yellow"],
                        )

            snapshot = {
                "t": snap.t,
                "thrust": snap.thrust,
                "yaw_rate": snap.yaw_rate,
                "dn_left_rate": snap.dn_left_rate,
                "dn_right_rate": snap.dn_right_rate,
                # The actual efferent signal driving physics (see
                # physics_worker.py) — dn_left/right_rate above are the
                # older flight_command() readout, kept for reference but no
                # longer connected to the body.
                "mn_activation_left": snap.mn_activation_left,
                "mn_activation_right": snap.mn_activation_right,
                # Efferent bottleneck readout: the 981-cell
                # connectivity-derived premotor DN population, plus the
                # individually-named locomotor DNs (DNb01/DNb02 forward,
                # DNp09 stop). See BrainLIF.dn_readout().
                "dn_premotor_rate": snap.dn_premotor_rate,
                "dn_named_rates": snap.dn_named_rates,
                "spiking_ids": snap.spiking_ids,
                "paused": False,
                "sensory_drive": bridge.brain.sensory_drive,
                "noise_std": bridge.brain.noise_std,
                "motor_threshold_boost": bridge.brain.motor_threshold_boost,
                "group_drive": bridge.brain.group_drive,
                "visual_L": bridge.brain.visual_L,
                "visual_R": bridge.brain.visual_R,
                "reward_signal": snap.reward_signal,
                "cumulative_reward": snap.cumulative_reward,
                "learning_enabled": snap.learning_enabled,
                "dan_clamp": bridge.brain.dan_clamp,
            }
            # Per-neuron plastic-weight-change map is compact (KC+MBON =
            # ~4.1k neurons here) but still not free — send it every 30
            # ticks (~2x/sec) rather than every broadcast tick; the
            # dashboard just keeps showing the last one it got otherwise.
            if self._tick_count % 30 == 0:
                snapshot["plastic_weight_changes"] = bridge.brain.plastic_node_weight_changes()
            with self._lock:
                self._latest = snapshot

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None


worker: SimWorker | None = None
physics_worker: PhysicsBridge | None = None
connections: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker, physics_worker
    if worker is None:
        worker = SimWorker(GRAPH_PATH, seed=0)
    if physics_worker is None:
        physics_worker = PhysicsBridge()
    worker.start()
    physics_worker.start()
    task = asyncio.create_task(broadcaster())
    yield
    task.cancel()
    worker.stop()
    physics_worker.stop()


app = FastAPI(title="fruit_fly telemetry", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if GRAPH_PATH.parent.exists():
    app.mount("/static", StaticFiles(directory=str(GRAPH_PATH.parent)), name="static")


async def broadcaster():
    period = 1.0 / BROADCAST_HZ
    while True:
        try:
            t0 = time.monotonic()
            if worker is not None and connections:
                snap = worker.latest()
                if snap is not None:
                    # Whatever the physics thread has computed so far —
                    # None until flygym/MuJoCo finishes initializing, then
                    # literally "the last pose it finished computing,"
                    # repeated as many broadcast ticks as it takes for the
                    # next real ~3.9ms step to land. That's the actual
                    # computation rate, not a bug to paper over.
                    phys = physics_worker.latest() if physics_worker is not None else None
                    if phys is not None:
                        snap["fly_pos"] = phys["pos"]
                        snap["fly_quat"] = phys["quat"]
                        snap["fly_fatigue"] = phys.get("fatigue", 0.0)
                        snap["fly_resting"] = phys.get("resting", False)
                        snap["fly_joints"] = phys.get("joint_angles")
                        # Real 9x9-per-eye spatial downsample of the actual
                        # retina render (binned by each ommatidium's true
                        # position — see physics_worker.py), for the
                        # dashboard's L/R ommatidia grid.
                        snap["vision_grid"] = phys.get("vision_grid")
                    payload = json.dumps(snap)
                    # Snapshot to a list before iterating: `connections` can be
                    # mutated concurrently by ws_endpoint() on connect/disconnect,
                    # and iterating a live set while it's mutated raises
                    # RuntimeError — which, left uncaught, would silently kill
                    # this whole task forever (asyncio only surfaces unretrieved
                    # task exceptions at GC time, easy to miss).
                    dead = set()
                    for ws in list(connections):
                        try:
                            await ws.send_text(payload)
                        except Exception:
                            dead.add(ws)
                    connections.difference_update(dead)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, period - elapsed))
        except Exception:
            log.exception("broadcaster tick failed — continuing")
            await asyncio.sleep(period)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    log.info("client connected (%d total)", len(connections))
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                cmd = json.loads(msg)
            except json.JSONDecodeError:
                log.warning("dropped malformed control message: %r", msg[:200])
                continue
            if worker is not None:
                worker.submit(cmd)
    except (WebSocketDisconnect, RuntimeError):
        # Starlette normally raises WebSocketDisconnect on a clean close, but
        # receive_text() can also raise a bare RuntimeError ("WebSocket is
        # not connected. Need to call 'accept' first.") if the socket was
        # already torn down by the time this call runs — e.g. the
        # broadcaster's own send_text() already discovered the client gone
        # and dropped it from `connections` first. Same outcome either way:
        # this client is gone, clean up and end the handler quietly instead
        # of it propagating as an unhandled 500 in the logs.
        pass
    finally:
        connections.discard(websocket)
        log.info("client disconnected (%d total)", len(connections))


@app.get("/health")
async def health():
    snap = worker.latest() if worker else None
    return {"status": "ok", "has_snapshot": snap is not None, "clients": len(connections)}


def main():
    import uvicorn

    global worker, physics_worker

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    worker = SimWorker(GRAPH_PATH, seed=0)
    physics_worker = PhysicsBridge()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
