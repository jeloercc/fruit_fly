#!/usr/bin/env python3
"""
telemetry_server.py — FastAPI + WebSocket server for the vision-first flight
model. Drives run_simulation.VisionFlightBridge (connectome LIF brain only —
no flygym/MuJoCo, no physics engine) in a background thread and exchanges
state with the browser over one WebSocket, bidirectionally:

  server -> client, ~60Hz:
    - which neurons just spiked (bodyIds, sparse)
    - the DN-derived (thrust, yaw_rate) flight command
  client -> server, whenever the browser has a new sample:
    - {"cmd": "visual_input", "left": 0..1, "right": 0..1} — brightness
      sampled from the fly's-eye camera in the browser-rendered park scene,
      the actual compound-eye input to the optic-lobe sensory neurons.

There's no MuJoCo/GLFW rendering context here at all anymore, so (unlike
the old flygym-based version) there's no macOS main-thread constraint to
work around — the sim loop just runs in an ordinary background thread on
every platform.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("telemetry_server")

BROADCAST_HZ = 60.0
GRAPH_PATH = Path("data/brain_graph.json")


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
            elif kind == "visual_input":
                self.bridge.set_visual_input(float(cmd.get("left", 0.5)), float(cmd.get("right", 0.5)))
            elif kind == "inject_spike":
                body_ids = [int(b) for b in cmd.get("body_ids", [])]
                n = self.bridge.brain.force_spike(body_ids)
                log.info("inject_spike: %d/%d bodyIds matched", n, len(body_ids))
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

            snapshot = {
                "t": snap.t,
                "thrust": snap.thrust,
                "yaw_rate": snap.yaw_rate,
                "spiking_ids": snap.spiking_ids,
                "paused": False,
                "sensory_drive": bridge.brain.sensory_drive,
                "noise_std": bridge.brain.noise_std,
                "visual_L": bridge.brain.visual_L,
                "visual_R": bridge.brain.visual_R,
                "reward_signal": snap.reward_signal,
                "cumulative_reward": snap.cumulative_reward,
                "learning_enabled": snap.learning_enabled,
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
connections: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker
    if worker is None:
        worker = SimWorker(GRAPH_PATH, seed=0)
    worker.start()
    task = asyncio.create_task(broadcaster())
    yield
    task.cancel()
    worker.stop()


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
    except WebSocketDisconnect:
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

    global worker

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    worker = SimWorker(GRAPH_PATH, seed=0)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
