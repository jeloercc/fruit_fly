#!/usr/bin/env python3
"""
telemetry_server.py — FastAPI + WebSocket server that drives the
connectome-brain <-> flygym bridge (run_simulation.NeuroMechFlyBrainBridge)
in a background thread and broadcasts live state to the React Three Fiber
dashboard at 60Hz:

  - which neurons just spiked (bodyIds, sparse — not the full state vector)
  - the fly's pose in the arena
  - the DN-derived [left, right] CPG drive
  - a JPEG-encoded camera frame (throttled, so bandwidth stays sane)

The physics/brain loop runs as fast as the machine allows in its own thread
(flygym/MuJoCo stepping is blocking, so it must not run on the asyncio event
loop); a separate asyncio task reads the latest snapshot and fans it out to
every connected websocket client exactly at 60Hz, decoupling wall-clock
telemetry rate from simulation stepping rate.

Run:
    python3 telemetry_server.py --host 0.0.0.0 --port 8000
    python3 telemetry_server.py --no-render   # telemetry only, no video frames

(On macOS, MuJoCo's on-screen-capable GL backend must create its window on
the process's real main thread — an AppKit constraint — so when rendering is
on, this launcher runs the sim loop on the main thread and pushes uvicorn to
a background thread. `--no-render`, or any non-macOS host, uses the simpler
layout: uvicorn owns the main thread and the sim runs in a worker thread.
The bare `uvicorn telemetry_server:app` invocation still works but always
disables rendering on macOS for that reason.)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("telemetry_server")

BROADCAST_HZ = 60.0
GRAPH_PATH = Path("data/brain_graph.json")


class SimWorker:
    """Owns the NeuroMechFlyBrainBridge and runs it continuously in a
    background thread, exposing the latest snapshot under a lock."""

    def __init__(self, graph_path: Path, render: bool, physics_timestep: float, seed: int):
        self.graph_path = graph_path
        self.render = render
        self.physics_timestep = physics_timestep
        self.seed = seed
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_frame_jpeg: str | None = None
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
            elif kind == "inject_spike":
                body_ids = [int(b) for b in cmd.get("body_ids", [])]
                n = self.bridge.brain.force_spike(body_ids)
                log.info("inject_spike: %d/%d bodyIds matched", n, len(body_ids))
            elif kind == "pause":
                self.paused = True
            elif kind == "resume":
                self.paused = False
            elif kind == "reset":
                self.bridge.reset()
                log.info("reset: fly pose + brain state cleared")
            else:
                log.warning("unknown control command: %r", cmd)
        except Exception:
            log.exception("failed to apply command %r", cmd)

    def start(self):
        """Run the sim loop in a background thread. Only safe when rendering
        is off, or on platforms where offscreen GL doesn't require the
        process's main thread (e.g. Linux + EGL/OSMesa)."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def run_blocking(self):
        """Run the sim loop on the calling thread, forever. Use this on
        macOS when rendering is on: MuJoCo's GLFW window must be created on
        the process's actual main thread (AppKit constraint), so here the
        sim owns the main thread and uvicorn is pushed to a background
        thread instead — see `main()`."""
        self._running = True
        self._run()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        from run_simulation import NeuroMechFlyBrainBridge
        from PIL import Image

        log.info("SimWorker: initializing brain + flygym (graph=%s, render=%s)...",
                  self.graph_path, self.render)
        bridge = NeuroMechFlyBrainBridge(
            self.graph_path, physics_timestep=self.physics_timestep,
            render=self.render, seed=self.seed,
        )
        self.bridge = bridge
        log.info("SimWorker: simulation ready, entering step loop.")

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

            frame_jpeg = None
            if snap.frame is not None:
                # bridge.step() itself now only ever returns a frame on the
                # physics ticks its camera's own play_speed/fps gate says
                # are due (roughly 1-in-33) — no need for a second,
                # independent throttle stacked on top of that.
                buf = io.BytesIO()
                Image.fromarray(snap.frame).save(buf, format="JPEG", quality=70)
                frame_jpeg = base64.b64encode(buf.getvalue()).decode("ascii")
                self._last_frame_jpeg = frame_jpeg

            snapshot = {
                "t": snap.t,
                "fly_pos": snap.fly_pos,
                "fly_quat": snap.fly_quat,
                "dn_action": snap.dn_action,
                "spiking_ids": snap.spiking_ids,
                "frame_jpeg": frame_jpeg,  # only non-null on frames we actually encoded
                "paused": False,
                "sensory_drive": bridge.brain.sensory_drive,
                "noise_std": bridge.brain.noise_std,
            }
            with self._lock:
                self._latest = snapshot

    def latest(self) -> dict | None:
        with self._lock:
            if self._latest is None:
                return None
            # Reuse the last encoded frame on ticks where we didn't re-encode,
            # so every broadcast still carries *a* current-ish frame.
            snap = dict(self._latest)
            if snap["frame_jpeg"] is None:
                snap["frame_jpeg"] = self._last_frame_jpeg
            return snap


worker: SimWorker | None = None
connections: set[WebSocket] = set()
# Set True by main() when it has already started `worker` on the main thread
# (macOS + rendering). In that case `lifespan` must not spawn a second copy.
_worker_started_externally = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker
    if worker is None:
        # Default path (generic `uvicorn telemetry_server:app`, or any
        # platform where offscreen GL doesn't need the main thread): render
        # off by default here since a background-thread renderer is only
        # known-safe on non-macOS. Use the `python3 telemetry_server.py`
        # launcher below to get rendering on macOS too.
        worker = SimWorker(GRAPH_PATH, render=(sys.platform != "darwin"),
                            physics_timestep=1e-4, seed=0)
    if not _worker_started_externally:
        worker.start()
    task = asyncio.create_task(broadcaster())
    yield
    task.cancel()
    if worker:
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

    global worker, _worker_started_externally

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    render = not args.no_render

    if render and sys.platform == "darwin":
        log.info("macOS + rendering: sim/render loop will own the main thread; "
                  "uvicorn runs in a background thread.")
        worker = SimWorker(GRAPH_PATH, render=True, physics_timestep=1e-4, seed=0)
        _worker_started_externally = True

        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        server_thread = threading.Thread(
            target=lambda: asyncio.run(server.serve()), daemon=True,
        )
        server_thread.start()

        try:
            worker.run_blocking()
        except KeyboardInterrupt:
            pass
    else:
        worker = SimWorker(GRAPH_PATH, render=render, physics_timestep=1e-4, seed=0)
        _worker_started_externally = True
        worker.start()
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
