#!/usr/bin/env python3
"""
watch_and_swap.py — Waits for the full male-cns:v1.0 fetch (fetch_brain.py,
writing data/brain_graph_full.json) to finish, then hot-swaps it in as the
active graph and restarts telemetry_server.py so the dashboard picks it up.

Run once in the background; it exits after performing the swap (or after
--timeout-hours with nothing to swap).

    nohup python3 watch_and_swap.py > data/watch_and_swap.log 2>&1 &
"""

import argparse
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent
PILOT = ROOT / "data" / "brain_graph.json"
FULL = ROOT / "data" / "brain_graph_full.json"
PILOT_BACKUP = ROOT / "data" / "brain_graph_pilot.json"
SERVER_LOG = ROOT / "data" / "server.log"


def log(msg: str):
    print(f"[watch_and_swap] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_server_pid() -> int | None:
    out = subprocess.run(
        ["pgrep", "-f", "telemetry_server.py --host"],
        capture_output=True, text=True,
    ).stdout.strip()
    pids = [int(p) for p in out.splitlines() if p.strip()]
    return pids[0] if pids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    deadline = time.time() + args.timeout_hours * 3600
    log(f"watching for {FULL} to appear (fetch_brain.py --out {FULL.name})...")
    while not FULL.exists():
        if time.time() > deadline:
            log(f"timed out after {args.timeout_hours}h — giving up, no swap performed")
            return
        time.sleep(args.poll_seconds)

    # The file is written via a single json.dump() call at the very end of
    # fetch_brain.py, but make sure its size has stabilized (dump fully
    # flushed to disk) before treating it as complete.
    last_size = -1
    while True:
        size = FULL.stat().st_size
        if size == last_size and size > 0:
            break
        last_size = size
        time.sleep(5)
    log(f"full graph ready: {FULL} ({last_size / 1e6:.1f} MB)")

    if not PILOT_BACKUP.exists():
        shutil.copy2(PILOT, PILOT_BACKUP)
        log(f"backed up pilot graph -> {PILOT_BACKUP}")

    shutil.copy2(FULL, PILOT)
    log(f"swapped {PILOT} -> full male-cns:v1.0 dataset")

    pid = find_server_pid()
    if pid:
        log(f"stopping telemetry_server.py (pid {pid})...")
        subprocess.run(["kill", str(pid)])
        time.sleep(3)

    log("restarting telemetry_server.py with the full graph "
        "(this will take noticeably longer to load: ~176k neurons vs 2.5k)...")
    with open(SERVER_LOG, "a") as logf:
        logf.write(f"\n\n=== restarted by watch_and_swap.py at {time.strftime('%H:%M:%S')} ===\n")
        subprocess.Popen(
            ["python3", str(ROOT / "telemetry_server.py"),
             "--host", args.host, "--port", str(args.port)],
            cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log("done — telemetry_server.py relaunched against the full connectome.")


if __name__ == "__main__":
    main()
