# fruit_fly

A real `male-cns:v1.0` fruit-fly connectome (176k neurons, fetched from neuPrint)
run as a leaky-integrate-and-fire spiking network, driving a live 5-module
telemetry dashboard: a procedural closed-loop optic-flow tunnel, the real 3D
connectome morphology, a sensory ommatidia matrix, a spike raster, and
mushroom-body dopamine/reward metrics.

[![Fruit Fly Simulation Demo](https://img.youtube.com/vi/CQzLku5Q6Is/0.jpg)](https://www.youtube.com/watch?v=CQzLku5Q6Is)

Two processes, both required:

- **Backend** (`telemetry_server.py`) — FastAPI + WebSocket server that steps
  the LIF brain (`run_simulation.py`) in a background thread and streams
  telemetry at 60Hz.
- **Frontend** (`dashboard/`) — Vite + React + Three.js dashboard that
  connects to the backend over WebSocket and renders it.

## 1. Prerequisites

- Python 3.11
- Node.js (for `npm`)
- A neuPrint API token for `male-cns:v1.0` (only needed if you're fetching
  the connectome yourself — see step 4)

## 2. Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (gitignored) if you need to
(re-)fetch data from neuPrint:

```
NEUPRINT_TOKEN=<your token>
NEUPRINT_DATASET=male-cns:v1.0
```

## 3. Frontend setup

```bash
cd dashboard
npm install
```

Create `dashboard/.env` (gitignored) pointing at wherever you'll run the
backend:

```
VITE_API_BASE=http://127.0.0.1:8010
```

## 4. Data

The backend reads `data/brain_graph.json` plus skeleton fiber buffers. If
`data/` is already populated (check for `brain_graph.json`,
`dn_skeleton_segments*.bin`, `optic_skeleton_segments*.bin`), skip to step 5.

Otherwise, fetch and prepare it (requires `NEUPRINT_TOKEN` above):

```bash
python3 fetch_brain.py                   # nodes + edges -> data/brain_graph.json
python3 fetch_skeletons.py                # DN skeletons -> data/dn_skeletons.parquet
python3 fetch_skeletons.py --superclasses ol_sensory,visual_projection \
    --out data/optic_skeletons.parquet    # optic-lobe skeletons

python3 prepare_skeleton_viz.py           # decimates dn_skeletons.parquet -> data/dn_skeleton_segments*.bin
python3 prepare_skeleton_viz.py --in data/optic_skeletons.parquet \
    --out-bin data/optic_skeleton_segments.bin \
    --out-meta data/optic_skeleton_meta.json   # same, for the optic set
```

`fetch_brain.py` is resumable (checkpoints under `data/checkpoints/`) and
supports `--only-superclasses` / `--skip-edges` for a bounded pilot run
instead of the full ~170k-neuron dataset — see the docstring at the top of
the script for exact flags.

The dashboard also loads coarse LOD variants (`*_segments_coarse.bin`,
switched in automatically past a camera-distance threshold) — produce those
with a second `prepare_skeleton_viz.py` pass using a larger `--stride` and
`_coarse`-suffixed `--out-bin`/`--out-meta` paths.

## 5. Run it

Terminal 1 — backend:

```bash
source .venv/bin/activate
python3 telemetry_server.py --port 8010
```

Wait for `SimWorker: brain ready, entering step loop.` in the logs (loading
+ building the ~10.6M-synapse adjacency matrix takes well under a minute).

Terminal 2 — frontend:

```bash
cd dashboard
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`). The
dashboard fetches the connectome, opens a WebSocket to the backend, and the
tunnel starts flying itself — DN spike rates from the real LIF network drive
the camera, wall proximity feeds back into the real optic-lobe sensory
neurons, and collisions/clean flight trigger the real PPL/PAM dopaminergic
populations that drive the mushroom-body learning panel.

Health check: `curl http://127.0.0.1:8010/health`.

## Notes

- `--host`/`--port` on `telemetry_server.py` default to `0.0.0.0:8000`;
  `dashboard/.env`'s `VITE_API_BASE` must match whatever you actually pass.
- `watch_and_swap.py` is an optional background helper that hot-swaps in a
  full (non-pilot) `brain_graph_full.json` once `fetch_brain.py --skip-edges`
  finishes fetching it, restarting `telemetry_server.py` automatically.
- `train.py` is an alias for `run_simulation.py`'s standalone CLI (benchmarks
  the LIF brain without the dashboard/WebSocket layer).
