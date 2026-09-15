# fruit_fly — connectome-driven *Drosophila* vision-flight model

**Snapshot**: 2026-09-14, ~15:15. Two facts here are moving targets — verify
before trusting them: `data/brain_graph.json`'s `meta` block (which graph is
currently active), and whether `data/brain_graph_full.json` exists yet (the
full-dataset background fetch — see "Live background jobs" below).

## What this is

A real connectome — male-cns:v1.0, the Janelia/HHMI FlyEM+Google male
*Drosophila* central nervous system reconstruction, accessed via neuPrint —
drives a biophysically-parameterized spiking neuron model. That model's
output flies a fly avatar through a browser-rendered park. Nothing here is
trained or learned: synaptic weights are literal neuPrint synapse counts
(signed by real neurotransmitter predictions), and the neuron model's
parameters are ported from a published paper rather than fit to this
project's own data. The explicit standing constraint through this whole
project has been "use the ecosystem's own tools, don't reinvent, don't
train what you can wire up directly from real data."

## How the project got here (matters for understanding *why*, not just *what*)

1. **Original architecture** (now removed): flygym/NeuroMechFly simulated a
   physical fly body (MuJoCo contact physics, walking legs) in a
   `MixedTerrain` arena. A LIF brain's descending-neuron (DN) output drove
   flygym's native `HybridTurningFly` CPG controller directly — no RL
   training, matching the standing constraint above.
2. Performance was investigated in depth: MuJoCo's own contact solver at
   the biologically-standard `dt=1e-4` timestep is the hard bottleneck
   (~90-110 physics steps/s measured, i.e. ~0.01x real time) — this is true
   of flygym's own reference examples too, not something this project's
   code did wrong. Two *real* inefficiencies were found and fixed on top of
   that floor: the camera's `render()` was being called (and paying its
   cost) on ticks that could never produce a frame, and `camera._frames`
   (meant for offline video export) grew unbounded — a genuine memory leak
   in a long-running server. An attempt to raise the timestep to `5e-4` for
   a ~4x speedup was tested and looked stable in isolation, but broke under
   real interactive use (the fly went airborne and tumbled) — reverted to
   `1e-4`. This is a documented example of "tested but insufficiently" in
   this project's own history; see it as a caution about trusting short
   isolated benchmarks for anything touched by user interaction.
3. **The neuron model was rewritten to match a specific published paper**:
   Shiu et al. 2023, "A leaky integrate-and-fire computational model based
   on the connectome of the entire adult Drosophila brain"
   ([paper](https://doi.org/10.1101/2023.05.02.539144),
   [reference code](https://github.com/philshiu/Drosophila_brain_model)) —
   fetched and read directly, not summarized from a search result. The
   original ad-hoc LIF (normalized units, no refractory period, no synaptic
   delay, constant-current sensory drive) was replaced with the paper's
   actual equations and constants.
4. **Full pivot to vision-first flight** (explicit user directive, executed
   only after the user was warned it meant discarding all the validated
   flygym/physics work and confirmed anyway): flygym, MuJoCo, and all body
   physics were deleted. The fly's body and the park it flies through are
   now a lightweight kinematic model in the browser. This also incidentally
   *solved* the performance problem by construction — there's no physics
   engine left to bottleneck on; the brain alone benchmarks at 400-900+
   steps/s.
5. Wiring the connectome's actual visual pathway in surfaced two more real
   findings, both verified empirically rather than assumed: (a) the
   originally-scoped pilot subgraph had **zero** direct-or-2-hop path from
   visual sensory neurons to DNs — the real relay population
   (`visual_projection`) had to be identified and added; (b) even after
   adding it, the pathway stayed completely silent, because the optic-lobe
   sensory neurons are 100% histaminergic and real Drosophila
   photoreceptors are *inhibitory* — a purely-recurrent downstream network
   with no other tonic drive has nothing for that inhibition to act on, so
   it just sits at rest. A small background Poisson drive
   (`noise_std=0.4`) fixed this, confirmed by direct spike-count
   comparison (0 downstream spikes at `noise_std=0`, 100k+ at 0.4).
6. A visualization pass replaced flat `LineSegments` for the real DN
   fiber morphology with instanced oriented cylinders (real volumetric
   thickness, still one draw call), and fixed an "empty screen" report by
   switching every park/avatar material to self-illuminated
   `MeshBasicMaterial` and adding a visible fly body + third-person chase
   camera (a pure first-person view over open terrain reads as nothing on
   screen).

## Repository layout

```
fetch_brain.py           neuPrint node+edge extraction (checkpointed, resumable)
fetch_skeletons.py       navis skeleton (3D morphology) extraction for DNs
prepare_skeleton_viz.py  decimates raw skeletons into browser-viable LOD tiers
run_simulation.py        BrainLIF (the neuron model) + VisionFlightBridge (brain-only bridge)
telemetry_server.py      FastAPI + WebSocket server, bidirectional with the browser
watch_and_swap.py        background watcher: swaps in the full dataset when its fetch finishes
train.py                 thin CLI alias for run_simulation.py's benchmark entrypoint
dashboard/               Vite + React Three Fiber frontend
data/                    brain_graph.json (active graph), skeleton files, logs, checkpoints/
requirements.txt         Python deps (see below — trimmed of now-unused flygym/mujoco/etc)
AGENTS.md                this file
```

No automated test suite exists anywhere in this project. All verification
has been manual and empirical: standalone Python scripts exercising
`BrainLIF`/`VisionFlightBridge` directly, and real `websockets` client
scripts against the actually-running server — not mocked, not assumed.
That pattern (write a small script, run it against the live thing, read
the real numbers) is how to verify any change here; there's no `pytest`
target to reach for instead.

## The brain (`run_simulation.py`)

### `BrainLIF`

Loads `brain_graph.json`, builds a sparse `scipy.sparse.csr_matrix`
adjacency matrix, and runs the Shiu et al. LIF model exactly:

```
dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
dg/dt = -g / tau                 (unless refractory)
spike when v > v_th; on spike: v = v_rst, g = 0, refractory for t_rfc
synapse: on presynaptic spike (after t_dly delay), g += w
w per connection = (+1 or -1 per neurotransmitter) * w_syn * (raw synapse count)
external drive: independent Poisson input per neuron, each event adding
  w_syn * f_poi directly to v (not g); Poisson-target neurons have no
  refractory period, exactly as in the reference model
```

Constants (class attributes on `BrainLIF`, mV/ms/Hz throughout):
`V_0=V_RST=-52`, `V_TH=-45`, `T_MBR=20`, `TAU_SYN=5`, `T_RFC=2.2`,
`T_DLY=1.8`, `W_SYN=0.275`, `R_POI=150`, `F_POI=250`. Citations for each
(Kakaria & de Bivort 2017 for the membrane constants, Jürgensen et al. for
the synaptic time constant, Lazar et al. for the refractory period, Paul et
al. 2015 for the synaptic delay) are in the class docstring, carried over
verbatim from the reference model's own comments — not independently
re-derived.

Numerically: the outer step (`dt` param, seconds — tied to whatever cadence
the bridge calls `.step()` at) is subdivided into `SUB_DT=0.5ms` internal
Euler sub-steps, since 0.5ms is small relative to the 5ms synaptic time
constant and needed for a numerically accurate integration — the outer `dt`
alone (2ms default) would be too coarse. Synaptic transmission has a real
delay: a ring buffer of length `round(T_DLY / SUB_DT)` holds recent spike
vectors, and `g` only receives a spike's contribution `T_DLY` ms later, not
instantly. Refractory period is tracked as a per-neuron float countdown
(`refrac_ms`), and — matching the reference model precisely — Poisson-input
target neurons (`is_sensory`) are exempt from it entirely, so they can spike
as fast as their driving input allows.

**Threshold-check ordering matters and was a real bug once**: the check
happens at the *top* of each sub-step, on the `v` value carried in from the
previous sub-step, before that sub-step's own leak/integration runs. This
is mathematically equivalent to checking after integration for organic
dynamics (a neuron crossing threshold during integration is always
sub-threshold by the time that sub-step returns, so checking first-thing
next call is a no-op until it isn't) — but it's what makes
`force_spike()` (manually pushing a neuron's `v` above threshold between
calls) actually work: checking *after* the leak term first would decay a
freshly-forced supra-threshold `v` back under threshold before it was ever
detected as a spike, since the leak shaves off a non-trivial fraction of
`v` every sub-step. This was found, diagnosed, and fixed during this
project — `force_spike` silently did nothing before the fix.

Sign convention (`NT_SIGN` module constant): acetylcholine, dopamine,
octopamine, serotonin → excitatory (+1); GABA, glutamate, histamine →
inhibitory (-1). This is standard-in-the-field, not invented for this
project, and was directly confirmed against this graph's own data: all
6,098 `ol_sensory` (optic-lobe/photoreceptor) neurons carry `nt =
"histamine"`, matching real *Drosophila* photoreceptor biology (histamine-
gated chloride channels — light literally *reduces* photoreceptor firing;
these cells are inhibitory onto their targets, not excitatory). Neurons
with an NT this project's map doesn't recognize (unclear/unpredicted)
default to +1 (excitatory) — a simplification, not something separately
validated.

### Populations identified from real neuPrint metadata, not guessed

- **Descending neurons (DNs)**: `superclass` field equals
  `descending_neuron` or `descending_neuron_tbc`. **Not** `type` starting
  with "DN" — that was checked and rejected early on: `DN1a`/`DN1p` in this
  dataset are circadian "Dorsal Neurons" in the central brain, unrelated to
  descending motor command neurons. `superclass` is the field the dataset's
  own documentation and structure treat as authoritative.
- **Optic-lobe sensory (vision input)**: `superclass == 'ol_sensory'`.
- **Visual relay (vision→central-brain link)**: `superclass ==
  'visual_projection'` — added specifically because the direct
  sensory→DN path was empirically confirmed to not exist in the originally
  scoped subgraph (see "Live background jobs" for exact edge counts).
- Left/right split for both DNs and sensory neurons uses the `soma_side`
  field (`'L'`/`'R'`) where present. **Caveat found empirically**: only 36
  of the 6,098 `ol_sensory` neurons have `soma_side` populated at all — the
  vast majority fall into an "unsided" bucket (see `is_sensory_unsided` in
  `BrainLIF.__init__`), which gets driven by the *average* of left/right
  visual input rather than a true per-eye signal. The L/R visual asymmetry
  that does reach DN output is real but weak for this reason — a known,
  documented limitation, not a hidden one.

### `VisionFlightBridge`

Thin wrapper: owns one `BrainLIF`, exposes `set_visual_input(left, right)`
(0..1 normalized brightness, clipped, → `BrainLIF.visual_L/visual_R`),
`step()` (advances the brain one outer step, returns a `BrainSnapshot`:
`t`, `thrust`, `yaw_rate`, `spiking_ids`), and `reset()` (clears brain
state — membrane potentials, conductances, refractory timers, delay
buffers, DN-rate EMA — back to rest). `flight_command()` on `BrainLIF`
does the DN-rate → (thrust, yaw) mapping: overall DN firing rate → forward
thrust, left-minus-right DN rate → yaw bias. This is the same L/R-readout
math the old flygym-era walking controller used, just relabeled from "CPG
leg drive" to "flight kinematics" — the brain-side logic didn't need to
change for the pivot, only its interpretation downstream.

### CLI (`python3 run_simulation.py [--graph PATH] [--steps N] [--seed N]`)

Standalone benchmark/smoke-test: constructs a `VisionFlightBridge`, feeds
it a synthetic slowly-oscillating L/R visual signal so `set_visual_input`
is actually exercised, steps it `--steps` times, logs periodic
thrust/yaw/spike-count and a final steps/sec figure. This is the fastest
way to sanity-check a change to `BrainLIF` without needing the server or
browser running at all.

## Data pipeline

### `fetch_brain.py` — nodes and edges via neuprint-python

Connects to `neuprint.janelia.org` using `NEUPRINT_TOKEN`/`NEUPRINT_DATASET`
from `.env` (dataset is `male-cns:v1.0`). Two phases, independently
checkpointed:

- **Nodes**: one `fetch_custom` Cypher query per superclass (the known
  superclass list is hardcoded as `KNOWN_SUPERCLASSES`, discovered by
  directly querying `MATCH (n:Neuron) WHERE n.superclass IS NOT NULL RETURN
  DISTINCT n.superclass, count(n)` against the live database — not
  guessed), each batch written to
  `data/checkpoints/nodes/<superclass>.parquet`. A final `_unclassified`
  batch (`superclass IS NULL`) is included **only** on a full,
  unscoped run — a scoped run (`--only-superclasses`) correctly excludes
  it now, but didn't originally: an early bug had every scoped run also
  pull the full unclassified set (thousands of irrelevant neurons),
  massively inflating fetch time for no reason. Fixed.
- **Edges**: `neuprint.fetch_adjacencies`, batched by source-bodyId chunks
  (`--edge-chunk-size`, default 300 for full runs). Each chunk's result is
  written to `data/checkpoints/edges/edges_<first_bodyid>_<last_bodyid>_
  <count>.parquet`. **This content-based key is a fix, not the original
  design**: the first version keyed chunks by positional index
  (`chunk_00000.parquet`), which works fine within one run but silently
  reuses the *wrong* cached chunk if the run's node scope changes between
  invocations (a pilot run's "chunk 0" and a full run's "chunk 0" cover
  completely different source bodyIds). Caught before it corrupted a real
  fetch; fixed to be content-keyed.
- `--min-weight` (default 3) drops edges below that total synapse count —
  standard connectome-export noise filtering, not arbitrary: the raw graph
  is far too dense/noisy to be useful otherwise.
- Output: a single JSON (`--out`, default `data/brain_graph.json`) with
  `meta` (dataset, generated_at, node/edge/DN counts), `nodes` (id, type,
  instance, class, superclass, is_dn, soma_side, soma [x,y,z] in nm,
  nt, status), `edges` (source, target, weight).

### `fetch_skeletons.py` — 3D morphology via navis

Uses `navis.interfaces.neuprint.fetch_skeletons`, the documented method
from Janelia's own male-cns download page
(`male-cns.janelia.org/download/`) — fetched and read directly, not
assumed. **Install `navis` (core), not `navis[all]`**: the `[all]` extra
pulls a full Qt GUI stack (PySide6, VTK, pyvista, ~1GB) for plotting
backends this headless server never uses; this was tried, the runaway
install was caught mid-download, and the core package (confirmed
sufficient — `fetch_skeletons` works identically) was installed instead.

Default scope is DNs only (`descending_neuron`/`descending_neuron_tbc`,
1,316 neurons). Full-morphology skeletons are heavy — one DN's skeleton
alone runs ~16,000 nodes at 8nm resolution; the full ~176k-neuron dataset
at that density would be tens of GB, not browser- or even disk-viable.
Batched (`--batch-size`, default 20 neurons/call) and checkpointed the same
content-keyed way as edges. Output: `data/dn_skeletons.parquet`
(`body_id, node_id, x, y, z, radius, parent_id`, float32).

### `prepare_skeleton_viz.py` — decimation for the browser

The raw skeleton parquet (8.29M nodes across 1,316 neurons) is far too
dense to render directly — an early attempt at topology-preserving
decimation (keeping every branch/leaf point) still produced 1M+ segments.
The current approach instead pre-filters to nodes at or above a radius
percentile (`--min-radius-percentile`, default 0.8 — i.e. keep the
thicker "major tract" fibers, not fine terminal dendrites) and then
stride-decimates *within* that filtered set (`--stride`, default 15),
connecting each kept node to its nearest still-kept ancestor by walking up
the real parent chain (not just positional stride) so segments always
follow the actual skeleton topology. Two tiers are generated for the
dashboard's LOD switch: "fine" (~134k segments, `--stride 15
--min-radius-percentile 0.8`) and "coarse" (~11k segments, `--stride 60
--min-radius-percentile 0.92`). Output: flat float32 binary
(`[x0,y0,z0,x1,y1,z1]` per segment, pre-centered/scaled to match the
dashboard's soma-point convention exactly so the two overlay correctly) —
`data/dn_skeleton_segments.bin` / `_coarse.bin`, plus a small JSON sidecar
with the centroid/scale/counts used.

## Server (`telemetry_server.py`)

FastAPI app, one WebSocket route (`/ws`), a `/health` endpoint, and
`data/` mounted at `/static/` (so the dashboard can `fetch()` the graph
JSON and skeleton `.bin` files directly).

**Architecture**: `SimWorker` owns one `VisionFlightBridge` and runs it in
an ordinary background `threading.Thread` (`while self._running: ...
bridge.step()`, as fast as the machine allows), guarding the latest
snapshot with a `threading.Lock`. A separate `asyncio` task
(`broadcaster()`) reads that snapshot and fans it out to every connected
websocket at a fixed `BROADCAST_HZ=60` cadence, independent of the brain's
own stepping rate. Control commands arrive over the *same* websocket
(`ws_endpoint`'s receive loop) and get pushed into a `queue.Queue`
(`SimWorker.submit`), drained and applied on the worker thread itself at
the top of each loop iteration (`_apply_command`) — never applied directly
from the asyncio side, to avoid touching `BrainLIF`'s numpy arrays from two
threads at once.

**Historical note, no longer relevant but worth knowing if you see old
code/comments referencing it**: the flygym-era version of this file had an
elaborate macOS-specific thread dance, because MuJoCo's GLFW window has to
be created on the process's actual main thread (an AppKit constraint) —
`run_blocking()`/`_worker_started_externally` existed solely to handle
that. With flygym removed, there's no GL context at all anymore, and the
current file is a plain, ordinary background-thread server on every
platform. If you find a reference to this constraint anywhere, it's stale.

**A real bug worth knowing about if you touch the broadcaster**: it used
to iterate the live `connections` set directly while sending
(`for ws in connections: ...`). `connections` can be mutated concurrently
by `ws_endpoint` on connect/disconnect, and iterating a live `set` while
it's mutated raises `RuntimeError: Set changed size during iteration` —
which, uncaught, silently killed the *entire broadcast task forever*
(asyncio only surfaces an unretrieved task exception at garbage-collection
time, easy to miss entirely). This was caught live: a second test
websocket client connecting while telemetry was flowing killed live
updates for the one real connected client too, with no visible error. Now
iterates `list(connections)` (a snapshot) and wraps the whole tick in
try/except with logging, so it can't die silently again.

### WebSocket protocol

**Server → client**, one JSON message per broadcast tick (~60Hz):
```json
{
  "t": 12.34,               // brain's own simulated-seconds clock
  "thrust": 0.5,             // 0..1, DN-derived forward drive
  "yaw_rate": 0.12,          // -2..2, DN-derived turn rate
  "spiking_ids": [123, 456], // neuprint bodyIds that spiked this tick (sparse)
  "paused": false,
  "sensory_drive": 1.0,      // current value, for the UI slider to stay in sync
  "noise_std": 0.4,
  "visual_L": 0.5,           // last visual input received, echoed back
  "visual_R": 0.5
}
```

**Client → server**, sent whenever the browser has something new:
```json
{"cmd": "visual_input", "left": 0.7, "right": 0.3}
{"cmd": "set_params", "sensory_drive": 1.2, "noise_std": 0.3}   // either key optional
{"cmd": "inject_spike", "body_ids": [123, 456]}                 // forces those neurons to fire next sub-step
{"cmd": "pause"} / {"cmd": "resume"} / {"cmd": "reset"}
```

## Dashboard (`dashboard/`, Vite + React Three Fiber)

Single main component file, `src/dashboard.jsx`. `src/App.jsx` just
renders `<Dashboard/>`; `src/App.css` has all layout/panel styling.

**Layout**: `.dashboard-root` is a CSS grid, 62% hero / 38% side column.
Hero (`.hero-fly`) holds one `<Canvas>` for the park+flight scene. Side
column stacks a second `<Canvas>` (brain viewer, `.brain-view`) over the
`ControlPanel`.

**Constants worth knowing**: `BRAIN_SCALE = 1/4000` (nm → scene units for
soma positions — chosen after actually checking the real soma bounding box
extent, not guessed; an earlier arbitrary scale placed the camera *inside*
the point cloud). `FLIGHT_SPEED = 7` (world units/sec at thrust=1 — a
tuning constant, not derived from anything). `COLOR_DECAY = 0.85` (per-frame
fraction of a spike-flash color that remains — controls how long a spiking
neuron stays lit).

**Component map**:
- `useBrainLayout(graph)` — centers/scales soma positions once, builds the
  bodyId→instance-index map spike flashing needs.
- `BrainInstances` — one `THREE.InstancedMesh` (icosahedron per neuron,
  `MeshBasicMaterial`), colors DNs vs. others, flashes white on spike (read
  from `spikeQueueRef`, decayed each frame in `useFrame`).
- `useSegmentBuffer(url)` / `TubeFibers` / `SkeletonFibers` — fetches the
  precomputed `.bin` segment buffers and renders them as instanced,
  individually-oriented-and-scaled unit cylinders (real 3D tube geometry,
  one draw call regardless of segment count) rather than flat
  `LineSegments`. `SkeletonFibers` owns the fine/coarse LOD switch,
  re-evaluated every 10 frames based on camera distance from origin.
- `BrainScene` — wraps the above in drei's `<Bounds fit clip observe>`
  (auto-frames the camera to whatever's actually in the scene — fixes the
  earlier camera-inside-the-point-cloud bug categorically, regardless of
  future scale changes) plus `OrbitControls`.
- `Park` — procedural (a `mulberry32` seeded PRNG, no external assets):
  a ground plane and ~70 cone-on-cylinder "trees" scattered by seeded
  random angle/distance. All `MeshBasicMaterial`.
- `FlyAvatar` — a small capsule+sphere+two-plane-wings mesh, position/
  heading read each frame from `flightStateRef`. Exists specifically
  because a pure first-person camera over open terrain was reported as
  "an empty screen" — there being nothing to visually anchor on.
- `FlightRig` — the kinematic integrator. Reads `flightCmdRef.current`
  (latest server thrust/yaw_rate, zero-order-held between websocket
  messages) each `useFrame`, integrates `flightStateRef.current`
  (`x, y, z, heading`) forward, positions the **third-person chase
  camera** (`camera` from `useThree()`, offset behind+above the avatar,
  `lookAt`-ed at it) for what the user sees, and separately drives a
  second, *never-rendered-to-screen* `PerspectiveCamera`
  (`eyeCamRef`, fov 100, matching the fly's actual position/heading) used
  only for the compound-eye brightness sample: every 3rd frame, that
  camera's view is rendered into a tiny offscreen `16x8`
  `WebGLRenderTarget`, read back via `gl.readRenderTargetPixels`, and
  split into left-half/right-half average luminance — sent to the server
  as `visual_input`. This two-camera split (display camera vs. sampling
  camera) exists because the user-facing view and "what the fly's compound
  eye sees" are conceptually different things once the camera became
  third-person.
- `ControlPanel` — connection status, pause/resume, reset, two sliders
  (`sensory_drive`, `noise_std` — sent live via `set_params` on every
  `onChange`), and three injection buttons (stimulate a random sample of
  sensory neurons, fire a random sample of left- or right-soma-side DNs).
- `Minimap` — plain 2D `<canvas>`, redrawn every `requestAnimationFrame`
  from `trailRef` (an array of `[x,z]` points `FlightRig` pushes into
  every 5th frame, capped at 800 entries).

**Material choice is deliberate and was a real, twice-repeated bug**: every
mesh in this project uses `MeshBasicMaterial` (self-illuminated, ignores
scene lighting entirely), never `MeshStandardMaterial`. A lit material's
vertex/instance colors depend on getting the light rig, exposure, and
tonemapping all correct — get any of it wrong and everything silently
renders near-black on a black background, indistinguishable from "nothing
is there." This happened once for the brain viewer (instance colors read
as invisible dots) and once for the park (reported as "an empty screen").
Don't reintroduce a lit material without adding real lights *and visually
confirming it*, not just building without errors.

Env: `VITE_API_BASE` (defaults `http://localhost:8000`) points the
dashboard at the backend; set in `dashboard/.env` for local dev against a
non-default port.

### Bug: static tree patch (and minimap) vs. an unbounded fly

Reported symptom: the hero Park canvas showed sky and the fly avatar but no
trees or ground; the top-down Minimap showed nothing at all, even after
minutes of unpaused flight with real thrust.

Investigated empirically, not guessed — with no browser access available
in that session, a plain-React (non-WebGL) diagnostic overlay was added
first specifically to separate "did the JS logic run" from "did the GPU
draw it": live `Park.trees` count, first tree's world position,
`flightStateRef`'s live `x/y/z/heading`, and `camera.position`, all as
plain HTML text polled off a ref every 500ms. That overlay reported
`Park.trees=70` (so tree generation itself worked) and a sane-looking
camera, which at first suggested a rendering-layer failure (WebGL context
loss from two simultaneous heavy `<Canvas>`es was the leading hypothesis).
That hypothesis was killed by the actual numbers: `flyPos` was ~309 world
units from the origin after several unpaused minutes, and manually
clicking "reset flight" made the trees reappear immediately.

Real root cause: `Park()` scattered its ~70 trees *once*, in a fixed disk
of radius 6-76 around world-origin `(0,0)`, on a fixed 500×500 ground
plane also centered on the origin — neither had any relationship to the
fly's *current* position. Once the fly (kinematically unbounded — nothing
in the vision-first flight model constrains how far it can travel) got far
enough from the origin, it flew straight out of the one small patch of
world that had anything in it. The Minimap had the identical bug in
miniature: it drew the trail at `screen = center + worldPos * scale`,
i.e. anchored on world-origin rather than the fly, so at 309 units out the
entire trail plotted far outside its 150×150 canvas — a canvas
`width`/`height` mismatch (the specific failure mode raised as a
hypothesis to check) was ruled out; both attributes were already correct
HTML attributes, not a CSS issue.

Fix: an infinite deterministic chunked field replaces the static patch.
World XZ space is divided into fixed `CELL_SIZE=24` cells; each cell's
trees are derived from `mulberry32(hash(cellX, cellZ))` (`cellSeed()`, a
small murmur-style integer hash) — a pure function of the cell's own
coordinates, so the same point in the world always has the same trees
regardless of mount/unmount history, with no stored world state anywhere.
`ParkChunks` recomputes which cells fall within `VIEW_RADIUS_CELLS=6` of
`flightStateRef`'s current position roughly 4x/second (only when the fly
has actually crossed into a new center cell, not every check), mounting
newly-in-range `TreeCell`s and letting out-of-range ones unmount. The
ground plane was simplified rather than chunked: one large (1200×1200)
`<mesh>` re-centered under the fly's `(x, z)` every frame — far simpler
than tiling terrain, and combined with re-centering it effectively never
runs out for any realistic session length. `Minimap` was fixed the same
way conceptually: it now anchors on the trail's own latest point (the
fly's current position) rather than world origin, so the fly indicator
stays fixed at canvas-center and the trail is always drawn relative to it.

Verified two ways, since browser access wasn't available in that session:
(1) the exact `cellSeed`/`mulberry32` logic run standalone in Node against
test positions from `(0,0)` out to `(-50000, 12345)` — every position
produced the same `676` nearby trees (`13×13` cells × 4 trees/cell,
matching `VIEW_RADIUS_CELLS=6`), and querying the same cell twice produced
byte-identical trees, confirming determinism; (2) the diagnostic overlay
(extended to report `nearbyTreeCount`/`activeCellCount`/`centerCell`) was
left in place — grep for `TEMP DIAGNOSTIC` in `dashboard.jsx` if it's
still there and no longer wanted, or use it again for the next investigation.

## Running everything

```bash
# backend (from repo root)
source .venv/bin/activate
python3 telemetry_server.py --host 127.0.0.1 --port 8010

# frontend
cd dashboard && npm run dev
```

`.venv` is Python 3.11 specifically — the system default (3.14) is too new
for some scientific packages' wheels; this was hit and worked around early
on. `pip install -r requirements.txt` covers the backend; `npm install` in
`dashboard/` covers the frontend (`package.json` lists exact pins —
`react`/`react-dom` are pinned to `19.2.8` specifically because
`@react-three/fiber@9.7.0` requires `react <19.3`, and npm's default
resolution picked `19.3.0` and broke the install until pinned down).

## Live background jobs (check before assuming any of this is done)

- **PID from `fetch_brain.py --out data/brain_graph_full.json`**: fetching
  the complete, unscoped male-cns:v1.0 dataset (all superclasses, ~176k
  neurons total — confirmed from the node-fetch phase's own log line
  before this got backgrounded). Check progress:
  `grep -oE "chunk [0-9]+/[0-9]+" data/fetch_full.log | tail -1`. This is
  the real neuPrint API responding at its own pace (roughly 20-30s/chunk)
  — not something a code change can speed up.
- **`watch_and_swap.py`**: polling for `data/brain_graph_full.json` to
  appear and stabilize on disk. When it does: backs up the *current*
  `data/brain_graph.json` to `data/brain_graph_pilot.json` (only if that
  backup doesn't already exist — check first if you specifically need
  today's vision-pilot graph, not an older pilot), copies the full graph
  into place, and restarts `telemetry_server.py`. The full graph will be
  far heavier than anything benchmarked so far — `BrainLIF` construction
  time and per-step sparse-matvec cost both scale with node/edge count,
  and this hasn't been measured at ~176k-node scale yet.
- **Active graph right now** (`data/brain_graph.json`, check its `meta`
  block for current truth): 17,323 nodes / 246,365 edges, scoped to
  `descending_neuron`, `descending_neuron_tbc`, `vnc_motor`, `ol_sensory`,
  `visual_projection`, plus (added for the mushroom-body learning circuit,
  see below) the `Kenyon_Cell`, `MBON`, and `DAN` **classes**. Current
  totals: 21,824 nodes, 467,378 edges. The sensory-scope decision itself
  was arrived at empirically, not chosen upfront — the first vision-pilot
  fetch (DN+motor+`ol_sensory` only) had **zero** sensory→DN paths (checked
  directly: 0 direct edges, 0 two-hop paths); `visual_projection` was added
  and re-fetched, which produced a real path (4,637 sensory→relay edges,
  17,119 relay→DN edges).

## Visual quality pass: park + fly avatar (no external assets)

The original park/fly geometry (single-cone tree canopies, a capsule +
two flat rectangles for the fly) was placeholder-grade. Upgraded without
introducing an external glTF dependency (no asset licensing to track, no
network fetch at runtime) and without touching the brain/data architecture
at all — purely a `dashboard.jsx` rendering change:

- **Trees** (`TreeCell`): 3-layer stacked canopy instead of one cone, plus
  real per-tree color variation (`hsl()` hue/lightness drawn from the same
  seeded `mulberry32(cellSeed(cx,cz))` RNG already used for tree placement
  — still fully deterministic per cell, just one more draw per tree).
- **Fly** (`FlyAvatar`): replaced the capsule-and-rectangles placeholder
  with a segmented thorax/abdomen/head, two actual compound-eye spheres
  (previously a single flat red disc), and curved teardrop wing
  silhouettes built from a `THREE.Shape` with quadratic Bezier curves
  (`useWingGeometry`) instead of flat `planeGeometry` rectangles — one
  shared geometry, mirrored via negative X scale for the second wing
  rather than duplicated.
- **Sky**: replaced the flat `<color attach="background">` with drei's
  real `<Sky>` component (Preetham atmospheric model). Safe to add without
  the MeshStandardMaterial light-rig problem below — `<Sky>` is a
  self-contained shader, not lit by scene lights.
- **Lighting stayed MeshBasicMaterial-only, deliberately.** The brief
  offered a choice: add real lights *and visually confirm them*, or stay
  unlit with more color/shading variation instead. No browser tool was
  available in that session (Claude-in-Chrome was off, confirmed by
  retrying), so introducing `MeshStandardMaterial` — which this project has
  already shipped as literally-invisible-black twice for want of a matched
  light rig (see the Dashboard section above) — without a way to check the
  result was judged not worth the risk. The fake-volume color variation
  (canopy layers lighten going up, per-tree hue jitter) substitutes for it.

Verified by build only (`npm run build` clean) — **not** visually
confirmed, for the same no-browser-tool reason. Flagged explicitly to the
user for a screenshot-based check rather than claimed as done.

## Dopamine-modulated plasticity: KC→MBON, three-factor learning rule

The one deliberate, narrow exception to this project's "wire up real data,
don't train" rule: a biologically-grounded three-factor plasticity rule
(Hebbian coincidence × eligibility trace × dopamine neuromodulator) on the
real mushroom-body circuit, modeling how *Drosophila* actually does
associative (aversive/appetitive) learning — not a generic trainable
layer bolted onto the connectome.

**Population identification — queried live against neuPrint first, per
this project's own established pattern, not assumed:**
- No `kenyon_cell` or `mushroom_body` **superclass** exists in
  male-cns:v1.0 (confirmed: empty result set). Kenyon cells and MBONs live
  one level down, in the finer-grained `class` field —
  `class == 'Kenyon_Cell'` (4,064 neurons, real subtypes confirmed via
  `type`: KCg-m, KCab-s, KCab-m, KCa'b'-ap2, etc. — matching the real
  γ/α-β/α'-β' lobe taxonomy) and `class == 'MBON'` (97 neurons). Since
  `superclass` alone (`cb_intrinsic`, 32,164 neurons) can't isolate either
  population, `fetch_brain.py` gained a new `--only-classes` scoping
  option (parallel to `--only-superclasses`, same checkpoint machinery,
  keyed `class_<name>` so it can't collide with a superclass checkpoint).
- Dopaminergic neurons: `class == 'DAN'`, 340 in this dataset. Real PAM
  (reward/appetitive) vs. PPL (punishment/aversive) subpopulations
  confirmed by `type` prefix, not assumed from the literature: 316 PAM
  neurons (15 subtypes: PAM01-PAM15) and 24 PPL neurons (12 subtypes:
  PPL101-108, PPL201-204) — matching the real Aso et al. Drosophila
  mushroom-body reward/punishment circuit organization.
- Real connectivity confirmed before writing any plasticity code: 61,210
  raw KC→MBON synaptic connection edges exist in the live graph (a Cypher
  count query, before any fetch). After threshold-filtering and matrix
  construction, **44,042 real KC→MBON synapses** ended up in `W` and were
  made the plastic subset (`BrainLIF._build_plasticity`) — 9.4% of the
  467,378-edge graph, not the ~246k-edge full graph mentioned as a ceiling
  in the original brief (that number was from the graph's *previous*,
  smaller vision-only scope; the graph grew when KC/MBON/DAN were added).

**Mechanism** (`BrainLIF._substep`, step "1.5", and `step()`):
eligibility (`self.eligibility`, one float per plastic synapse — never a
246k/467k-length array) decays every sub-step
(`exp(-SUB_DT/tau_elig_ms)`, `tau_elig_ms=2000`) and increments by 1.0
wherever a KC and its MBON partner spike in the *same* sub-step (a direct
coincidence detector — the simplified eligibility formulation given in
Fremaux & Gerstner's 2016 three-factor-rule review). The third factor,
`dopamine_rate`, is a signed EMA of (PAM spike fraction − PPL spike
fraction) updated once per outer `step()` (a behavioral-timescale signal,
not sub-ms). When `learning_enabled` is true, each sub-step applies
`Δ = eta * eligibility * dopamine_rate * dt` **along each synapse's own
sign** (so reward always potentiates and punishment always depresses,
for excitatory and inhibitory synapses alike — "stronger" means "further
from zero in its own direction" either way), magnitude-clipped to
`[0.1, 3.0] × |w0|` so a synapse can weaken or strengthen but never flips
sign or diverges. All of it operates as an **in-place mutation of
`W.data[self.plastic_pos]`** — `self.plastic_pos` are real integer
positions into the CSR `.data` array, located once at startup by walking
`W`'s actual (sorted) sparsity structure per MBON row rather than trusting
the pre-construction edge-list order (scipy's `csr_matrix` constructor can
reorder/sum entries) — never a matrix rebuild.

**Performance, measured before/after on the same graph** (isolating the
plasticity mechanism's own cost from the unrelated fact that the graph
also grew by ~4,500 KC/MBON/DAN neurons for this feature):
99.6 steps/s with the plastic-subset code paths disabled vs. 91.8 steps/s
with them active — **~8.5% overhead**, paid whether or not
`learning_enabled` is true (the eligibility trace itself is always live;
only the weight-mutation step is gated by the toggle).

**Verified functionally with real data, not "it compiles"**: with
`learning_enabled=False`, 50 organic steps produced exactly 0 weight
drift (`mean |Δw/w0| = 0.0`) — confirms the toggle genuinely gates the
mutation. With `learning_enabled=True` and a forced PPL (aversive) spike
burst (`force_spike` on all 24 real PPL bodyIds, matched 24/24), 150 steps
produced `dopamine_rate` consistently negative (-0.08 to -0.03, correct
sign for pure punishment with no reward present) and moved **41,480 of
44,042 plastic synapses (94%) by more than 1%** — real, substantial,
downstream-propagated depression. Also confirmed live over the actual
running WebSocket server (not just the standalone script): `set_learning`
toggles `learning_enabled` in the broadcast, `reward_signal`/
`cumulative_reward` update every tick, and `plastic_weight_changes`
(a compact per-neuron aggregate — 4,158 KC/MBON entries, not all 44,042
synapse-level values, which would be tens of MB/s at 60Hz) arrived with
real nonzero entries on schedule.

One honest caveat this testing surfaced, not tuned away: 94% of the
*entire* plastic subset moved within 0.3 simulated seconds, most hitting
the depression floor — broader and faster than a selective,
stimulus-specific memory trace should look. Most likely cause: KC
activity here is driven by generic background noise (`noise_std`) and
undiscriminating visual input, not the sparse, odor-specific coding
pattern real Kenyon cells produce from actual olfactory receptor input —
so almost the whole KC population is "eligible" most of the time rather
than a sparse, stimulus-selective subset. The rule itself is verified
correct (toggle gates it, sign is right, coincidence-gated, converges to
its clip bounds and stays there); making the *learning* selective rather
than broad would need either a lower `plasticity_eta`, a stricter
coincidence window, or (more fundamentally) sparser/more differentiated KC
input than this project currently drives them with.

**Task 5 (obstacle avoidance) is wired end-to-end, not just the underlying
mechanism**: `FlightRig`'s `useFrame` recomputes the deterministic tree
field within the fly's own cell + 8 neighbors every frame (via the exact
same `cellSeed`/`mulberry32` math `TreeCell` renders from — no separate
"collision mesh" to keep in sync) and, on a real distance-based collision
(with a 1.5s cooldown so one collision doesn't spam the PPL population
every frame), sends `{cmd: 'inject_spike', body_ids: pplIds}` over the
*existing* WebSocket control channel — the same path the manual
control-panel buttons already used, no new backend command needed for the
stimulus itself.

**Dashboard additions**: a "learning on/off" toggle and dedicated
fire-PPL/fire-PAM buttons in `ControlPanel`; a reward HUD panel (signal +
cumulative, same visual style as the existing panels) in the hero
overlay; and `BrainInstances`' existing per-instance spike-flash color
mechanism (not reimplemented — literally the same array-write loop) was
extended to also pull KC/MBON soma points toward `LEARN_COLOR` (magenta)
proportional to `plastic_weight_changes`, alongside new base-color tints
for KC/MBON/DAN populations in `useBrainLayout` so they're visually
identifiable even before any learning has happened.

## Plasticity hardening: graph-swap safety + real KC sparsity via APL

Two follow-ups raised urgently because `watch_and_swap.py`'s full-dataset
fetch was closing in on completion (chunk 447/589 at the time) — and in
fact **finished and fired mid-investigation**: the live graph swapped from
21,824 nodes to the full 176,422-neuron dataset (10,615,306 edges) while
this work was in progress, giving an unplanned but very real test of the
first concern below against production behavior, not just a synthetic one.

**1) Graph-swap safety for the plastic index set.** The worry: `W.data`
positions are meaningless across a scope change (this is literally the
class of bug `fetch_brain.py`'s edge checkpoints already hit once,
fixed by switching from positional to content-derived keys — see the Data
Pipeline section). Investigated rather than assumed fixed or assumed
broken: `plastic_pos`/`plastic_pre_idx`/`plastic_post_idx` are derived in
`_build_plasticity`, called at the end of every `BrainLIF.__init__`, from
`is_kc`/`is_mbon` — boolean arrays computed fresh from *that* graph's own
`class` field — never cached or reused across instances. A grep of the
entire project for `pickle`/`np.save`/`np.load`/`joblib`/`shelve` returned
nothing: there is no cross-process persistence of `W`, weights, or the
plastic subset anywhere. `watch_and_swap.py` swaps by killing the whole
server process and starting a fresh one (`subprocess.Popen`), so every
swap already gets a from-scratch `BrainLIF` construction against whatever
graph is on disk at that moment. This was confirmed against **the actual
swap event**, not a staged test: the server log shows a clean restart at
18:12, rebuilding `is_kc/is_mbon/plastic_pos` against the new
176k-node graph and finding the identical real populations (KC=4064,
MBON=97, DAN=340, 44,042 plastic synapses) — because those are the same
real anatomical populations, just now reached via a much bigger
surrounding graph. It ran uninterrupted for hours afterward. Separately,
also stress-tested by construction: two `BrainLIF` instances built from
the same graph with `nodes` shuffled into a completely different order
still each correctly identified all sampled plastic synapses as genuine
KC→MBON pairs — proof the derivation is identity-based (via `class` and
real `id_to_idx` lookups), not position-cached, independent of the
"process restart saves you anyway" argument above.

So: the specific failure mode described (silent, wrong-synapse mutation
after a swap) does not exist in the current architecture, for the concrete
reason that nothing persists across a swap to *be* stale. What was still
worth adding, since the request's deeper point stands — a future change
that adds any form of saved/restored weight state would reintroduce
exactly this risk — is a real guardrail rather than just a clean bill of
health: `_build_plasticity` now asserts (not logs — raises, refusing to
continue) that `W.indices[plastic_pos] == plastic_pre_idx` exactly and
that every plastic pre-index is a real KC / every post-index a real MBON,
and computes a `graph_fingerprint` (node/edge/KC/MBON/plastic counts plus
a SHA1 of the sorted bodyId set) that isn't consulted by anything yet —
nothing saves or restores weight state today — but exists so that the day
something does, there's already a concrete "is this checkpoint for the
graph I just loaded" check to reach for, instead of needing to invent one
under pressure at that point.

**2) KC coding sparsity via the real APL circuit — checked for before
inventing anything, and it existed.** A direct neuPrint query found APL
(Anterior Paired Lateral) exactly as the real Drosophila mushroom-body
literature describes it: 2 neurons (`APL_L`, `APL_R`, one per hemisphere),
both confirmed GABAergic (`predictedNt`/`consensusNt` = 'gaba'), with real
synaptic connectivity in both directions — 4,249 KC→APL synapses (cholinergic,
confirmed positive sign) and 4,210 APL→KC synapses (confirmed negative/
GABAergic sign). This is the actual biological negative-feedback loop that
produces sparse KC coding in real flies (Lin et al. 2014, Papadopoulou et
al. 2011): KCs excite APL, APL inhibits KCs back, globally.

Because APL's superclass (`cb_intrinsic`) is included in the full,
unscoped dataset, it arrived "for free" the moment the background fetch
completed and `watch_and_swap.py` swapped it in — no new fetch, no new
code needed to wire the circuit itself. It was already sitting in `W`
with the correct signs the moment it was checked (`NT_SIGN['gaba'] =
-1.0` already existed from the original Shiu et al. port). The
sparsification this project's own three-factor rule needed is therefore
implemented with **zero new mechanism** — no k-WTA, no invented threshold —
just the real circuit already being simulated correctly by code written
for an unrelated earlier task.

What *was* fighting it: `noise_std` (this project's own earlier addition,
a background Poisson drive on every neuron, added to give the
histaminergic/inhibitory vision pathway something to disinhibit — see the
BrainLIF docstring) was still defaulting to 0.4, a flat excitatory push
working directly against APL's inhibition. Measured directly (reusing one
constructed `BrainLIF` and only mutating `noise_std` + `reset_state()`
between trials, to avoid re-paying the ~45s full-scale construction cost
per data point):

| `noise_std` | KC active fraction/tick | DN spikes / 80 steps |
|---|---|---|
| 0.4 (old default) | 20.8% | 18,981 |
| 0.2 | 17.4% | 11,636 |
| 0.1 | 14.9% | 7,698 |
| 0.05 | 13.6% | 5,715 |
| 0.0 | **11.8%** | **4,162** |

DN activity — the thing `noise_std` was originally added to protect —
stayed very much alive even at `noise_std=0` (down from 18,981 to 4,162
spikes over the same window, not to zero), so the graph having grown ~8x
richer since that original fix apparently gives the vision→DN pathway
enough other recurrent support that it no longer needs this particular
crutch. Default changed to **`noise_std=0.0`**. Re-ran the same
forced-PPL-spike learning-selectivity test from the Prompt 2 entry with
this change: the fraction of the 44,042 plastic synapses that moved
dropped from **94% to 76%** — a real, measured improvement, from removing
an artificial excitation rather than adding an artificial suppression.

**Honestly, not fully solved.** 76% is still far from the ~5-10%
population sparsity reported for real fly odor coding (Turner et al.
2008, Honegger et al. 2011), and this was not tuned further to chase that
number down with more knob-turning. The likely reason, per the same
diagnosis as before: `tau_elig_ms=2000` means the eligibility trace
integrates coincidences over a full 2 seconds (1,000 outer steps) — even
at a genuinely-reduced ~12%-per-tick KC firing rate, enough *distinct* KCs
fire at least once somewhere across that whole window to eventually touch
most of the plastic subset, which is a different problem than "too many
KCs active at any given instant" (the thing APL actually fixes) and would
need either a shorter eligibility window or, more fundamentally, KC input
that's structured like sparse odor coding rather than diffuse background
drive to fully close the gap. Left as a documented open item rather than
patched with an arbitrary additional threshold under time pressure.

## Prompt 4: the Prompt 3 fix was checked, and one of its metrics was wrong

Prompt 3's `noise_std=0.0` change was justified by "DN activity stays
alive" — but that only shows *something* keeps DNs firing, not that
*vision* is what's doing it. Background Poisson noise on 176k neurons can
sustain downstream activity all by itself. This section covers three
follow-up checks that were explicitly demanded rather than assumed: does
vision still drive behavior, does the eligibility window actually control
selectivity, and does the whole thing still run at real-world scale.

### 1. Visual→behavior coupling, measured directly

Built one `BrainLIF` against the full 176,422-node graph, `seed=1`, and
swept `set_visual_input` between `(left=1, right=0)` and `(left=0,
right=1)` — same seed, only reusing the instance via `reset_state()` +
mutating `noise_std` between trials to avoid re-paying construction cost.
Measured `yaw_rate` after a settle period, over a 100-step window, and
also tracked the *within-condition* standard deviation of yaw as a noise
floor (since this is a stochastic Poisson-driven network, a single-trial
point estimate isn't trustworthy — an earlier, hastier 40-step version of
this test showed a "13x" difference between `noise_std=0.0` and `0.4`
that this more careful re-run did not reproduce; that number should be
treated as noise, not a real effect size):

| `noise_std` | coupling (Δyaw, L1R0 − L0R1) | yaw std (noise floor) | SNR | KC active/tick |
|---|---|---|---|---|
| 0.0 | −0.0006 | 0.0026 | 0.2x | 12.1% |
| 0.2 | **+0.0071** | 0.0051 | **1.4x** | 17.4% |
| 0.4 (pre-Prompt-3) | +0.0048 | 0.0046 | 1.0x | 20.9% |

At `noise_std=0.0`, coupling is *smaller than its own measurement noise*
— indistinguishable from zero. Prompt 3's change genuinely broke
vision→behavior coupling while leaving DN spike counts looking healthy,
exactly the gap the user's critique named. Note also that `0.4` — the
*original* value, before any of this session's changes — only clears its
own noise floor by ~1.0x, i.e. barely at all; this coupling metric is
weak and noisy at every setting tested on this connectome, not just at
the broken one. `noise_std=0.2` is the only point that clears its noise
floor by a real margin (1.4x) while still improving KC sparsity over the
old `0.4` default (17.4% vs 20.9% active/tick). **Default changed to
`noise_std=0.2`** — a measured trade-off between two real constraints
(visual coupling, KC sparsity) on the same shared parameter, not a value
picked by eye. It does not fully recover whatever coupling strength the
original single-trial "13x" measurement implied, because that number
itself doesn't look trustworthy on re-test.

### 2. Eligibility window sweep — a negative result

Hypothesis going into this (from the Prompt 2 entry above): a shorter
`tau_elig_ms` should reduce how many KC→MBON synapses move per learning
event, improving selectivity. Tested by forcing 8 PPL (aversive) spike
events over 150 outer steps — a realistic multi-pulse learning episode,
not a single isolated pulse — at `tau_elig_ms` = 2000/1000/500/200/100,
restoring `W.data` to its pre-trial values and clearing eligibility
between each point:

| `tau_elig_ms` | fraction of 44,042 plastic synapses moved (>1%) | mean \|Δw/w₀\| |
|---|---|---|
| 2000 | 0.908 | 0.760 |
| 1000 | 0.907 | 0.756 |
| 500 | 0.908 | 0.747 |
| 200 | 0.912 | 0.718 |
| 100 | 0.918 | 0.650 |

Flat. Shortening the window by 20x changed the fraction moved by less
than 1 percentage point. The reason: eligibility decay resets *between*
coincidences, it doesn't gate *which* KC/MBON pairs ever coincide in the
first place — with 8 repeated dopamine pulses over the test window,
essentially every pair that fires together even once gets pushed, and
whether that push happens under a 2000ms or 100ms decay barely changes
whether the pair crosses the "moved >1%" bar. The actual lever on
selectivity is which pairs spike together at all, i.e. KC sparsity (the
APL mechanism from the Prompt 3 section above), not this time constant.
**`tau_elig_ms` left unchanged at 2000ms** — the literature-grounded
value — since the data shows shortening it would not have bought
anything, and changing a biophysical parameter with no measured benefit
would just be knob-turning.

### 3. Re-benchmark at real scale

The 99.6→91.8 steps/s figures from the Prompt 2 entry were measured on
the 17,323-node pilot graph. Re-ran the same benchmark (`BrainLIF.step()`
in a tight loop, no I/O) on the current full 176,422-node / 10,615,306-edge
graph:

| | steps/s |
|---|---|
| learning disabled | 5.23 |
| learning enabled | 5.33 |

Roughly 17-19x slower than the pilot-scale numbers, consistent with the
graph being ~10x more nodes and ~23x more edges (`W`'s sparse matvec
dominates `step()`'s cost, and edge count is what actually drives that).
Plasticity's own overhead is still negligible relative to everything
else at this scale (no measurable slowdown from learning being on — the
100-synapse-per-substep eligibility/weight-update work is nothing next
to a 10.6M-edge sparse matvec). At ~5.3 steps/s, a `tau_elig_ms=2000`
window now covers roughly 1,000 outer steps but only ~5.3 *wall-clock*
seconds worth of real-time stepping takes over a minute of CPU time to
simulate — real-time playback at this scale needs either a faster W
representation or accepting sub-real-time simulation speed; neither was
in scope for this prompt, but it's the number that determines whether
the eligibility window in seconds means the same thing in wall-clock
time as it did at pilot scale, and now it doesn't.

### Unrelated, found while re-benchmarking: the dashboard's payload choke

While restarting the server to apply the `noise_std` fix, the dashboard
was reported frozen on "Loading brain_graph.json...". Root-caused before
touching anything: `data/brain_graph.json` is now **542MB** (176,422
nodes + 10,615,306 edges, each edge a verbose `{"source", "target",
"weight"}` JSON object) — and a `grep` across `dashboard.jsx` for every
use of the parsed `graph` object showed it only ever reads `graph.nodes`
and `graph.meta`; `graph.edges` is not referenced anywhere in the
frontend. The dashboard was fetching and `JSON.parse`-ing the entire
542MB payload — over 90% of which is the edge list it immediately
discards — synchronously on the main thread, which is what froze the
tab. (Nodes alone are ~35MB; there is no per-node skeleton/morphology
data in this file at all, contrary to what it might look like from the
outside — soma position is a single `[x,y,z]` point per neuron, not a
navis skeleton.)

Fixed at the source rather than by adding frontend-side streaming/binary
encoding/Web Worker infrastructure the payload size doesn't actually
require: `BrainLIF.__init__` (`run_simulation.py`) already parses the
full graph once at startup to build `W`; it now also writes a
`brain_graph_nodes.json` sidecar (`{meta, nodes}`, no edges) next to the
source file every time it loads, regenerated on every load so a
`watch_and_swap.py` graph swap can't leave a stale one behind. The
dashboard's `GRAPH_URL` now points at that sidecar instead of the raw
graph file. Verified: sidecar came out to **37.1MB** (a 93% cut from
542MB), written to disk within ~20s of server startup (well before the
~50s full adjacency-matrix construction finishes), and fetches over the
already-mounted `/static` route in 1.5s locally.

## Git

One commit exists (`git log --oneline`: `558ae8f`, message "Checkpoint
before vision-first architecture discussion") — made as a safety net
*before* the vision-first rewrite began, since no version control existed
in this project until that point. Everything described in this document —
the entire vision-first pivot — is currently **uncommitted working-tree
state**. If you're picking this project up, check `git status` first and
consider committing before making further changes that could be hard to
undo by hand.
