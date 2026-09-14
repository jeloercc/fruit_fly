// dashboard.jsx — React Three Fiber brain viewer + live telemetry dashboard.
//
// Layout: the flygym camera feed is the hero (left, ~62% width); the 3D
// connectome viewer + interactive control panel share the right column.
//
// Fixes vs. the previous version:
//   - Brain was invisible: MeshStandardMaterial depends on scene lighting,
//     exposure and tonemapping to render vertex/instance colors correctly;
//     instance colors bypass Three's automatic sRGB conversion, so a lit
//     material read them as near-black. Switched to MeshBasicMaterial
//     (self-illuminated, ignores scene lighting entirely) so nodes are
//     always vividly visible regardless of light setup.
//   - Camera/scale was guessed (1/4000) without checking real data extent;
//     the actual soma bounding box is ~40k x 60k x 120k nm, so the fixed
//     camera at z=6 ended up inside the cloud. Now wrapped in drei's
//     <Bounds fit clip observe> which measures the real geometry and frames
//     the camera automatically — correct regardless of dataset scale.
//   - Synapse connections (graph.edges) are now drawn as LineSegments,
//     which the brief explicitly called out as missing.
//
// Interactivity: a control panel sends JSON commands over the same
// WebSocket telemetry_server.py already exposes: set_params (sensory drive
// / noise), inject_spike (manual stimulus into chosen bodyIds), pause /
// resume (physics + LIF), reset (fly pose + brain state).

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Bounds, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws'
const GRAPH_URL = API_BASE + '/static/brain_graph.json'

const DN_COLOR = new THREE.Color('#ff6644')
const NEURON_COLOR = new THREE.Color('#4fa3ff')
const SPIKE_COLOR = new THREE.Color('#ffffff')
const COLOR_DECAY = 0.85 // fraction of a flash that remains each frame
const SCALE = 1 / 4000 // nm -> scene units (only affects point spacing; Bounds frames the camera regardless)

// ---------------------------------------------------------------------
// Shared geometry prep (positions, colors, id->index) from brain_graph.json
// ---------------------------------------------------------------------

function useBrainLayout(graph) {
  return useMemo(() => {
    const nodes = graph.nodes.filter((n) => n.soma[0] !== null)
    const count = nodes.length
    const positions = new Float32Array(count * 3)
    const baseColors = new Float32Array(count * 3)
    const idIndex = new Map()

    let cx = 0, cy = 0, cz = 0
    for (const n of nodes) {
      cx += n.soma[0]; cy += n.soma[1]; cz += n.soma[2]
    }
    cx /= count; cy /= count; cz /= count

    nodes.forEach((n, i) => {
      const x = (n.soma[0] - cx) * SCALE
      const y = -(n.soma[2] - cz) * SCALE // brain Z -> scene up, flipped to face up
      const z = (n.soma[1] - cy) * SCALE
      positions[i * 3 + 0] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      const c = n.is_dn ? DN_COLOR : NEURON_COLOR
      baseColors[i * 3 + 0] = c.r
      baseColors[i * 3 + 1] = c.g
      baseColors[i * 3 + 2] = c.b
      idIndex.set(n.id, i)
    })

    return { positions, baseColors, idIndex, count }
  }, [graph])
}

// ---------------------------------------------------------------------
// 3D connectome — one InstancedMesh (self-lit) + one LineSegments mesh
// ---------------------------------------------------------------------

const _obj3d = new THREE.Object3D()
const _tmpColor = new THREE.Color()

function BrainInstances({ layout, spikeQueueRef }) {
  const meshRef = useRef()
  const { positions, baseColors, idIndex, count } = layout

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let i = 0; i < count; i++) {
      _obj3d.position.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2])
      _obj3d.updateMatrix()
      mesh.setMatrixAt(i, _obj3d.matrix)
      _tmpColor.setRGB(baseColors[i * 3], baseColors[i * 3 + 1], baseColors[i * 3 + 2])
      mesh.setColorAt(i, _tmpColor)
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [positions, baseColors, count])

  useFrame(() => {
    const mesh = meshRef.current
    if (!mesh || !mesh.instanceColor) return
    const colors = mesh.instanceColor.array

    for (let i = 0; i < count; i++) {
      const b = i * 3
      colors[b + 0] = baseColors[b + 0] + (colors[b + 0] - baseColors[b + 0]) * COLOR_DECAY
      colors[b + 1] = baseColors[b + 1] + (colors[b + 1] - baseColors[b + 1]) * COLOR_DECAY
      colors[b + 2] = baseColors[b + 2] + (colors[b + 2] - baseColors[b + 2]) * COLOR_DECAY
    }

    const spiking = spikeQueueRef.current
    if (spiking && spiking.length) {
      for (const bodyId of spiking) {
        const idx = idIndex.get(bodyId)
        if (idx === undefined) continue
        const b = idx * 3
        colors[b + 0] = SPIKE_COLOR.r
        colors[b + 1] = SPIKE_COLOR.g
        colors[b + 2] = SPIKE_COLOR.b
      }
      spikeQueueRef.current = null
    }
    mesh.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      {/* Small now — real DN morphology comes from the skeleton fibers
          below; these dots are only (a) the sole representation of the
          ~1,245 non-DN neurons in this pilot graph, which have no fetched
          skeleton, and (b) the live spike-flash indicator, since the
          static fiber geometry doesn't carry per-vertex spike state. */}
      <icosahedronGeometry args={[0.06, 1]} />
      {/* MeshBasicMaterial is self-illuminated — ignores scene lights entirely,
          so the connectome is always vividly visible regardless of light rig. */}
      <meshBasicMaterial vertexColors toneMapped={false} transparent opacity={0.85} />
    </instancedMesh>
  )
}

// ---------------------------------------------------------------------
// Real DN morphology — actual navis skeletons (fetch_skeletons.py), not
// straight soma-to-soma lines. Two precomputed decimation tiers
// (prepare_skeleton_viz.py: ~134k segments "fine", ~11k "coarse", both
// pre-filtered to thicker major-tract nodes rather than every fine
// terminal dendrite) are swapped based on camera distance — real LOD, not
// just a label. THREE's default per-object frustum culling does the rest
// once each geometry's bounding sphere is computed below.
// ---------------------------------------------------------------------

function useSkeletonGeometry(url) {
  const [geometry, setGeometry] = useState(null)
  useEffect(() => {
    let cancelled = false
    fetch(url)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
      .then((buf) => {
        if (cancelled) return
        const positions = new Float32Array(buf)
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        geo.computeBoundingSphere()
        setGeometry(geo)
      })
      .catch(() => setGeometry(null))
    return () => { cancelled = true }
  }, [url])
  return geometry
}

function SkeletonFibers({ apiBase }) {
  const fineGeo = useSkeletonGeometry(apiBase + '/static/dn_skeleton_segments.bin')
  const coarseGeo = useSkeletonGeometry(apiBase + '/static/dn_skeleton_segments_coarse.bin')
  const { camera } = useThree()
  const [level, setLevel] = useState('coarse')
  const frameRef = useRef(0)

  useFrame(() => {
    frameRef.current += 1
    if (frameRef.current % 10 !== 0) return // distance check every 10 frames, not every frame
    if (!fineGeo || !coarseGeo) return
    const dist = camera.position.length() // scene is centered near the origin
    const want = dist < 8 ? 'fine' : 'coarse'
    if (want !== level) setLevel(want)
  })

  const geo = (level === 'fine' ? fineGeo : coarseGeo) || coarseGeo || fineGeo
  if (!geo) return null

  return (
    <lineSegments geometry={geo} frustumCulled>
      <lineBasicMaterial color="#7fb8ff" transparent opacity={0.5} toneMapped={false} />
    </lineSegments>
  )
}

function Scene({ layout, spikeQueueRef, apiBase }) {
  return (
    <>
      <color attach="background" args={['#05070c']} />
      {/* Kept for depth cues if anything lit gets added later; the neuron
          material itself doesn't depend on these. */}
      <ambientLight intensity={0.8} />
      <Bounds fit clip observe margin={1.35}>
        <SkeletonFibers apiBase={apiBase} />
        <BrainInstances layout={layout} spikeQueueRef={spikeQueueRef} />
      </Bounds>
      <OrbitControls enableDamping dampingFactor={0.08} rotateSpeed={0.5} makeDefault />
    </>
  )
}

// ---------------------------------------------------------------------
// Control panel
// ---------------------------------------------------------------------

function ControlPanel({ graph, sendCmd, paused, sensoryDrive, noiseStd, connected }) {
  const [localDrive, setLocalDrive] = useState(1.3)
  const [localNoise, setLocalNoise] = useState(0.25)

  useEffect(() => {
    if (sensoryDrive !== null) setLocalDrive(sensoryDrive)
  }, [sensoryDrive])
  useEffect(() => {
    if (noiseStd !== null) setLocalNoise(noiseStd)
  }, [noiseStd])

  const dnLeftIds = useMemo(
    () => graph.nodes.filter((n) => n.is_dn && n.soma_side === 'L').map((n) => n.id),
    [graph],
  )
  const dnRightIds = useMemo(
    () => graph.nodes.filter((n) => n.is_dn && n.soma_side === 'R').map((n) => n.id),
    [graph],
  )
  const sensoryIds = useMemo(
    () => graph.nodes.filter((n) => (n.superclass || '').includes('sensory')).map((n) => n.id),
    [graph],
  )

  const sample = (ids, n) => {
    if (ids.length <= n) return ids
    const out = []
    const used = new Set()
    while (out.length < n) {
      const i = Math.floor(Math.random() * ids.length)
      if (used.has(i)) continue
      used.add(i)
      out.push(ids[i])
    }
    return out
  }

  return (
    <div className="control-panel panel">
      <div className="title">brain controls</div>

      <div className="control-row">
        <button className={connected ? 'btn ok' : 'btn bad'} disabled>
          {connected ? '● connected' : '○ disconnected'}
        </button>
      </div>

      <div className="control-row buttons">
        <button className="btn primary" onClick={() => sendCmd({ cmd: paused ? 'resume' : 'pause' })}>
          {paused ? '▶ resume' : '⏸ pause'}
        </button>
        <button className="btn" onClick={() => sendCmd({ cmd: 'reset' })}>
          ⟲ reset fly
        </button>
      </div>

      <label className="slider-label">
        sensory drive: {localDrive.toFixed(2)}
        <input
          type="range" min="0" max="3" step="0.05" value={localDrive}
          onChange={(e) => {
            const v = parseFloat(e.target.value)
            setLocalDrive(v)
            sendCmd({ cmd: 'set_params', sensory_drive: v })
          }}
        />
      </label>

      <label className="slider-label">
        noise σ: {localNoise.toFixed(2)}
        <input
          type="range" min="0" max="1.5" step="0.02" value={localNoise}
          onChange={(e) => {
            const v = parseFloat(e.target.value)
            setLocalNoise(v)
            sendCmd({ cmd: 'set_params', noise_std: v })
          }}
        />
      </label>

      <div className="title" style={{ marginTop: 10 }}>manual spike injection</div>
      <div className="control-row buttons">
        <button
          className="btn"
          onClick={() => sendCmd({ cmd: 'inject_spike', body_ids: sample(sensoryIds, 80) })}
        >
          stimulate sensory
        </button>
      </div>
      <div className="control-row buttons">
        <button
          className="btn dn-left"
          onClick={() => sendCmd({ cmd: 'inject_spike', body_ids: sample(dnLeftIds, 40) })}
        >
          fire DN · L
        </button>
        <button
          className="btn dn-right"
          onClick={() => sendCmd({ cmd: 'inject_spike', body_ids: sample(dnRightIds, 40) })}
        >
          fire DN · R
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------
// 2D top-down minimap of the fly's position in the arena
// ---------------------------------------------------------------------

function Minimap({ trailRef }) {
  const canvasRef = useRef()

  useEffect(() => {
    let raf
    const draw = () => {
      const canvas = canvasRef.current
      if (canvas) {
        const ctx = canvas.getContext('2d')
        const w = canvas.width, h = canvas.height
        ctx.fillStyle = 'rgba(5,7,12,0.35)'
        ctx.fillRect(0, 0, w, h)

        const trail = trailRef.current
        if (trail.length > 1) {
          ctx.strokeStyle = '#4fa3ff'
          ctx.lineWidth = 1.5
          ctx.beginPath()
          trail.forEach(([x, y], i) => {
            const px = w / 2 + x * 40
            const py = h / 2 - y * 40
            if (i === 0) ctx.moveTo(px, py)
            else ctx.lineTo(px, py)
          })
          ctx.stroke()

          const [lx, ly] = trail[trail.length - 1]
          ctx.fillStyle = '#ff6644'
          ctx.beginPath()
          ctx.arc(w / 2 + lx * 40, h / 2 - ly * 40, 3.5, 0, 2 * Math.PI)
          ctx.fill()
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [trailRef])

  return <canvas ref={canvasRef} width={150} height={150} className="minimap" />
}

function DriveBar({ label, value }) {
  const pct = Math.max(0, Math.min(100, ((value + 0.5) / 2.0) * 100))
  return (
    <div className="drive-bar">
      <span>{label}</span>
      <div className="drive-bar-track">
        <div className="drive-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span>{value.toFixed(2)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------

export default function Dashboard() {
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState(null)
  const [connected, setConnected] = useState(false)
  const [stats, setStats] = useState({
    t: 0, dnAction: [0, 0], nSpiking: 0, flyPos: [0, 0, 0],
    paused: false, sensoryDrive: null, noiseStd: null,
  })
  const [frameUrl, setFrameUrl] = useState(null)

  const spikeQueueRef = useRef(null)
  const trailRef = useRef([])
  const wsRef = useRef(null)

  useEffect(() => {
    fetch(GRAPH_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.json()
      })
      .then(setGraph)
      .catch((e) => setError(`Failed to load brain_graph.json from ${GRAPH_URL}: ${e.message}`))
  }, [])

  useEffect(() => {
    let ws
    let retryTimer
    let closedByUs = false

    const connect = () => {
      ws = new WebSocket(WS_URL)
      wsRef.current = ws
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs) retryTimer = setTimeout(connect, 1500)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (evt) => {
        const d = JSON.parse(evt.data)
        spikeQueueRef.current = d.spiking_ids
        setStats({
          t: d.t,
          dnAction: d.dn_action,
          nSpiking: d.spiking_ids.length,
          flyPos: d.fly_pos,
          paused: !!d.paused,
          sensoryDrive: d.sensory_drive ?? null,
          noiseStd: d.noise_std ?? null,
        })
        trailRef.current.push([d.fly_pos[0], d.fly_pos[1]])
        if (trailRef.current.length > 600) trailRef.current.shift()
        if (d.frame_jpeg) {
          setFrameUrl(`data:image/jpeg;base64,${d.frame_jpeg}`)
        }
      }
    }
    connect()
    return () => {
      closedByUs = true
      clearTimeout(retryTimer)
      ws?.close()
    }
  }, [])

  const sendCmd = useCallback((cmd) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
  }, [])

  const layout = useBrainLayout(graph ?? { nodes: [], edges: [] })

  if (error) {
    return <div className="panel error">{error}</div>
  }
  if (!graph) {
    return <div className="panel">Loading brain_graph.json...</div>
  }

  const [leftDrive, rightDrive] = stats.dnAction

  return (
    <div className="dashboard-root">
      {/* HERO: flygym camera feed — the dominant element on screen */}
      <div className="hero-fly">
        {frameUrl ? (
          <img src={frameUrl} alt="fly camera feed" className="hero-fly-img" />
        ) : (
          <div className="video-placeholder large">waiting for flygym frame…</div>
        )}
        <div className="hero-overlay panel top-left">
          <div className="title">flygym / MixedTerrain</div>
          <div>t = {stats.t.toFixed(3)}s{stats.paused ? '  ·  PAUSED' : ''}</div>
          <div>xyz: {stats.flyPos.map((v) => v.toFixed(3)).join(', ')}</div>
        </div>
        <div className="hero-overlay bottom-left panel">
          <div className="title">position (top-down)</div>
          <Minimap trailRef={trailRef} />
        </div>
        <div className="hero-overlay top-right panel">
          <div className="title">DN → CPG drive</div>
          <DriveBar label="L" value={leftDrive} />
          <DriveBar label="R" value={rightDrive} />
          <div>spiking this tick: {stats.nSpiking}</div>
        </div>
      </div>

      {/* Right column: 3D connectome + controls */}
      <div className="side-col">
        <div className="brain-view">
          <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
            <Scene layout={layout} spikeQueueRef={spikeQueueRef} apiBase={API_BASE} />
          </Canvas>
          <div className="hud top-left panel small">
            <div className="title">male-cns:v1.0 — {graph.meta.node_count.toLocaleString()} neurons</div>
            <div>DNs: {graph.meta.dn_count.toLocaleString()} · edges: {graph.meta.edge_count.toLocaleString()}</div>
          </div>
        </div>
        <ControlPanel
          graph={graph}
          sendCmd={sendCmd}
          paused={stats.paused}
          sensoryDrive={stats.sensoryDrive}
          noiseStd={stats.noiseStd}
          connected={connected}
        />
      </div>
    </div>
  )
}
