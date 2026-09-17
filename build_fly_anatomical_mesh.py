#!/usr/bin/env python3
"""
build_fly_anatomical_mesh.py — one-off asset build, NOT part of the live
telemetry/physics pipeline. Assembles flygym's own real per-segment
MuJoCo visual meshes (69 real STL parts shipped inside the flygym package
itself — Thorax, six-legged Coxa/Femur/Tibia/Tarsus1-5, wings, halteres,
antennae, A1A2/A3-A6 abdomen segments; confirmed via
flygym.Fly().model.find_all('mesh'), nothing downloaded or invented),
using the EXACT body-tree transforms (dm_control mjcf RootElement from
flygym.Fly()) the live physics simulation itself uses for these same
parts — not re-authored, not guessed.

Two outputs:
  --rigged (default): dashboard/public/fly_rigged.glb, a real multi-node
    hierarchy — one glTF node per flygym joint (all 42 actuated ones,
    named exactly as flygym names them, e.g. "joint_LFCoxa_yaw"), so
    dashboard.jsx can rotate each node live from real obs["joints"]
    angles. Several of flygym's 42 joints stack on the SAME MuJoCo body
    (e.g. Coxa gets yaw+pitch+roll = 3 hinges before its own frame is
    final) — verified via fly.model.find('body','LFCoxa').joint — so this
    is a genuine per-JOINT chain, not one node per body part.
  --static: dashboard/public/fly.glb, the original single fused static
    mesh (kept for reference/fallback) — same real geometry, no
    articulation.

Rooted at "FlyBody" (the real free-joint attachment body whose position IS
what physics_worker.py/obs["fly"][0] tracks), not at "Thorax" — Thorax sits
~[0.5, 0, 1.3]mm offset from FlyBody in the real model, so rooting here
keeps the assembled mesh in exact alignment with the telemetry the
dashboard already applies every frame.

Basis change: every part stays in flygym/MuJoCo's own native local axes
(Z-up, +X-forward) at every node EXCEPT one single top-level "MJ_ROOT"
node, which carries the one-time change of basis into three.js's axes
(Y-up, -Z-forward) — three_x=-mj_y, three_y=mj_z, three_z=-mj_x, matching
dashboard.jsx's MJ_TO_THREE constant exactly (see its own comment for the
live-data verification against the real three.js library). Applying the
basis change ONCE at the root means every per-joint LIVE rotation dashboard.jsx
applies at runtime is a plain single-axis local rotation in flygym's own
axis convention — no per-joint basis conversion needed.

Run once (re-run only if flygym's model or init_pose changes):
    python3 build_fly_anatomical_mesh.py            # builds fly_rigged.glb
    python3 build_fly_anatomical_mesh.py --static    # builds fly.glb too
"""
from __future__ import annotations

import io
import sys

import numpy as np
import trimesh

import flygym

RIGGED_OUT_PATH = "dashboard/public/fly_rigged.glb"
STATIC_OUT_PATH = "dashboard/public/fly.glb"

# Same verified basis change as dashboard.jsx's MJ_TO_THREE (see that
# constant's comment): three_x=-mj_y, three_y=mj_z, three_z=-mj_x.
MJ_TO_THREE_MATRIX = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
])


def _quat_matrix(quat) -> np.ndarray:
    q = quat if quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
    return trimesh.transformations.quaternion_matrix(q)  # trimesh also uses wxyz


def _local_transform(pos, quat) -> np.ndarray:
    m = _quat_matrix(quat)
    m[:3, 3] = pos if pos is not None else np.zeros(3)
    return m


def _load_part(g) -> trimesh.Trimesh:
    raw = g.mesh.file.contents
    scale = g.mesh.scale if g.mesh.scale is not None else np.ones(3)
    part = trimesh.load(io.BytesIO(raw), file_type="stl")
    part.apply_scale(scale)
    part.apply_transform(_local_transform(g.pos, g.quat))
    # flygym's own micro-CT-derived meshes are far denser than any of these
    # ~70 small parts needs on-screen — decimate each part to ~12% of its
    # own face count (floor of 300, below which decimation would mangle
    # small parts like antennae/tarsus segments more than it helps). Done
    # per-part (not once after concatenation) so this applies identically
    # whether parts stay separate (rigged) or get merged (static).
    target = max(300, len(part.faces) // 8)
    if len(part.faces) > target:
        part = part.simplify_quadric_decimation(face_count=target)
    # flygym's STLs carry no material/vertex-color data of their own — a
    # uniform dark chitin tone, not an invented "accurate" color (no real
    # color data exists to be accurate to).
    part.visual.vertex_colors = np.tile([40, 40, 46, 255], (len(part.vertices), 1))
    return part


def build_rigged_model(out_path: str = RIGGED_OUT_PATH):
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    model = fly.model

    scene = trimesh.Scene()
    root_matrix = np.eye(4)
    root_matrix[:3, :3] = MJ_TO_THREE_MATRIX
    scene.graph.update(frame_to="MJ_ROOT", matrix=root_matrix)

    # dm_control preserves MJCF document order, which for a body's own
    # <joint> children is exactly the order MuJoCo composes them in
    # (parent frame -> joint1 -> joint2 -> ... -> body's own final frame,
    # where geoms and child bodies attach) — verified against a real
    # example: LFCoxa's joints come out as
    # [joint_LFCoxa_yaw(axis X), joint_LFCoxa(axis Y, "pitch"),
    #  joint_LFCoxa_roll(axis Z)], each a pure hinge pivoting at the
    # body's own origin (pos=[0,0,0]), matching flygym's documented
    # yaw->pitch->roll leg convention.
    joints_by_body: dict[str, list] = {}
    for j in model.find_all("joint"):
        joints_by_body.setdefault(j.parent.name, []).append(j)

    processed: dict[str, str] = {}  # body.full_identifier -> its terminal node name

    def node_for(b) -> str:
        key = b.full_identifier
        if key in processed:
            return processed[key]
        parent_node = "MJ_ROOT" if b.parent.tag == "worldbody" else node_for(b.parent)

        # The body's own fixed offset from its parent's (already-final)
        # frame — this is the ONE place body.pos/body.quat get used;
        # everything after this is pure joint rotation, baked as identity
        # (the rest/zero-angle pose) and rotated live at runtime.
        cur = b.name
        scene.graph.update(frame_to=cur, frame_from=parent_node, matrix=_local_transform(b.pos, b.quat))

        for j in joints_by_body.get(b.name, []):
            scene.graph.update(frame_to=j.name, frame_from=cur, matrix=np.eye(4))
            cur = j.name  # each joint's own node becomes the parent for the next

        for g in b.geom:
            if g.mesh is None:
                continue
            part = _load_part(g)
            mesh_node = f"{cur}__mesh"
            scene.add_geometry(part, node_name=mesh_node, parent_node_name=cur, geom_name=mesh_node)

        processed[key] = cur
        return cur

    bodies = model.find_all("body")
    for b in bodies:
        node_for(b)

    n_joint_nodes = sum(len(v) for v in joints_by_body.values())
    print(f"assembled {len(bodies)} real bodies, {n_joint_nodes} real joint nodes "
          f"({len(fly.actuated_joints)} of them actuated/live-drivable)")
    scene.export(out_path)
    print(f"wrote {out_path}")


def build_static_mesh(out_path: str = STATIC_OUT_PATH):
    fly = flygym.Fly(enable_adhesion=True, init_pose="stretch", control="position")
    model = fly.model

    world_xform: dict[str, np.ndarray] = {}

    def body_world(b) -> np.ndarray:
        key = b.full_identifier
        if key in world_xform:
            return world_xform[key]
        local = _local_transform(b.pos, b.quat)
        world = local if b.parent.tag == "worldbody" else body_world(b.parent) @ local
        world_xform[key] = world
        return world

    parts = []
    for b in model.find_all("body"):
        w = body_world(b)
        for g in b.geom:
            if g.mesh is None:
                continue
            part = _load_part(g)
            part.apply_transform(w)
            parts.append(part)

    print(f"assembled {len(parts)} real anatomical parts from flygym's own MuJoCo meshes "
          f"(each already decimated per-part in _load_part)")
    combined = trimesh.util.concatenate(parts)
    basis_change = np.eye(4)
    basis_change[:3, :3] = MJ_TO_THREE_MATRIX
    combined.apply_transform(basis_change)
    combined.export(out_path)
    print(f"wrote {out_path}: {len(combined.vertices)} verts, {len(combined.faces)} faces")


if __name__ == "__main__":
    build_rigged_model()
    if "--static" in sys.argv:
        build_static_mesh()
