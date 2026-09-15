#!/usr/bin/env python3
"""
prepare_skeleton_viz.py — Decimate the raw DN skeletons (data/dn_skeletons.parquet,
8.29M nodes across 1,316 neurons, fetched via navis) into a compact binary
buffer the dashboard can render as real fibrous morphology (THREE.LineSegments)
instead of one point per soma.

Topology-aware decimation: branch points (>=2 children), leaves (0 children),
and roots are always kept (they define the fiber's actual shape); points
along a simple unbranched run are kept only every --stride nodes. Skipped
points are elided by walking up to the nearest *kept* ancestor, so segments
always connect kept points directly — this is a visual simplification (not
a scientific reconstruction), trading fine sub-branch detail for a fiber
count the browser can render at interactive framerate.

Output: data/dn_skeleton_segments.bin — a flat Float32 buffer of
[x0,y0,z0, x1,y1,z1, ...] per kept segment, in the SAME centering/scale
convention dashboard.jsx already applies to soma points from brain_graph.json
(so the two overlay correctly), plus a small JSON sidecar with counts and
the centering/scale constants used.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd


def decimate_neuron(g: pd.DataFrame, stride: int, min_radius: float):
    """g: rows for one body_id, indexed by node_id. Returns list of
    (x0,y0,z0,x1,y1,z1) segments connecting kept nodes to their nearest
    kept ancestor.

    Keeping every branch point turned out to explode the segment count
    (real dendritic arbors have huge numbers of fine terminal twigs — 1316
    DNs alone produced >1M segments that way). Since the goal here is "DNs
    and major tracts" rather than exact fine-branch topology, we instead
    pre-filter to thicker-than-`min_radius` skeleton nodes (the main
    trunks/tracts, not terminal dendrites) and then stride-decimate *that*
    — roots are always kept so every neuron still anchors a connected tree.
    """
    parent_of = dict(zip(g.index, g["parent_id"]))
    radius_of = dict(zip(g.index, g["radius"]))

    node_ids = list(g.index)
    keep = set()
    for i, nid in enumerate(node_ids):
        p = parent_of[nid]
        is_root = p == -1
        thick_enough = radius_of[nid] >= min_radius
        if is_root or (thick_enough and i % stride == 0):
            keep.add(nid)

    nearest_cache: dict[int, int | None] = {}

    def nearest_kept_ancestor(nid: int) -> int | None:
        chain = []
        cur = parent_of.get(nid, -1)
        while cur != -1 and cur not in keep:
            if cur in nearest_cache:
                result = nearest_cache[cur]
                break
            chain.append(cur)
            cur = parent_of.get(cur, -1)
        else:
            result = cur if cur != -1 else None
        for c in chain:
            nearest_cache[c] = result
        return result

    xyz = g[["x", "y", "z"]]
    segments = []
    for nid in keep:
        p = parent_of[nid]
        if p == -1:
            continue
        anc = p if p in keep else nearest_kept_ancestor(p)
        if anc is None or anc not in xyz.index:
            continue
        p0 = xyz.loc[anc]
        p1 = xyz.loc[nid]
        segments.append((p0.x, p0.y, p0.z, p1.x, p1.y, p1.z))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/dn_skeletons.parquet")
    ap.add_argument("--out-bin", default="data/dn_skeleton_segments.bin")
    ap.add_argument("--out-meta", default="data/dn_skeleton_meta.json")
    ap.add_argument("--stride", type=int, default=15,
                     help="keep every Nth thick-enough node (higher = fewer segments)")
    ap.add_argument("--min-radius-percentile", type=float, default=0.8,
                     help="only consider nodes at/above this radius percentile "
                          "(major tracts, not fine terminal dendrites)")
    # Must match dashboard.jsx's soma-point centering exactly so skeletons
    # and somas overlay correctly in the same scene.
    ap.add_argument("--scale", type=float, default=1 / 4000)
    ap.add_argument("--centroid", default=None,
                     help="'x,y,z' to center on instead of this batch's own mean — "
                          "REQUIRED to align with brain_graph.json's soma cloud and with "
                          "any other skeleton population processed separately, since each "
                          "batch's own centroid differs by neuron population (e.g. DNs vs. "
                          "optic-lobe neurons sit in very different parts of the brain). "
                          "Pass the mean soma [x,y,z] across brain_graph_nodes.json's nodes.")
    args = ap.parse_args()

    print(f"Loading {args.inp} ...")
    df = pd.read_parquet(args.inp)
    n_bodies = df["body_id"].nunique()
    print(f"{len(df):,} nodes across {n_bodies} neurons")

    min_radius = float(df["radius"].quantile(args.min_radius_percentile))
    print(f"min_radius (p{args.min_radius_percentile*100:.0f}): {min_radius:.2f}")

    if args.centroid:
        cx, cy, cz = (float(v) for v in args.centroid.split(","))
        print(f"centroid (explicit, shared): ({cx:.0f}, {cy:.0f}, {cz:.0f})  scale: {args.scale}")
    else:
        # Falls back to this batch's own mean when no shared centroid is
        # given — fine in isolation, but this population's centroid can sit
        # far from another population's (e.g. DNs vs. optic-lobe neurons),
        # so anything meant to overlay with brain_graph.json's soma cloud or
        # with another skeleton batch MUST pass --centroid explicitly.
        cx, cy, cz = float(df["x"].mean()), float(df["y"].mean()), float(df["z"].mean())
        print(f"centroid (this batch's own mean — NOT aligned with other batches): "
              f"({cx:.0f}, {cy:.0f}, {cz:.0f})  scale: {args.scale}")

    all_segments = []
    for i, (body_id, g) in enumerate(df.groupby("body_id", sort=False)):
        g = g.set_index("node_id")
        segs = decimate_neuron(g, args.stride, min_radius)
        all_segments.extend(segs)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_bodies} neurons -> {len(all_segments):,} segments so far")

    print(f"Total segments: {len(all_segments):,} ({len(all_segments) * 2:,} vertices)")

    arr = np.empty((len(all_segments), 6), dtype=np.float32)
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(all_segments):
        arr[i] = (
            (x0 - cx) * args.scale, -(z0 - cz) * args.scale, (y0 - cy) * args.scale,
            (x1 - cx) * args.scale, -(z1 - cz) * args.scale, (y1 - cy) * args.scale,
        )

    out_bin = Path(args.out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_bin)
    print(f"Wrote {out_bin} ({out_bin.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "n_segments": len(all_segments),
        "n_neurons": int(n_bodies),
        "stride": args.stride,
        "min_radius": min_radius,
        "centroid": [cx, cy, cz],
        "scale": args.scale,
        "format": "float32 flat [x0,y0,z0,x1,y1,z1] per segment, little-endian",
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
