#!/usr/bin/env python3
"""
fetch_skeletons.py — Fetch 3D neuron skeletons (full morphology, not just
soma points) for male-cns:v1.0 via navis, per the documented method at
https://male-cns.janelia.org/download/#__tabbed_1_2 :

    import navis.interfaces.neuprint as neu
    skels = neu.fetch_skeletons(neu.NeuronCriteria(...))

Each skeleton node carries [body_id, node_id, x, y, z, radius, parent_id] —
real reconstructed morphology at ~8nm resolution. A single DN's skeleton
already runs ~16,000 nodes, so this is far heavier than the soma-point graph
in brain_graph.json: fetching skeletons for the *entire* ~170k-neuron
dataset would be tens of GB. Default scope is therefore Descending Neurons
only (superclass in {descending_neuron, descending_neuron_tbc}) — the
neurons this project's brain->CPG bridge actually reads out — with
--superclasses/--body-ids to widen or narrow that.

Resumable: fetched in small bodyId batches, one checkpoint parquet per
batch (named by content, not position), so a dropped connection only costs
the in-flight batch.

Usage:
    python fetch_skeletons.py                          # all DNs
    python fetch_skeletons.py --body-ids 12781,556329   # specific bodies
    python fetch_skeletons.py --superclasses vnc_motor  # widen scope
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fetch_brain import DN_SUPERCLASSES, load_env, retry, get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("fetch_skeletons")


def dn_body_ids(client, superclasses: set[str]) -> list[int]:
    from neuprint import fetch_custom

    clauses = " OR ".join(f"n.superclass = '{sc}'" for sc in superclasses)
    query = f"MATCH (n:Neuron) WHERE {clauses} RETURN n.bodyId AS bodyId"
    df = fetch_custom(query, client=client)
    return sorted(df["bodyId"].tolist())


def fetch_skeleton_batch(client, body_ids: list[int]) -> pd.DataFrame:
    import navis.interfaces.neuprint as neu

    skels = neu.fetch_skeletons(list(body_ids), client=client)
    frames = []
    for n in skels:
        df = n.nodes[["node_id", "x", "y", "z", "radius", "parent_id"]].copy()
        df.insert(0, "body_id", n.id)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["body_id", "node_id", "x", "y", "z", "radius", "parent_id"])
    out = pd.concat(frames, ignore_index=True)
    # float32 is plenty of precision for visualization/geometry at this scale
    # and roughly halves storage vs. navis' default float64.
    for col in ["x", "y", "z", "radius"]:
        out[col] = out[col].astype("float32")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="neuprint.janelia.org")
    ap.add_argument("--out", default="data/dn_skeletons.parquet")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints/skeletons")
    ap.add_argument("--batch-size", type=int, default=20,
                     help="bodyIds per navis fetch_skeletons call (checkpointed individually)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--superclasses", default=None,
                     help="comma-separated superclasses to fetch skeletons for "
                          "(default: descending_neuron,descending_neuron_tbc)")
    ap.add_argument("--body-ids", default=None,
                     help="comma-separated explicit bodyIds, overrides --superclasses")
    args = ap.parse_args()

    env = load_env()
    client = get_client(args.server, env["dataset"], env["token"])
    log.info("Connected to %s dataset=%s", args.server, env["dataset"])

    if args.body_ids:
        body_ids = sorted(int(b) for b in args.body_ids.split(","))
    else:
        superclasses = (
            set(args.superclasses.split(",")) if args.superclasses else DN_SUPERCLASSES
        )
        log.info("Resolving bodyIds for superclasses=%s ...", superclasses)
        body_ids = dn_body_ids(client, superclasses)
    log.info("Fetching skeletons for %d neurons", len(body_ids))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    batches = [body_ids[i:i + args.batch_size] for i in range(0, len(body_ids), args.batch_size)]
    frames = []
    for i, batch in enumerate(batches):
        ckpt_file = ckpt_dir / f"skel_{batch[0]}_{batch[-1]}_{len(batch)}.parquet"
        if ckpt_file.exists():
            log.info("batch %d/%d: checkpoint found, skipping", i + 1, len(batches))
            frames.append(pd.read_parquet(ckpt_file))
            continue

        log.info("batch %d/%d: fetching %d skeletons...", i + 1, len(batches), len(batch))
        df = retry(
            lambda batch=batch: fetch_skeleton_batch(client, batch),
            max_retries=args.max_retries,
            label=f"skeletons[batch {i}]",
        )
        df.to_parquet(ckpt_file)
        log.info("batch %d/%d: %d nodes across %d neurons -> %s",
                  i + 1, len(batches), len(df), df["body_id"].nunique(), ckpt_file.name)
        frames.append(df)

    if not frames:
        log.warning("No skeletons fetched.")
        return
    full = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path)
    log.info("Wrote %s: %d nodes across %d neurons (%.1f MB)",
              out_path, len(full), full["body_id"].nunique(),
              out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
