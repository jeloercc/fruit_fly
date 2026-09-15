#!/usr/bin/env python3
"""
fetch_brain.py — Extract the male-cns:v1.0 connectome from neuPrint and export
a lightweight brain_graph.json (nodes + edges) for run_simulation.py and the
React Three Fiber dashboard.

Descending Neurons (DNs) are identified from real neuPrint metadata: the
`superclass` property, which this dataset populates with 'descending_neuron'
/ 'descending_neuron_tbc' for VNC-projecting command neurons. (Note: neuron
`type` names starting with "DN" are NOT a reliable DN filter in this dataset —
e.g. DN1a/DN1p are circadian "Dorsal Neurons" in the central brain, not
descending neurons. superclass is the authoritative field.)

Resumable by design: nodes are fetched in per-superclass batches and edges in
fixed-size bodyId-chunk batches, each written to its own checkpoint file under
--checkpoint-dir. Re-running the script skips any batch whose checkpoint file
already exists, so a dropped connection just costs the in-flight batch.

Usage:
    python fetch_brain.py                                  # full dataset
    python fetch_brain.py --only-superclasses descending_neuron,vnc_motor \
                           --edge-chunk-size 100             # bounded pilot run
    python fetch_brain.py --skip-edges                       # nodes only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_brain")

# Superclass values actually present in male-cns:v1.0 as of this writing
# (queried live via `MATCH (n:Neuron) WHERE n.superclass IS NOT NULL RETURN
# DISTINCT n.superclass, count(n)`). Kept as an explicit list (rather than
# discovered at runtime) so batches are deterministic and resumable.
KNOWN_SUPERCLASSES = [
    "ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "visual_projection",
    "vnc_sensory", "ol_sensory", "cb_sensory", "ascending_neuron",
    "descending_neuron", "vnc_motor", "visual_centrifugal",
    "sensory_ascending", "cb_motor", "vnc_efferent", "cb_endocrine", "ENS",
    "vnc_tbc", "vnc_sensory_tbc", "vnc_endocrine", "cb_sensory_tbc",
    "sensory_descending", "efferent_ascending", "cb_efferent",
    "efferent_descending", "descending_neuron_tbc", "sensory_ascending_tbc",
    "visual_projection_tbc",
]

DN_SUPERCLASSES = {"descending_neuron", "descending_neuron_tbc"}

NODE_QUERY_TEMPLATE = """
MATCH (n:Neuron)
WHERE {where}
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance,
       n.class AS class, n.superclass AS superclass, n.somaSide AS somaSide,
       n.status AS status, n.somaLocation AS somaLocation,
       n.consensusNt AS consensusNt, n.predictedNt AS predictedNt,
       n.predictedNtConfidence AS predictedNtConfidence
"""


def load_env() -> dict:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    token = os.environ.get("NEUPRINT_TOKEN")
    dataset = os.environ.get("NEUPRINT_DATASET")
    if not token or not dataset:
        log.error("NEUPRINT_TOKEN / NEUPRINT_DATASET missing from .env")
        sys.exit(1)
    return {"token": token, "dataset": dataset}


def get_client(server: str, dataset: str, token: str):
    from neuprint import Client

    return Client(server, dataset=dataset, token=token)


def retry(fn, *, max_retries: int, label: str):
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - want to retry on any transient network error
            if attempt == max_retries:
                log.error("%s: giving up after %d attempts (%s)", label, attempt, e)
                raise
            log.warning(
                "%s: attempt %d/%d failed (%s) — retrying in %.1fs",
                label, attempt, max_retries, e, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def soma_xyz(soma_location) -> tuple[float | None, float | None, float | None]:
    if isinstance(soma_location, dict):
        coords = soma_location.get("coordinates")
        if coords is not None and len(coords) == 3:
            return float(coords[0]), float(coords[1]), float(coords[2])
    return None, None, None


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def fetch_nodes(client, checkpoint_dir: Path, superclasses: list[str],
                 max_retries: int, include_unclassified: bool,
                 classes: list[str] | None = None) -> pd.DataFrame:
    from neuprint import fetch_custom

    nodes_dir = checkpoint_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    batches = [(sc, f"n.superclass = '{sc}'") for sc in superclasses]
    # `class` is a finer-grained field than `superclass` for some real
    # populations — e.g. Kenyon cells and MBONs both share the huge
    # (32k-neuron) `cb_intrinsic` superclass, so scoping by superclass alone
    # would pull far more than intended. `--only-classes` scopes by the
    # `class` property instead, checkpointed the same way (prefixed
    # `class_` so it can't collide with a superclass name).
    for cls in (classes or []):
        batches.append((f"class_{cls}", f"n.class = '{cls}'"))
    if include_unclassified:
        # Only pull unlabeled bodies when doing a full-dataset export —
        # a scoped/pilot run (--only-superclasses) should stay scoped.
        batches.append(("_unclassified", "n.superclass IS NULL"))

    frames = []
    for name, where_clause in batches:
        ckpt_file = nodes_dir / f"{name}.parquet"
        if ckpt_file.exists():
            log.info("nodes[%s]: checkpoint found, skipping fetch", name)
            frames.append(pd.read_parquet(ckpt_file))
            continue

        log.info("nodes[%s]: fetching from neuPrint...", name)
        query = NODE_QUERY_TEMPLATE.format(where=where_clause)
        df = retry(
            lambda: fetch_custom(query, client=client),
            max_retries=max_retries,
            label=f"nodes[{name}]",
        )
        df.to_parquet(ckpt_file)
        log.info("nodes[%s]: %d neurons -> %s", name, len(df), ckpt_file.name)
        frames.append(df)

    nodes = pd.concat(frames, ignore_index=True)
    nodes = nodes.drop_duplicates(subset="bodyId").reset_index(drop=True)
    return nodes


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

def fetch_edges(client, checkpoint_dir: Path, body_ids: list[int],
                 min_weight: int, edge_chunk_size: int,
                 max_retries: int) -> pd.DataFrame:
    from neuprint import fetch_adjacencies

    edges_dir = checkpoint_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)

    body_ids = sorted(body_ids)
    chunks = [
        body_ids[i:i + edge_chunk_size]
        for i in range(0, len(body_ids), edge_chunk_size)
    ]
    log.info(
        "edges: %d source bodies in %d chunks of <=%d (min_total_weight=%d)",
        len(body_ids), len(chunks), edge_chunk_size, min_weight,
    )

    frames = []
    for i, chunk in enumerate(chunks):
        # Keyed by chunk *content* (first/last bodyId + size), not position:
        # the chunk boundaries depend on the full sorted body_ids list, which
        # changes if the run's scope changes (e.g. pilot -> full dataset).
        # A positional "chunk_00000.parquet" would silently be reused for a
        # completely different set of source bodies in that case.
        ckpt_file = edges_dir / f"edges_{chunk[0]}_{chunk[-1]}_{len(chunk)}.parquet"
        if ckpt_file.exists():
            frames.append(pd.read_parquet(ckpt_file))
            continue

        def do_fetch(chunk=chunk):
            _, conn_df = fetch_adjacencies(
                sources=chunk,
                targets=body_ids,
                min_total_weight=min_weight,
                omit_rois=True,
                properties=[],
                client=client,
            )
            return conn_df[["bodyId_pre", "bodyId_post", "weight"]]

        log.info("edges: chunk %d/%d (%d sources)...", i + 1, len(chunks), len(chunk))
        conn_df = retry(do_fetch, max_retries=max_retries, label=f"edges[chunk {i}]")
        conn_df.to_parquet(ckpt_file)
        log.info("edges: chunk %d/%d -> %d connections -> %s",
                  i + 1, len(chunks), len(conn_df), ckpt_file.name)
        frames.append(conn_df)

    if not frames:
        return pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def build_graph_json(nodes: pd.DataFrame, edges: pd.DataFrame, dataset: str) -> dict:
    node_list = []
    dn_count = 0
    # `class` is a Python keyword, so pandas.itertuples() can't expose it as
    # an attribute — iterate plain dict records instead.
    def clean(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    for row in nodes.to_dict(orient="records"):
        x, y, z = soma_xyz(row["somaLocation"])
        is_dn = row["superclass"] in DN_SUPERCLASSES
        if is_dn:
            dn_count += 1
        node_list.append({
            "id": int(row["bodyId"]),
            "type": clean(row["type"]),
            "instance": clean(row["instance"]),
            "class": clean(row["class"]),
            "superclass": clean(row["superclass"]),
            "is_dn": is_dn,
            "soma_side": clean(row["somaSide"]),
            "soma": [x, y, z],
            "nt": clean(row["consensusNt"]) or clean(row["predictedNt"]),
            "status": clean(row["status"]),
        })

    edge_list = [
        {"source": int(r.bodyId_pre), "target": int(r.bodyId_post), "weight": int(r.weight)}
        for r in edges.itertuples(index=False)
    ]

    return {
        "meta": {
            "dataset": dataset,
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "node_count": len(node_list),
            "edge_count": len(edge_list),
            "dn_count": dn_count,
            "dn_superclasses": sorted(DN_SUPERCLASSES),
        },
        "nodes": node_list,
        "edges": edge_list,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="neuprint.janelia.org")
    ap.add_argument("--out", default="data/brain_graph.json")
    ap.add_argument("--checkpoint-dir", default="data/checkpoints")
    ap.add_argument("--min-weight", type=int, default=3,
                     help="min_total_weight synapse threshold for an edge to be kept "
                          "(standard connectome-export noise filter)")
    ap.add_argument("--edge-chunk-size", type=int, default=300,
                     help="bodyIds per fetch_adjacencies batch (checkpointed individually)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--only-superclasses", default=None,
                     help="comma-separated subset of superclasses, for bounded/pilot runs")
    ap.add_argument("--only-classes", default=None,
                     help="comma-separated subset of the finer-grained `class` property "
                          "(e.g. Kenyon_Cell,MBON,DAN) — for populations too small to "
                          "isolate by --only-superclasses alone")
    ap.add_argument("--skip-edges", action="store_true")
    args = ap.parse_args()

    env = load_env()
    client = get_client(args.server, env["dataset"], env["token"])
    log.info("Connected to %s dataset=%s", args.server, env["dataset"])

    checkpoint_dir = Path(args.checkpoint_dir)
    scoped_run = args.only_superclasses is not None or args.only_classes is not None
    superclasses = (
        args.only_superclasses.split(",") if args.only_superclasses is not None else
        (KNOWN_SUPERCLASSES if args.only_classes is None else [])
    )
    classes = args.only_classes.split(",") if args.only_classes else None

    nodes = fetch_nodes(client, checkpoint_dir, superclasses, args.max_retries,
                         include_unclassified=not scoped_run, classes=classes)
    log.info("Total nodes: %d (DNs: %d)", len(nodes),
              (nodes["superclass"].isin(DN_SUPERCLASSES)).sum())

    if args.skip_edges:
        edges = pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "weight"])
    else:
        edges = fetch_edges(
            client, checkpoint_dir, nodes["bodyId"].tolist(),
            args.min_weight, args.edge_chunk_size, args.max_retries,
        )
    log.info("Total edges: %d", len(edges))

    graph = build_graph_json(nodes, edges, env["dataset"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f)
    log.info("Wrote %s (%d nodes, %d edges, %d DNs)",
              out_path, graph["meta"]["node_count"], graph["meta"]["edge_count"],
              graph["meta"]["dn_count"])


if __name__ == "__main__":
    main()
