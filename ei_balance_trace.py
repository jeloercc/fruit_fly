#!/usr/bin/env python3
"""
ei_balance_trace.py — read-only connectivity analysis. Changes nothing,
imports nothing from the live pipeline, and is imported by nothing.

Question: along the food-odour ORN -> ... -> premotor-DN pathway, where
does excitatory drive get cancelled by inhibition?

Method and its limits, stated up front so the numbers are not over-read:
this is a LINEAR signed propagation over the real weight matrix
(W[post,pre] = sign(nt_pre) * synapse_count * W_SYN, exactly as BrainLIF
builds it). It deliberately ignores spike thresholds, refractoriness,
synaptic delay and saturation, so it describes how much signed current the
WIRING delivers per hop — not what the spiking network will actually do.
A layer flagged here as net-inhibitory is a structural candidate for
gating; confirming it would require a causal knockout (silencing that
population and re-measuring), which this script does NOT do.

Each hop's vector is L1-normalised so the reported E/I ratios are
scale-invariant (otherwise repeated multiplication by a matrix with
supra-threshold entries explodes and the numbers mean nothing).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import scipy.sparse as sp

GRAPH = "data/brain_graph.json"
W_SYN = 0.275
NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "octopamine": 1.0,
           "serotonin": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
FOOD_GLOM = ("ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_VA2", "ORN_VM2")
N_HOPS = 4


def main():
    g = json.load(open(GRAPH))
    nodes, edges = g["nodes"], g["edges"]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)

    ty = np.array([nd.get("type") or "" for nd in nodes])
    sc = np.array([nd.get("superclass") or "" for nd in nodes])
    nt = np.array([(nd.get("nt") or "").lower() for nd in nodes])
    sign_of = np.array([NT_SIGN.get(x, 1.0) for x in nt])

    r, c, v = [], [], []
    for e in edges:
        s, t, w = e["source"], e["target"], e["weight"]
        if s in idx and t in idx:
            r.append(idx[t]); c.append(idx[s])
            v.append(NT_SIGN.get(nt[idx[s]], 1.0) * w * W_SYN)
    W = sp.csr_matrix((v, (r, c)), shape=(n, n))
    Wpos = W.maximum(0)
    Wneg = W.minimum(0)

    food = np.isin(ty, FOOD_GLOM)
    motor = sc == "vnc_motor"
    is_dn = np.array([bool(nd.get("is_dn")) for nd in nodes])
    premotor = np.zeros(n, dtype=bool)
    premotor[np.unique(W[motor].indices)] = True
    premotor &= is_dn

    print(f"source : food ORNs (DM1/DM2/DM4/VA2/VM2) = {int(food.sum())} cells")
    print(f"target : premotor DNs (direct DN->vnc_motor) = {int(premotor.sum())} cells")
    print(f"matrix : {W.shape}, nnz={W.nnz}, {100 * (W.data < 0).mean():.1f}% inhibitory entries\n")

    x = food.astype(float)
    x /= x.sum()
    print(f"{'hop':<5}{'E current':>13}{'I current':>13}{'net':>12}{'E/I':>8}"
          f"{'reached':>10}{'  dominant populations (by |current|)'}")
    print("-" * 110)
    for hop in range(1, N_HOPS + 1):
        # np.asarray(...).ravel(): scipy's spmatrix classes carry np.matrix
        # semantics, so a matvec can come back 2-D and silently break the
        # scalar arithmetic/formatting below.
        exc = np.asarray(Wpos @ x).ravel()
        inh = np.asarray(Wneg @ x).ravel()
        net = exc + inh                      # per-neuron signed current (vector)
        E, I = float(exc.sum()), float(-inh.sum())
        net_total = E - I                    # scalar, for the summary row
        reached = int((np.abs(exc) + np.abs(inh) > 0).sum())
        contrib = np.abs(exc) + np.abs(inh)
        top = np.argsort(contrib)[::-1][:3]
        lbl = ", ".join(f"{sc[i] or '?'}/{ty[i] or '?'}" for i in top)
        print(f"{hop:<5}{E:>13.2f}{I:>13.2f}{net_total:>12.2f}{(E / I if I else float('inf')):>8.2f}"
              f"{reached:>10}  {lbl[:58]}")

        # premotor-DN slice of this hop
        pE, pI = float(exc[premotor].sum()), float(-inh[premotor].sum())
        if pE or pI:
            print(f"{'':5}{'-> onto premotor DNs:':<26} E={pE:9.3f}  I={pI:9.3f}  "
                  f"net={pE - pI:+9.3f}  E/I={(pE / pI if pI else float('inf')):.2f}")
        # RECTIFY. Propagating the signed `net` directly is wrong: a
        # negative entry passed through an inhibitory (negative) synapse
        # produces POSITIVE current, so "inhibitory current" flips sign and
        # the E/I decomposition becomes meaningless from hop 2 onward (an
        # earlier version of this script reported a spurious net-inhibitory
        # hop 3 and a negative I at hop 4 for exactly that reason). A
        # hyperpolarised neuron does not fire negatively — it just does not
        # fire — so only positively-driven neurons pass signal on.
        x = np.maximum(net, 0.0)
        s = np.abs(x).sum()
        if s == 0:
            print("signal fully extinguished")
            break
        x /= s

    # ---- who delivers the inhibition onto the premotor DNs? --------------
    print("\n" + "=" * 78)
    print("INHIBITORY INPUT ONTO THE 981 PREMOTOR DNs (whole-graph, unweighted by hop)")
    print("=" * 78)
    sub = W[premotor].tocoo()
    neg = sub.data < 0
    pre_idx = sub.col[neg]
    weight = -sub.data[neg]
    tot_inh = float(weight.sum())
    tot_exc = float(sub.data[sub.data > 0].sum())
    print(f"total excitatory input: {tot_exc:12.1f} mV-equivalent")
    print(f"total inhibitory input: {tot_inh:12.1f} mV-equivalent   (E/I = {tot_exc / tot_inh:.2f})\n")

    by_sc, by_ty = Counter(), Counter()
    for i, w in zip(pre_idx, weight):
        by_sc[sc[i] or "?"] += w
        by_ty[ty[i] or "?"] += w
    print("top inhibitory SOURCES by superclass:")
    for k, w in by_sc.most_common(6):
        print(f"   {k:<22}{w:12.1f}  ({100 * w / tot_inh:5.1f}% of all inhibition)")
    print("\ntop inhibitory SOURCES by cell type:")
    for k, w in by_ty.most_common(10):
        print(f"   {k:<22}{w:12.1f}  ({100 * w / tot_inh:5.1f}%)")


if __name__ == "__main__":
    main()
