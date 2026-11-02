#!/usr/bin/env python
"""E6 oracle: exact float64 late-inject-replace on ViT-B/16 (e5_late_inject_vit.late_inject in numpy, mirroring
perseus.examples.vit.weights.exact_blocks). The private patches S are dropped from attention for the first L - k
layers (every token runs, S keys masked), then the S rows are replaced by their raw embeddings and the last k layers
run with full attention. Returns the plain hidden state at the hand-over, the per-layer hidden states of the last k
layers, and the logits."""
from __future__ import annotations

import numpy as np

from e3_reference import _attn, _ln


def late_replace(store, x, S, k, n_layers=12):
    from scipy.special import erf
    from perseus.examples.gpt2 import weights as GW
    x = np.asarray(x, dtype=np.float64)                 # embedded rows (197, 768): raw embeddings
    T = x.shape[0]
    eps = float(store.config.get("layer_norm_eps", 1e-12))
    t = store.tensor
    S = np.asarray(sorted(int(s) for s in S), dtype=np.int64)
    tokS = S + 1
    isS = np.zeros(T, dtype=bool)
    isS[tokS] = True
    P = x.copy()
    hidden = []
    for blk in range(n_layers):
        if blk == n_layers - k:                           # the hand-over: raw embeddings replace the placeholders
            handover = P.copy()
            P[tokS] = x[tokS]
        kmask = None if blk >= n_layers - k else ~isS
        n = GW.block_names(blk)
        h = _ln(P, t(n["ln1_w"]), t(n["ln1_b"]), eps)
        q, kk, v = np.split(h @ t(n["qkv_w"]).T + t(n["qkv_b"]), 3, axis=1)
        o = _attn(q, kk, v, kmask)
        P = P + o @ t(n["out_w"]).T + t(n["out_b"])
        h2 = _ln(P, t(n["ln2_w"]), t(n["ln2_b"]), eps)
        u = h2 @ t(n["up_w"]).T + t(n["up_b"])
        u = 0.5 * u * (1.0 + erf(u / np.sqrt(2.0)))
        P = P + u @ t(n["down_w"]).T + t(n["down_b"])
        hidden.append(P.copy())
    if k == 0:
        handover = P.copy()
    return dict(handover=handover, hidden=hidden, tokS=tokS)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    from perseus.examples.vit import weights as V
    import e3_harness as H
    store = V.VitStore()
    rows = H.sample_rows(V)
    x = V.embed(store, rows)
    full = V.exact_blocks(store, x, 12)
    S = [(3 + i) * 14 + (3 + j) for i in range(4) for j in range(4)]
    for k in (12, 2, 1):
        r = late_replace(store, x, S, k)
        lg = V.logits(store, r["hidden"][-1])[0]; lf = V.logits(store, full[-1])[0]
        print(f"[oracle] k={k:2d}: max |h - full| {np.abs(r['hidden'][-1] - full[-1]).max():.3g}, argmax {int(lg.argmax())} vs full {int(lf.argmax())}, "
              f"|logit diff| {np.abs(lg - lf).max():.3g}")
