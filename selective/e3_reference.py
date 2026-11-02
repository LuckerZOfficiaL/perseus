#!/usr/bin/env python
"""E3 stage 0: exact float64 two-stream ViT-B/16 (the one-way rule, "drop" placeholder), the oracle for the encrypted
implementation. Mirrors perseus.examples.vit.weights.exact_blocks.

Given the embedded rows x (197, 768) and the sensitive patch set S (patch indices 0..195, token p+1):
  P stream: the standard ViT on all 197 tokens with the S tokens removed from every attention (keys masked).
            Its per-layer K / V on the non-sensitive patch tokens NS are what the server hands to the encrypted stream.
  E stream: tokens [CLS] + S, each layer attending to its own keys and to P's NS keys of the same layer.
Returns per-layer E hidden states, per-layer P keys/values on NS (model units), and the logits of E's CLS.
S = all patches and S = none both reproduce exact_blocks / the full model.
"""
from __future__ import annotations

import numpy as np


def _ln(h, g, b, eps):
    mu = h.mean(axis=1, keepdims=True)
    return (h - mu) / np.sqrt(((h - mu) ** 2).mean(axis=1, keepdims=True) + eps) * g + b


def _attn(q, k, v, kmask=None):
    """Multi-head attention (12 heads of 64) of queries q against keys/values k, v; kmask (len k) True = usable."""
    o = np.zeros_like(q)
    for hh in range(12):
        sl = slice(hh * 64, (hh + 1) * 64)
        s = q[:, sl] @ k[:, sl].T / 8.0
        if kmask is not None:
            s = np.where(kmask[None, :], s, -np.inf)
        p = np.exp(s - s.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        o[:, sl] = p @ v[:, sl]
    return o


def two_stream(store, x, S, n_layers=12, record=None):
    """record(layer, name, rows): optional tap of E-stream intermediates in model units."""
    from scipy.special import erf
    from perseus.examples.gpt2 import weights as GW
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    eps = float(store.config.get("layer_norm_eps", 1e-12))
    t = store.tensor
    S = np.asarray(sorted(int(s) for s in S), dtype=np.int64)
    tokS = S + 1
    E_idx = np.concatenate([[0], tokS])                                     # E stream tokens
    isS = np.zeros(T, dtype=bool); isS[tokS] = True
    NS = np.array([i for i in range(1, T) if not isS[i]], dtype=np.int64)  # plaintext patch tokens E reads
    pmask = ~isS                                                             # P keys usable by P (S dropped)
    P = x.copy()
    E = x[E_idx].copy()
    E_hidden, P_kv = [], []
    for blk in range(n_layers):
        n = GW.block_names(blk)
        W = t(n["qkv_w"]).T; bq = t(n["qkv_b"])
        # P stream (plaintext, exact)
        hP = _ln(P, t(n["ln1_w"]), t(n["ln1_b"]), eps)
        qP, kP, vP = np.split(hP @ W + bq, 3, axis=1)
        # E stream
        if record:
            record(blk, "ln1_in", E)                                     # the encrypted stream's LayerNorm inputs
        hE = _ln(E, t(n["ln1_w"]), t(n["ln1_b"]), eps)
        qE, kE, vE = np.split(hE @ W + bq, 3, axis=1)
        if record:
            record(blk, "ln1", hE); record(blk, "q", qE); record(blk, "k", kE); record(blk, "v", vE)
        Kcat = np.vstack([kE, kP[NS]]); Vcat = np.vstack([vE, vP[NS]])
        oE = _attn(qE, Kcat, Vcat)
        oP = _attn(qP, kP, vP, pmask)
        if record:
            record(blk, "attn", oE)
        P = P + oP @ t(n["out_w"]).T + t(n["out_b"])
        E = E + oE @ t(n["out_w"]).T + t(n["out_b"])
        for X, tag in ((P, None), (E, "mlp")):
            pass
        hP2 = _ln(P, t(n["ln2_w"]), t(n["ln2_b"]), eps)
        uP = hP2 @ t(n["up_w"]).T + t(n["up_b"]); uP = 0.5 * uP * (1.0 + erf(uP / np.sqrt(2.0)))
        P = P + uP @ t(n["down_w"]).T + t(n["down_b"])
        if record:
            record(blk, "ln2_in", E)
        hE2 = _ln(E, t(n["ln2_w"]), t(n["ln2_b"]), eps)
        uE = hE2 @ t(n["up_w"]).T + t(n["up_b"]); uE = 0.5 * uE * (1.0 + erf(uE / np.sqrt(2.0)))
        E = E + uE @ t(n["down_w"]).T + t(n["down_b"])
        E_hidden.append(E.copy())
        P_kv.append((kP[NS].copy(), vP[NS].copy()))
    return dict(E_idx=E_idx, NS=NS, E_hidden=E_hidden, P_kv=P_kv)


def logits(store, rows):
    from perseus.examples.vit import weights as V
    return V.logits(store, rows)


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
    for S, name in ((range(196), "all patches"), ([], "no patches"), ([5, 17, 100], "3 patches")):
        r = two_stream(store, x, S)
        err = np.abs(r["E_hidden"][-1] - full[-1][r["E_idx"]]).max()
        lg = logits(store, r["E_hidden"][-1])[0]; lf = logits(store, full[-1])[0]
        print(f"[oracle] S = {name:11s}: E tokens {len(r['E_idx'])}, max |E - full| at layer 12 = {err:.3g}, "
              f"argmax E {int(lg.argmax())} vs full {int(lf.argmax())}, |logit diff| max {np.abs(lg - lf).max():.3g}")
