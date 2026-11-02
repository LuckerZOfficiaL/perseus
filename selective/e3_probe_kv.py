#!/usr/bin/env python
"""E3 stage 1: what do Perseus's K / V attention pages hold, in model units?

Runs block 0 of the fake ViT on one image, intercepts the pages handed to the paged CCMM attention, decrypts them
back to (token, feature) rows, and matches every page column against the exact numpy q / k / v columns of the same
layer. Prints the inferred column permutation and the scale, which is what the plaintext stream must reproduce.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H


def main():
    h = H.setup(layers=1)
    V, store, B = h.V, h.store, h.B
    from perseus.impl import ccmm_mha_paged as MP
    from perseus.impl import blocked as BL
    grabbed = {}
    orig = MP.attention

    def wrapped(rt, Q, K, Vv, cfg, ch, layer=0, h_real=12):
        if layer == 0 and "K" not in grabbed:
            for name, pages in (("Q", Q), ("K", K), ("V", Vv)):
                grabbed[name] = BL.decrypt_blocked(rt, pages, ch, B, B.d.hid)
            grabbed["ch"] = ch
        return orig(rt, Q, K, Vv, cfg, ch, layer=layer, h_real=h_real)
    MP.attention = wrapped

    rows = H.sample_rows(V)
    Xs, ch, _ = h.model.prefill(rows)
    print(f"[probe] pages decrypted: {[(k, v.shape) for k, v in grabbed.items() if k != 'ch']}", flush=True)

    # exact numpy layer-0 q / k / v (model units)
    from perseus.examples.gpt2 import weights as GW
    x0 = V.embed(store, rows)
    n = GW.block_names(0)
    t = store.tensor
    eps = float(store.config.get("layer_norm_eps", 1e-12))
    mu = x0.mean(1, keepdims=True)
    hln = (x0 - mu) / np.sqrt(((x0 - mu) ** 2).mean(1, keepdims=True) + eps) * t(n["ln1_w"]) + t(n["ln1_b"])
    q, k, v = np.split(hln @ t(n["qkv_w"]).T + t(n["qkv_b"]), 3, axis=1)
    T = len(rows)
    out = {}
    for name, ref in (("Q", q), ("K", k), ("V", v)):
        got = grabbed[name][:T]                      # (T, 1024) decrypted page rows
        # match each page column to the best model column by correlation over tokens
        gc = got - got.mean(0); rc = ref - ref.mean(0)
        gn = np.linalg.norm(gc, axis=0) + 1e-12; rn = np.linalg.norm(rc, axis=0) + 1e-12
        corr = (gc.T @ rc) / gn[:, None] / rn[None, :]          # (1024, 768)
        best = corr.argmax(1); bestc = corr.max(1)
        live = gn > 1e-6 * gn.max()
        scale = np.array([np.dot(got[:, c], ref[:, best[c]]) / (np.dot(ref[:, best[c]], ref[:, best[c]]) + 1e-12)
                          for c in range(got.shape[1])])
        resid = np.array([np.abs(got[:, c] - scale[c] * ref[:, best[c]]).max() / (np.abs(ref[:, best[c]]).max() + 1e-12)
                          for c in range(got.shape[1])])
        print(f"[{name}] live page columns {live.sum()} / {got.shape[1]}; corr>0.99 on {(bestc[live] > 0.99).sum()}; "
              f"scale median {np.median(scale[live]):.4f} (min {scale[live].min():.4f} max {scale[live].max():.4f}); "
              f"relative residual median {np.median(resid[live]):.3g} max {resid[live].max():.3g}")
        # the permutation pattern: page column c -> model column best[c]; describe as (head, dim) of the model column
        hd = best // 64; dm = best % 64
        print(f"      page col 0..7 -> model (head, dim): {[(int(hd[c]), int(dm[c])) for c in range(8)]}")
        print(f"      page col 64..67 -> {[(int(hd[c]), int(dm[c])) for c in range(64, 68)]}; "
              f"col 512..515 -> {[(int(hd[c]), int(dm[c])) for c in range(512, 516)]}")
        out[name] = dict(best=best, scale=scale, live=live)
    # is the permutation the same for Q, K, V, and does it equal a simple formula?
    perm = out["K"]["best"]
    liveK = out["K"]["live"]
    same = all((out[n]["best"][liveK] == perm[liveK]).all() for n in ("Q", "V"))
    print(f"[perm] Q, K, V share one permutation: {same}")
    c = np.arange(B.d.hid)
    cands = {"identity (h*64+j)": c, "interleaved (j*H+h -> h*64+j)": None}
    # candidate from Blocks.head_perm: page index = head_perm[r] for interleaved feature r = j*H + h
    hp = B.head_perm()
    inv = np.full(B.d.hid, -1); inv[hp] = np.arange(B.d.hid)       # page col -> interleaved r
    r = inv
    H_, hreal = B.d.H, B.d.H_real
    j, hh = r // H_, r % H_
    model_col = np.where((r >= 0) & (hh < hreal), hh * 64 + j, -1)
    cands["interleaved (j*H+h -> h*64+j)"] = model_col
    for nm, cand in cands.items():
        ok = (cand[liveK] == perm[liveK]).mean()
        print(f"[perm] matches '{nm}' on {ok * 100:.1f}% of live columns")
    np.savez(os.path.expanduser("~/data/selective-fhe/e3/probe_kv_layer0.npz"), perm=perm, live=liveK,
             scaleK=out["K"]["scale"], scaleV=out["V"]["scale"], scaleQ=out["Q"]["scale"])
    h.model.close(); h.sess.close()


if __name__ == "__main__":
    main()
