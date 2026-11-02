#!/usr/bin/env python
"""E3 stage 2 test: the selective ViT prefill on the fake session vs the exact two-stream oracle, block by block.

  python selective/e3_test_xpt.py --layers 2 --box 4      # 4x4 box of sensitive patches (17 encrypted tokens)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H
import e3_reference as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--box", type=int, default=4, help="side of the sensitive box (patches), at --at")
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--file", default=None, help="an image file instead of the --image sample")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--weights", default=None, help="a local HF checkpoint dir (fine-tuned weights) instead of the hub model")
    a = ap.parse_args()
    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=a.layers, capture=a.capture, cls=SelectivePrefill, weights=a.weights)
    V, store = h.V, h.store
    rows = V.patch_rows(V.pixels(a.file)) if a.file else H.sample_rows(V, a.image)
    x = V.embed(store, rows)
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    ref = R.two_stream(store, x, S, a.layers)
    full = V.exact_blocks(store, x, a.layers)
    E_idx = ref["E_idx"]
    n_e = len(E_idx)
    print(f"[xpt] {n_e} encrypted tokens (CLS + {len(S)} patches), {len(ref['NS'])} plaintext keys, {a.layers} layers",
          flush=True)
    errs = []

    def on_block(b, Xs, s):
        hE = V.unscale(h.model.hidden(Xs, h.model.last_ch))[:n_e]
        want = ref["E_hidden"][b][:n_e]
        err = np.abs(hE - want).max(1)
        errs.append(err.max())
        print(f"[xpt] block {b} {s:.1f}s maxerr={err.max():.3g} (tok {int(err.argmax())}) |x|={np.abs(want).max():.3g} "
              f"bts={int(h.inf.fhe.total_bootstraps)}", flush=True)
    t0 = time.time()
    Xs, ch, _ = h.model.prefill_selective(rows[E_idx], E_idx, ref["P_kv"], on_block=on_block)
    hE = V.unscale(h.model.hidden(Xs, ch))[:n_e]
    if a.layers == 12:
        lg = V.logits(store, hE)[0]
        lo = V.logits(store, ref["E_hidden"][-1])[0]
        lf = V.logits(store, full[-1])[0]
        sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        kl = float((sm(lo) * (np.log(sm(lo) + 1e-30) - np.log(sm(lg) + 1e-30))).sum())
        print(f"[xpt] logits: encrypted argmax {int(lg.argmax())}, oracle {int(lo.argmax())}, full model {int(lf.argmax())}; "
              f"KL(oracle || encrypted) {kl:.3g}; |logit diff| max {np.abs(lg - lo).max():.3g}", flush=True)
    print(f"[xpt] done in {time.time() - t0:.0f}s, bts={int(h.inf.fhe.total_bootstraps)}, worst block maxerr {max(errs):.3g}",
          flush=True)
    h.model.close()
    h.sess.close()


if __name__ == "__main__":
    main()
