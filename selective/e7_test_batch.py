#!/usr/bin/env python
"""E7 test: several inputs' encrypted two-stream tokens on ONE page (route B), on the fake session, against each
input's exact two-stream oracle (e3_reference), block by block and at the logits.

  python selective/e7_test_batch.py --inputs 3 --layers 12 --box 4 [--capture DIR] [--weights DIR]
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
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--box", type=int, default=4)
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--images", default="0,1,2", help="sample image indices (H.sample_rows), one per input")
    ap.add_argument("--files", default=None, help="image files (comma-separated), one per input, instead of --images")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--chunk0", type=int, default=None, help="inputs on chunk 0 (default all); the rest on chunk 1 (E10)")
    a = ap.parse_args()
    os.environ.setdefault("SEL_ROUTE", "pcmm")
    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=a.layers, capture=a.capture, cls=SelectivePrefill, weights=a.weights)
    V, store = h.V, h.store
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    srcs = a.files.split(",")[:a.inputs] if a.files else [int(v) for v in a.images.split(",")][:a.inputs]
    assert len(srcs) == a.inputs
    ins = []
    for i in srcs:
        rows = V.patch_rows(V.pixels(i)) if isinstance(i, str) else H.sample_rows(V, i)
        x = V.embed(store, rows)
        ref = R.two_stream(store, x, S, a.layers)
        ins.append(dict(rows=rows, ref=ref, full=V.exact_blocks(store, x, a.layers)))
    E_idx = ins[0]["ref"]["E_idx"]
    n_e = len(E_idx)
    print(f"[e7] {a.inputs} inputs x {n_e} encrypted tokens on one page, {len(ins[0]['ref']['NS'])} plaintext keys each, "
          f"{a.layers} layers", flush=True)
    errs = []

    def on_block(b, Xs, s):
        hE = V.unscale(h.model.hidden(Xs, h.model.last_ch))
        e = []
        for q, im in enumerate(ins):
            want = im["ref"]["E_hidden"][b][:n_e]
            e.append(np.abs(hE[q * n_e:(q + 1) * n_e] - want).max())
        errs.append(max(e))
        print(f"[e7] block {b} {s:.1f}s maxerr per input {' '.join(f'{v:.3g}' for v in e)} "
              f"|x|={np.abs(ins[0]['ref']['E_hidden'][b]).max():.3g} bts={int(h.inf.fhe.total_bootstraps)}", flush=True)
    t0 = time.time()
    h.model.chunk0 = a.chunk0
    Xs, ch, _ = h.model.prefill_selective_batch([im["rows"][E_idx] for im in ins], [E_idx] * a.inputs,
                                                [im["ref"]["P_kv"] for im in ins], on_block=on_block)
    hE = V.unscale(h.model.hidden(Xs, ch))
    if a.layers == 12:
        sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        for q, im in enumerate(ins):
            lg = V.logits(store, hE[q * n_e:(q + 1) * n_e])[0]
            lo = V.logits(store, im["ref"]["E_hidden"][-1])[0]
            lf = V.logits(store, im["full"][-1])[0]
            kl = float((sm(lo) * (np.log(sm(lo) + 1e-30) - np.log(sm(lg) + 1e-30))).sum())
            print(f"[e7] input {q}: encrypted argmax {int(lg.argmax())}, oracle {int(lo.argmax())}, full model "
                  f"{int(lf.argmax())}; KL(oracle || encrypted) {kl:.3g}; |logit diff| max {np.abs(lg - lo).max():.3g}",
                  flush=True)
    print(f"[e7] done in {time.time() - t0:.0f}s, bts={int(h.inf.fhe.total_bootstraps)}, worst block maxerr {max(errs):.3g}",
          flush=True)
    h.model.close()
    h.sess.close()


if __name__ == "__main__":
    main()
