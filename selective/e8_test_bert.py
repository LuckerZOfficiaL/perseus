#!/usr/bin/env python
"""E8 test: two-stream BERT, one or several SST-2 sentences' encrypted tokens on ONE page, on the fake session, against
each sentence's exact two-stream oracle (e8_reference), block by block and at the logits.

  python selective/e8_test_bert.py --idx 0,1,2 [--layers 12] [--frac 0.3] [--capture DIR] [--weights EXPORT_DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e8_harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default="0", help="SST-2 validation indices, one per input on the page")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-e-cap", type=int, default=10)
    ap.add_argument("--chunk0", type=int, default=None, help="inputs on chunk 0 (default all); the rest on chunk 1")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--private", default=None, help="detector file (e11_ner_sst2.py): private tokens instead of the span")
    ap.add_argument("--split", default="validation", help="SST-2 split of the --idx sentences (calibration: train)")
    a = ap.parse_args()
    h = H.setup(layers=a.layers, capture=a.capture, weights=a.weights, n_e_cap=a.n_e_cap)
    D, st, cl, m = h.D, h.store, h.cl, h.model
    idx = [int(v) for v in a.idx.split(",")]
    priv = H.load_private(a.private, a.split) if a.private else None
    ins = H.sentence_inputs(D, st, cl, idx, a.frac, k_max=a.k_max, n_layers=a.layers, split=a.split, private=priv)
    rows, pos, layout, P_kv = m.page([x["rows_e"] for x in ins], [x["pos"] for x in ins], [x["ref"]["P_kv"] for x in ins],
                                     a.chunk0)
    r0 = m.first_rows(layout)
    print(f"[e8] {len(ins)} sentences on one page, layout (lane, encrypted, plaintext keys) {layout}, {a.layers} layers",
          flush=True)
    errs = []

    def on_block(b, Xs, s):
        hE = m.hidden(Xs, m.last_ch)
        e = [np.abs(hE[s0:s0 + m.entry(lx)[0]] - x["ref"]["E_hidden"][b]).max() for s0, lx, x in zip(r0, layout, ins)]
        errs.append(max(e))
        print(f"[e8] block {b} {s:.1f}s maxerr per input {' '.join(f'{v:.3g}' for v in e)} "
              f"|x|={max(np.abs(x['ref']['E_hidden'][b]).max() for x in ins):.3g} bts={int(h.inf.fhe.total_bootstraps)}",
              flush=True)
    t0 = time.time()
    Xs, ch, _ = m.prefill_selective(rows, pos, layout, P_kv, on_block=on_block)
    hE = m.hidden(Xs, ch)
    if a.layers == 12:
        for s0, x in zip(r0, ins):
            lg = D.head(hE[s0], cl)
            lo = x["ref"]["logits"]
            print(f"[e8] sentence {x['i']} (n={x['n']}, |S|={len(x['S'])}): encrypted {np.round(lg, 4)} argmax "
                  f"{int(lg.argmax())}, oracle {int(lo.argmax())}, full model {int(x['full'].argmax())}, label {x['label']}; "
                  f"KL(oracle || encrypted) {D.kl2(lg, lo):.3g}; |logit diff| max {np.abs(lg - lo).max():.3g}", flush=True)
    print(f"[e8] done in {time.time() - t0:.0f}s, bts={int(h.inf.fhe.total_bootstraps)}, worst block maxerr {max(errs):.3g}",
          flush=True)
    m.close()
    h.sess.close()


if __name__ == "__main__":
    main()
