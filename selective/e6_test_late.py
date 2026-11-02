#!/usr/bin/env python
"""E6 test: late-inject-replace on the fake session vs the exact oracle (e6_reference), per encrypted block.

  python selective/e6_test_late.py --k 2 --box 4 [--capture DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H
import e6_reference as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--box", type=int, default=4)
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--at2", default=None, help="a second private box (row,col), e.g. in the second 128-token page")
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--file", default=None, help="an image file instead of the --image sample")
    ap.add_argument("--patches", default=None, help="the private patches (comma-separated) instead of --box / --at")
    ap.add_argument("--capture", default=None)
    ap.add_argument("--weights", default=None, help="a local HF checkpoint dir (fine-tuned weights) instead of the hub model")
    a = ap.parse_args()
    from sfhe.vit_late import LatePrefill
    h = H.setup(layers=12, capture=a.capture, cls=LatePrefill, weights=a.weights)
    V, store = h.V, h.store
    rows = V.patch_rows(V.pixels(a.file)) if a.file else H.sample_rows(V, a.image)
    x = V.embed(store, rows)
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    if a.at2:
        r2, c2 = (int(v) for v in a.at2.split(","))
        S = sorted(set(S) | {(r2 + i) * 14 + (c2 + j) for i in range(a.box) for j in range(a.box)})
    if a.patches:
        S = sorted(int(v) for v in a.patches.split(","))
    ref = R.late_replace(store, x, S, a.k)
    full = V.exact_blocks(store, x, 12)
    L = 12
    tokS = ref["tokS"]
    page = np.zeros_like(rows)
    page[tokS] = rows[tokS]                                  # the client's page: private pixel rows only
    print(f"[late] k={a.k}: {len(S)} private patches, hand-over at layer {L - a.k}", flush=True)
    errs = []

    def on_block(i, Xs, s):
        hE = V.unscale(h.model.hidden(Xs, h.model.last_ch))[:197]
        if i == 0:
            want = ref["handover"].copy(); want[tokS] = x[tokS]      # the hand-over state: plain rows + raw private rows
            tag = "handover"
        else:
            want = ref["hidden"][L - a.k + i - 1]
            tag = f"block {L - a.k + i - 1}"
        err = np.abs(hE - want).max(1)
        errs.append(err.max())
        print(f"[late] {tag} {s:.1f}s maxerr={err.max():.3g} (tok {int(err.argmax())}) |x|={np.abs(want).max():.3g} "
              f"bts={int(h.inf.fhe.total_bootstraps)}", flush=True)
    t0 = time.time()
    Xs, ch, _ = h.model.prefill_late(page, S, ref["handover"], a.k, on_block=on_block)
    hE = V.unscale(h.model.hidden(Xs, ch))[:197]
    lg = V.logits(store, hE)[0]; lo = V.logits(store, ref["hidden"][-1])[0]; lf = V.logits(store, full[-1])[0]
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    kl = float((sm(lo) * (np.log(sm(lo) + 1e-30) - np.log(sm(lg) + 1e-30))).sum())
    print(f"[late] logits: encrypted argmax {int(lg.argmax())}, oracle {int(lo.argmax())}, full model {int(lf.argmax())}; "
          f"KL(oracle || encrypted) {kl:.3g}; |logit diff| max {np.abs(lg - lo).max():.3g}", flush=True)
    print(f"[late] done in {time.time() - t0:.0f}s, bts={int(h.inf.fhe.total_bootstraps)}", flush=True)
    h.model.close(); h.sess.close()


if __name__ == "__main__":
    main()
