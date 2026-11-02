#!/usr/bin/env python
"""E11 twin test: face photos with different private sizes packed on one page (ViT, route B, page layouts, XPT_PAD), each
against its exact two-stream oracle, block by block and at the logits.

  XPT_PAD=1 python selective/e11_test_faces.py --n 6 --chunk0 3 --k-max 20 --cap 21 [--capture DIR]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H
import e11_faces_common as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--chunk0", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--cap", type=int, default=21)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--tail", action="store_true", help="take the photos from the end of the bucket (the held-out ones)")
    ap.add_argument("--blob-seed", type=int, default=None,
                    help="photos WITHOUT a face with random face-like private blobs (F.blob_sets) instead of face photos")
    ap.add_argument("--grow-first", action="store_true",
                    help="grow the first photo's private set to cap - 1 patches (every own-key offset live, for a capture)")
    a = ap.parse_args()
    os.environ.setdefault("SEL_ROUTE", "pcmm")
    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=a.layers, capture=a.capture, cls=SelectivePrefill, weights=a.weights)
    V, store, m = h.V, h.store, h.model
    m.n_e_cap = a.cap
    if a.blob_seed is not None:
        sets = F.blob_sets(3 * a.n, a.k_min, a.k_max, a.blob_seed)
    else:
        sets = F.private_sets(a.k_max, a.k_min)
        sets = (sets[::-1] if a.tail else sets)[a.first:a.first + (3 * a.n if a.chunk0 is not None else a.n)]
    if a.grow_first:
        sets[0] = (sets[0][0], sets[0][1], F.grow(sets[0][2], a.cap - 1))
    ins = F.inputs(V, store, sets, a.layers)
    if a.chunk0 is not None:                     # inputs that fit the chunks' lanes (e8_harness.pack_pages: the first page)
        import e8_harness as EH
        ins = [x for x in EH.pack_pages(ins, a.n, a.chunk0, m.blocks.t)[0][0]]
    rows, pos, layout, P_kv = m.page([x["rows_e"] for x in ins], [x["pos"] for x in ins], [x["ref"]["P_kv"] for x in ins],
                                     a.chunk0)
    print(f"[e11] {len(ins)} face photos on one page, layout {layout}, {a.layers} layers", flush=True)
    off = np.cumsum([0] + [x[-2] for x in layout])
    errs = []

    def on_block(b, Xs, s):
        hE = V.unscale(m.hidden(Xs, m.last_ch))
        e = [np.abs(hE[off[q]:off[q + 1]] - x["ref"]["E_hidden"][b]).max() for q, x in enumerate(ins)
             if not x.get("filler")]                     # fillers (pack_pages) may be cut: no oracle of their own
        errs.append(max(e))
        print(f"[e11] block {b} {s:.1f}s maxerr per input {' '.join(f'{v:.3g}' for v in e)} bts={int(h.inf.fhe.total_bootstraps)}",
              flush=True)
    t0 = time.time()
    Xs, ch, _ = m.prefill_page(rows, pos, layout, P_kv, on_block=on_block)
    hE = V.unscale(m.hidden(Xs, ch))
    if a.layers == 12:
        sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        for q, x in enumerate(ins):
            if x.get("filler"):
                continue
            lg = V.logits(store, hE[off[q]:off[q + 1]])[0]
            lo = V.logits(store, x["ref"]["E_hidden"][-1])[0]
            kl = float((sm(lo) * (np.log(sm(lo) + 1e-30) - np.log(sm(lg) + 1e-30))).sum())
            print(f"[e11] photo {os.path.basename(x['file'])} ({len(x['S'])} private patches): encrypted {int(lg.argmax())}, "
                  f"oracle {int(lo.argmax())}, full {int(x['full'].argmax())}, label {x['label']}; KL {kl:.3g}", flush=True)
    print(f"[e11] done in {time.time() - t0:.0f}s, bts={int(h.inf.fhe.total_bootstraps)}, worst block maxerr {max(errs):.3g}",
          flush=True)


if __name__ == "__main__":
    main()
