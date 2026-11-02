#!/usr/bin/env python
"""E12: the ViT softmax approximation ranges against realistic photos: per layer, the encrypted stream's attention scores
(against its own and the plaintext keys) versus the exponential's clip range [clip_lo, clip_hi], and the round-0
denominator D0 = 1 / kc sum_j exp((s_j - mid) / 2^log2delta2) (0.5 / kc, then 2 Re) (impl dense_xpt.softmax_x / smcheb) versus the band of
the calibrated reciprocal seed. Pool: Imagenette photos without a face (random private sets, as e12_vit_ln_bands.py) and
optionally named photos (with their face patches).

  python selective/e12_vit_sm_ranges.py --pool 200 [--photos REL,REL] [--weights DIR]
  python selective/e12_vit_sm_ranges.py --pool 300 --weights '' --weights E1B --weights E5B --refit --margin 4 \
      --base CONFIG --out CONFIG2      # widen every layer's round-0 reciprocal band to [min / 2, max x margin] of the pool
                                       # (never narrower than shipped) and redesign its rounds (smcheb.design, same target)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def measure(V, store, cfgs, items):
    import e3_reference as R
    out = {b: dict(smin=np.inf, smax=-np.inf, dmin=np.inf, dmax=-np.inf, n_over=0, n_tok=0) for b in range(12)}
    for f, S in items:
        x = V.embed(store, V.patch_rows(V.pixels(f)))
        rec = {}
        ref = R.two_stream(store, x, S, 12, record=lambda b, n, r: rec.__setitem__((b, n), r.copy()) if n in ("q", "k") else None)
        for b in range(12):
            sm = cfgs.block(b)[2]
            mid = (sm.clip_hi + sm.clip_lo) / 2.0
            q, kE = rec[(b, "q")], rec[(b, "k")]
            K = np.vstack([kE, ref["P_kv"][b][0]])
            kc = K.shape[0]
            for h in range(12):
                s = q[:, h * 64:(h + 1) * 64] @ K[:, h * 64:(h + 1) * 64].T / 8.0
                d0 = 1.0 / kc * np.exp((s - mid) / 2.0 ** sm.log2delta2).sum(1)   # 0.5 / kc x 2 Re (im_cleanse)
                o = out[b]
                o["smin"], o["smax"] = min(o["smin"], float(s.min())), max(o["smax"], float(s.max()))
                o["dmin"], o["dmax"] = min(o["dmin"], float(d0.min())), max(o["dmax"], float(d0.max()))
                o["n_over"] += int(((s > sm.clip_hi) | (s < sm.clip_lo)).any(1).sum())
                o["n_tok"] += s.shape[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--photos", default="", help="comma-separated photos (relative to the Imagenette val root) with their faces")
    ap.add_argument("--weights", action="append", default=[], help="HF ViT checkpoint dirs ('' = the shipped model)")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--base", default=None, help="config to refit (default: VIT_CONFIGS or upstream's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(REPO)
    import sfhe  # noqa: F401
    from perseus.examples.vit import weights as V
    from perseus.impl import config, smcheb
    import e11_faces_common as F
    base = a.base or os.environ.get("VIT_CONFIGS", "configs/model/approximation/vit_base_224/configs.json")
    cfgs = config.load_configs(base)
    det = json.load(open(F.FACES))
    items = []
    if a.photos:
        for rel in a.photos.split(","):
            items.append((os.path.join(F.ROOT, rel), det["detections"][rel]["patches"]))
    else:
        noface = F.calib_photos()
        rng = np.random.default_rng(a.seed)
        for f in rng.choice(noface, size=min(a.pool, len(noface)), replace=False):
            s = int(rng.integers(1, 9))
            r0, c0 = (int(v) for v in rng.integers(0, 15 - s, size=2))
            items.append((f, [(r0 + i) * 14 + (c0 + j) for i in range(s) for j in range(s)]))
    out = None
    for w in (a.weights or [""]):
        store = V.VitStore(os.path.expanduser(w)) if w else V.VitStore()
        o = measure(V, store, cfgs, items)
        out = o if out is None else {b: dict(smin=min(out[b]["smin"], o[b]["smin"]), smax=max(out[b]["smax"], o[b]["smax"]),
                                             dmin=min(out[b]["dmin"], o[b]["dmin"]), dmax=max(out[b]["dmax"], o[b]["dmax"]),
                                             n_over=out[b]["n_over"] + o[b]["n_over"], n_tok=out[b]["n_tok"] + o[b]["n_tok"])
                                     for b in range(12)}
    for b in range(12):
        sm = cfgs.block(b)[2]
        lo0, hi0 = smcheb.band_from_seed(sm.init_alpha, sm.init_beta)
        o = out[b]
        flag = " SCORES OUT" if o["n_over"] else ""
        flag += " D0 OUT" if (o["dmax"] > hi0 or o["dmin"] < lo0) else ""
        print(f"layer {b:2d}: scores [{o['smin']:.3g}, {o['smax']:.3g}] vs clip [{sm.clip_lo:.3g}, {sm.clip_hi:.3g}] "
              f"({o['n_over']}/{o['n_tok']} query-heads out) | D0 [{o['dmin']:.3g}, {o['dmax']:.3g}] vs band "
              f"[{lo0:.3g}, {hi0:.3g}]{flag}", flush=True)
    if a.refit:
        from perseus._paths import resolve
        from perseus.impl.config import parse_softmax
        j = json.load(open(resolve(base)))
        kcs = tuple(sorted(set(smcheb.KC_DESIGN) | {197}))
        for b in range(12):
            e = j["softmax"][f"transformer.h.{b}.attn"]
            c = parse_softmax(e)
            lo0, hi0 = smcheb.band_from_seed(c.init_alpha, c.init_beta)
            lo, hi = min(lo0, out[b]["dmin"] / 2.0), max(hi0, out[b]["dmax"] * a.margin)
            al, be = smcheb.cheb_seed(lo, hi)
            ks, tg = smcheb.design(al, be, c.log2delta2, float(e["recip_targets"][-1]), kcs)
            print(f"layer {b:2d}: round-0 band [{lo0:.3g}, {hi0:.3g}] -> [{lo:.3g}, {hi:.3g}], rounds {e['recip_iters_cheb']} -> {ks}")
            e.update(init_alpha=al, init_beta=be, recip_iters_cheb=ks, recip_targets=tg)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(j, open(a.out, "w"), indent=1)
        print(f"[sm-ranges] wrote {a.out}")


if __name__ == "__main__":
    main()
