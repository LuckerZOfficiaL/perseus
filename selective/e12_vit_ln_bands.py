#!/usr/bin/env python
"""E12: the ViT LayerNorm inverse-sqrt seed bands against realistic photos, and a refit (the ViT counterpart of
e8_ln_bands.py). z = c_eff_sq(position) * var(x) + eps * c_eff_sq(position) on the ENCRYPTED stream's LayerNorm inputs
(the plaintext stream is exact) of the two-stream ViT, over a calibration pool of Imagenette validation photos WITHOUT a
face (never evaluated), each with a random private set (a box of 1x1 to 8x8 patches at a random place, or a random
scatter of 4 to 63 patches); held-out check: the face photos' own private sets. Sites whose pool data come within
--headroom of a band edge are refitted on [min / safety, max x safety] (e8_ln_bands.fit_site).

  python selective/e12_vit_ln_bands.py --pool 800 --out selective/sfhe/configs/vit_base_224_photos/configs.json \
      [--weights DIR] [--weights DIR2]
"""
import argparse
import copy
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
CONFIG = "configs/model/approximation/vit_base_224/configs.json"


def collect(V, store, cfgs, items):
    """{site: z} over the encrypted stream's tokens of [(file, private patches)]."""
    from perseus.impl.config import parse_norm
    import e3_reference as R
    c2 = {n: np.array([parse_norm(cfgs[n]).c_eff_sq(p) for p in range(197)]) for n in cfgs}
    z = {}
    for f, S in items:
        x = V.embed(store, V.patch_rows(V.pixels(f)))
        E_idx = np.concatenate([[0], np.asarray(sorted(S)) + 1])

        def rec(b, name, rows):
            if name not in ("ln1_in", "ln2_in"):
                return
            nm = f"transformer.h.{b}.{'ln_1' if name == 'ln1_in' else 'ln_2'}"
            c = rows - rows.mean(-1, keepdims=True)
            cc = c2[nm][E_idx]
            z.setdefault(nm, []).append(cc * (c * c).mean(-1) + float(cfgs[nm]["eps"]) * cc)
        R.two_stream(store, x, S, 12, record=rec)
    return {k: np.concatenate(v) for k, v in z.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=800)
    ap.add_argument("--heldout", type=int, default=300, help="face photos checked after the fit")
    ap.add_argument("--weights", action="append", default=[], help="HF ViT checkpoint dirs ('' = the shipped model)")
    ap.add_argument("--headroom", type=float, default=1.1)
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--target-err", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.chdir(REPO)
    import sfhe  # noqa: F401
    from perseus._paths import resolve
    from perseus.examples.vit import weights as V
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    import e8_ln_bands as LB
    import e11_faces_common as F
    spec = importlib.util.spec_from_file_location("refit_norm_cheb", os.path.join(REPO, "scripts/utils/refit_norm_cheb.py"))
    RN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RN)
    j = json.load(open(resolve(CONFIG)))
    cfgs = {k: v for k, v in j["norm"].items() if k.startswith("transformer.h.")}
    det = json.load(open(F.FACES))
    noface = F.calib_photos()
    rng = np.random.default_rng(a.seed)
    pool = []
    for f in rng.choice(noface, size=min(a.pool, len(noface)), replace=False):
        if rng.random() < 0.5:
            s = int(rng.integers(1, 9))
            r0, c0 = (int(v) for v in rng.integers(0, 15 - s, size=2))
            S = [(r0 + i) * 14 + (c0 + jj) for i in range(s) for jj in range(s)]
        else:
            S = sorted(rng.choice(196, size=int(rng.integers(4, 64)), replace=False).tolist())
        pool.append((f, S))
    held = [(f, S) for f, _lab, S in F.private_sets(63)][:a.heldout]
    ztr, zva = {}, {}
    for w in (a.weights or [""]):
        store = V.VitStore(os.path.expanduser(w)) if w else V.VitStore()
        for dst, items in ((ztr, pool), (zva, held)):
            for k, v in collect(V, store, cfgs, items).items():
                dst.setdefault(k, []).append(v)
    ztr = {k: np.concatenate(v) for k, v in ztr.items()}
    zva = {k: np.concatenate(v) for k, v in zva.items()}
    out, ops, n_refit = copy.deepcopy(j), NumpyOps(), 0
    for nm in sorted(cfgs, key=lambda s: (int(s.split(".")[2]), s)):
        c = cfgs[nm]
        lo, hi = float(c["cheb_lo"]), float(c["cheb_hi"])
        dmin, dmax = float(ztr[nm].min()), float(ztr[nm].max())
        line = (f"{nm:22s} band [{lo:.3g}, {hi:.3g}] (deg {len(c['cheb_coeffs']) - 1}, NR{c['cheb_nr_iters']}) | pool "
                f"[{dmin:.3g}, {dmax:.3g}] | faces [{float(zva[nm].min()):.3g}, {float(zva[nm].max()):.3g}]")
        if dmax * a.headroom > hi or dmin / a.headroom < lo:
            nlo, nhi = min(lo, dmin / a.safety), max(hi, dmax * a.safety)
            (dep, _), deg, nr, coeffs, err = LB.fit_site(RN, nlo, nhi, float(c["inv_out_scale"]), a.target_err, 31)
            out["norm"][nm].update(cheb_coeffs=[float(v) for v in coeffs], cheb_lo=nlo, cheb_hi=nhi, cheb_nr_iters=nr)
            line += f" -> REFIT [{nlo:.3g}, {nhi:.3g}] deg {deg} NR{nr} depth {dep} err {err:.1e}"
            n_refit += 1
        oob = int(((zva[nm] < float(out["norm"][nm]["cheb_lo"])) | (zva[nm] > float(out["norm"][nm]["cheb_hi"]))).sum())
        print(line + f" | faces out of band {oob}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[vit-ln-bands] {n_refit} sites refitted -> {a.out}")


if __name__ == "__main__":
    main()
