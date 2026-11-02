#!/usr/bin/env python
"""E8: the LayerNorm inverse-sqrt seed bands of the BERT approximation config against SST-2 statistics, and a refit.

Per norm site, z = c_eff_sq(position) * var(x) + eps * c_eff_sq(position) on the LayerNorm input x (refit_norm_cheb.py's
quantity), over the tokens of both streams of the two-stream model (the plaintext stream = the redacted sentence; the
encrypted stream = CLS, private span, SEP) on SST-2 TRAIN sentences, for each given weight set. Sites whose data leave
the shipped Chebyshev band [cheb_lo, cheb_hi] (with --headroom) are refitted on the widened band
[data_min / safety, data_max * safety] with refit_norm_cheb.fit_site (same target error); the others are kept. Then the
validation sentences are checked against the new bands (held out).

  python selective/e8_ln_bands.py --train 2000 --val 872 --weights "" --weights EXPORT_DIR --out CONFIG.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
CONFIG = "configs/model/approximation/bert_base_n32_t128/configs.json"
TOKENS, FULL = 128, False                                  # --tokens, --full


def site(b, which):
    return f"transformer.h.{b}.{'ln_1' if which == 'ln1' else 'ln_2'}"


def collect(cfgs, idx, split, weights, frac=0.3, k_max=8):
    """{site: z over every token of both streams} for SST-2 `split` sentences idx under the given export."""
    from perseus.impl.config import parse_norm
    if weights:
        os.environ["BERT_EXPORT"] = weights
    else:
        os.environ.pop("BERT_EXPORT", None)
    import sfhe  # noqa: F401  (BERT_EXPORT is read at call time by the patched data.export_dir)
    from perseus.examples.bert import data as D
    import e8_reference as R
    st, cl = D.store(), D.client()
    sents = D.sentences(max(idx) + 1, split)
    c2 = {n: np.array([parse_norm(cfgs[n]).c_eff_sq(p) for p in range(TOKENS)]) for n in cfgs}
    z = {}
    for i in idx:
        s, _lab = sents[i]
        ids, n = D.encode(s, TOKENS)
        L = n - 2
        if L < 2:
            continue
        if FULL:                                          # every token encrypted: the full model's LayerNorm inputs
            S = list(range(1, n - 1))
        else:
            rng = np.random.default_rng(i)
            k = int(min(k_max, max(1, round(frac * L))))
            if L - k < 1:
                continue
            s0 = int(rng.integers(0, L - k + 1))
            S = list(range(1 + s0, 1 + s0 + k))
        E_idx, _NS = R.streams(ids, n, S)
        pos = {"P": np.arange(n), "E": E_idx}

        def rec(b, which, tag, y):
            if which not in ("ln1", "ln2"):
                return
            nm = site(b, which)
            c = y - y.mean(-1, keepdims=True)
            cc = c2[nm][np.minimum(pos[tag], TOKENS - 1)]
            z.setdefault(nm, []).append(cc * (c * c).mean(-1) + float(cfgs[nm]["eps"]) * cc)
        R.two_stream(st, cl, ids, n, S, record=rec)
    return {k: np.concatenate(v) for k, v in z.items()}


def fit_site(RN, lo, hi, scale, target, max_deg, max_nr=8):
    """refit_norm_cheb.fit_site with up to max_nr Newton iterations (the shipped wide bands use up to 7)."""
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    ops = NumpyOps()
    t = np.cos(np.pi * (np.arange(400) + 0.5) / 400)
    xs = (t + 1) / 2 * (hi - lo) + lo
    f = scale / np.sqrt(xs)
    grid = np.geomspace(lo, hi, 4001)
    exact = scale / np.sqrt(grid)
    best = None
    for deg in range(2, max_deg + 1):
        w = 1.0 / f
        c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w)
        for _ in range(30):
            r = np.abs(np.polynomial.chebyshev.chebval(t, c) / f - 1)
            c = np.polynomial.chebyshev.chebfit(t, f, deg, w=w * (r / r.max() + 0.05))
        for nr in range(0, max_nr + 1):
            y = inv_sqrt_newton_d2(ops, grid, eval_chebyshev(ops, grid, c, lo, hi), nr, 1.0 / (scale * scale))
            err = float(np.max(np.abs(y / exact - 1)))
            if np.isfinite(err) and err <= target:
                key = (RN.depth(deg, nr), deg + 3 * nr)
                if best is None or key < best[0]:
                    best = (key, deg, nr, c, err)
                break
    if best is None:
        raise RuntimeError(f"no chain up to degree {max_deg} / {max_nr} Newton steps meets {target} on [{lo:.3g}, {hi:.3g}]")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=872)
    ap.add_argument("--weights", action="append", default=[], help="export dirs ('' = the shipped SST-2 model)")
    ap.add_argument("--headroom", type=float, default=1.1, help="refit a site when its data come within this factor of a band edge")
    ap.add_argument("--safety", type=float, default=1.5)
    ap.add_argument("--target-err", type=float, default=1e-5)
    ap.add_argument("--tokens", type=int, default=128, help="T: texts padded / truncated to T (E14: 256)")
    ap.add_argument("--full", action="store_true", help="the full model (every token encrypted), not the two streams")
    ap.add_argument("--base", default=None, help="the config to start from (default: upstream's BERT config)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global TOKENS, FULL
    TOKENS, FULL = a.tokens, a.full
    os.chdir(REPO)
    from perseus._paths import resolve
    spec = importlib.util.spec_from_file_location("refit_norm_cheb", os.path.join(REPO, "scripts/utils/refit_norm_cheb.py"))
    RN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RN)
    from perseus.impl.config import parse_norm
    from perseus.impl.poly import NumpyOps, eval_chebyshev, inv_sqrt_newton_d2
    j = json.load(open(resolve(a.base or CONFIG)))
    cfgs = {k: v for k, v in j["norm"].items() if k.startswith("transformer.h.")}
    weights = a.weights or [""]
    ztr, zva = {}, {}
    for w in weights:
        for dst, idx, split in ((ztr, range(a.train), "train"), (zva, range(a.val), "validation")):
            for k, v in collect(cfgs, list(idx), split, w).items():
                dst.setdefault(k, []).append(v)
    ztr = {k: np.concatenate(v) for k, v in ztr.items()}
    zva = {k: np.concatenate(v) for k, v in zva.items()}
    out = copy.deepcopy(j)
    ops = NumpyOps()
    n_refit = 0
    for nm in sorted(cfgs, key=lambda s: (int(s.split(".")[2]), s)):
        c = cfgs[nm]
        lo, hi = float(c["cheb_lo"]), float(c["cheb_hi"])
        dmin, dmax = float(ztr[nm].min()), float(ztr[nm].max())
        vmin, vmax = float(zva[nm].min()), float(zva[nm].max())
        line = (f"{nm:22s} band [{lo:.3g}, {hi:.3g}] (deg {len(c['cheb_coeffs']) - 1}, NR{c['cheb_nr_iters']}) | train "
                f"[{dmin:.3g}, {dmax:.3g}] | val [{vmin:.3g}, {vmax:.3g}]")
        if dmax * a.headroom > hi or dmin / a.headroom < lo:
            nlo, nhi = min(lo, dmin / a.safety), max(hi, dmax * a.safety)
            (dep, _), deg, nr, coeffs, err = fit_site(RN, nlo, nhi, float(c["inv_out_scale"]), a.target_err, 31)
            h = zva[nm]
            y = inv_sqrt_newton_d2(ops, h, eval_chebyshev(ops, h, coeffs, nlo, nhi), nr, 1.0 / float(c["inv_out_scale"]) ** 2)
            herr = float(np.max(np.abs(y * np.sqrt(h) / float(c["inv_out_scale"]) - 1)))
            out["norm"][nm].update(cheb_coeffs=[float(v) for v in coeffs], cheb_lo=nlo, cheb_hi=nhi, cheb_nr_iters=nr)
            line += f" -> REFIT [{nlo:.3g}, {nhi:.3g}] deg {deg} NR{nr} depth {dep} err {err:.1e}, val err {herr:.1e}"
            n_refit += 1
        oob = int(((zva[nm] < float(out["norm"][nm]["cheb_lo"])) | (zva[nm] > float(out["norm"][nm]["cheb_hi"]))).sum())
        print(line + f" | val out of band {oob}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[ln-bands] {n_refit} sites refitted -> {a.out}")


if __name__ == "__main__":
    main()
