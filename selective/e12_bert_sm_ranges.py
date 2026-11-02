#!/usr/bin/env python
"""E12: the BERT softmax approximation ranges against SST-2 (the counterpart of e12_vit_sm_ranges.py): per layer, the
encrypted stream's attention scores versus the exponential's clip range and the round-0 denominator
D0 = 1 / kc sum_j exp((s_j - mid) / 2^log2delta2) versus the band of the calibrated reciprocal seed, over SST-2 TRAIN
sentences with a random private span (frac 0.1 to 0.5, as E1). --refit widens every layer's round-0 band to
[min / 2, max x margin] (never narrower than the base) and redesigns its rounds (smcheb.design, the base's final target).
--clip-margin M (E14): where scores leave the exponential's clip range, widen its domain around the same centre to
M x the observed half-range and refit its Chebyshev series (upstream's recipe, perseus.examples.bert.calib, sm_margin
1.1). --gelu-margin M: where the encrypted stream's GELU inputs exceed the config's domain X, refit the degree-d
Chebyshev of GELU(X t) / X on X = M x max |u| (the shipped config's form, M = 1.25).

  python selective/e12_bert_sm_ranges.py --train 3000 --weights '' --weights EXPORT --refit --margin 4 --base CFG --out CFG2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS, FULL = 128, False                                  # --tokens, --full
sys.path.insert(0, os.path.join(REPO, "python"))


def measure(cfgs, n, weights, seed):
    from perseus.examples.bert import data as D
    import e8_reference as R
    if weights:
        os.environ["BERT_EXPORT"] = weights
    else:
        os.environ.pop("BERT_EXPORT", None)
    st, cl = D.store(), D.client()
    sents = D.sentences(n, "train")
    rng = np.random.default_rng(seed)
    out = {b: dict(smin=np.inf, smax=-np.inf, dmin=np.inf, dmax=-np.inf, n_over=0, n_tok=0, umax=0.0) for b in range(12)}
    for s, _lab in sents:
        ids, nt = D.encode(s, TOKENS)
        L = nt - 2
        if L < 2:
            continue
        k = L if FULL else max(1, int(round(rng.uniform(0.1, 0.5) * L)))   # --full: every token encrypted
        if L - k < 1 and not FULL:
            continue
        s0 = int(rng.integers(0, L - k + 1))

        def rec(b, which, tag, y):
            if which == "gelu" and tag == "E":
                out[b]["umax"] = max(out[b]["umax"], float(np.abs(y).max()))
            if which != "qk":
                return
            q, K = y
            sm = cfgs.block(b)[2]
            mid = (sm.clip_hi + sm.clip_lo) / 2.0
            o = out[b]
            for h in range(12):
                sc = q[:, h * 64:(h + 1) * 64] @ K[:, h * 64:(h + 1) * 64].T / 8.0
                d0 = 1.0 / K.shape[0] * np.exp((sc - mid) / 2.0 ** sm.log2delta2).sum(1)
                o["smin"], o["smax"] = min(o["smin"], float(sc.min())), max(o["smax"], float(sc.max()))
                o["dmin"], o["dmax"] = min(o["dmin"], float(d0.min())), max(o["dmax"], float(d0.max()))
                o["n_over"] += int(((sc > sm.clip_hi) | (sc < sm.clip_lo)).any(1).sum())
                o["n_tok"] += sc.shape[0]
        R.two_stream(st, cl, ids, nt, list(range(1 + s0, 1 + s0 + k)), record=rec)
    return out


def widen_exp(sj, o, margin):
    """upstream's exp-domain widening (perseus.examples.bert.calib): the same centre, half-width margin x the observed
    half-range (never narrower), the series refitted at the same degree. Returns the log fragment."""
    from perseus.impl.config import parse_softmax
    sm = parse_softmax(sj)
    mean = (sm.clip_hi + sm.clip_lo) / 2.0
    sf = 2.0 ** (-sm.log2delta1 - sm.log2delta2)
    half = max(float(sj["squeeze_bound"]), margin * max(mean - o["smin"], o["smax"] - mean))
    u = np.cos(np.pi * (np.arange(64) + 0.5) / 64)
    cc = np.polynomial.chebyshev.chebfit(u, np.exp(half * sf * u), len(sj["cheb_coeffs"]) - 1)
    v = np.linspace(-1, 1, 4001)
    err = float(np.abs(np.polynomial.chebyshev.chebval(v, cc) - np.exp(half * sf * v)).max())
    sj.update(clip_lo=mean - half, clip_hi=mean + half, squeeze_bound=half, cheb_a=-half * sf, cheb_b=half * sf,
              cheb_coeffs=[float(x) for x in cc])
    return f" -> clip [{mean - half:.3g}, {mean + half:.3g}] exp err {err:.1e}"


def refit_gelu(g, X):
    """The shipped GELU form: a Chebyshev series of GELU(X t) / X on [-1, 1] (same degree) on the wider X."""
    from scipy.special import erf
    deg = len(g["cheb_coeffs"]) - 1
    gel = lambda x: 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    k = np.arange(4 * (deg + 1))
    t = np.cos(np.pi * (k + 0.5) / len(k))
    cc = np.polynomial.chebyshev.chebfit(t, gel(X * t) / X, deg)
    v = np.linspace(-1, 1, 20001)
    err = float(np.abs(np.polynomial.chebyshev.chebval(v, cc) * X - gel(X * v)).max())
    old = float(g["xmax"])
    g.update(xmax=float(X), cheb_coeffs=[float(x) for x in cc],
             note=f"degree-{deg} Chebyshev of GELU(X t)/X (erf) on [-1,1], X = {X:.4g} (refit by e12_bert_sm_ranges.py over "
                  f"the data's max |u|; was {old:.4g}); max abs err {err:.1e} on [-X, X]")
    return f" -> GELU X {old:.3g} -> {X:.3g} err {err:.1e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--weights", action="append", default=[], help="perseus export dirs ('' = the shipped SST-2 model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--base", default=None, help="config (default: BERT_CONFIGS or upstream's)")
    ap.add_argument("--tokens", type=int, default=128, help="T: texts padded / truncated to T (E14: 256)")
    ap.add_argument("--full", action="store_true", help="the full model (every token encrypted), not the two streams")
    ap.add_argument("--clip-margin", type=float, default=0.0, help="--refit: widen the exponential's domain where scores "
                                                                    "leave it (upstream's 1.1); 0 = keep")
    ap.add_argument("--gelu-margin", type=float, default=0.0, help="--refit: refit GELU where its inputs leave X "
                                                                    "(the shipped 1.25); 0 = keep")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global TOKENS, FULL
    TOKENS, FULL = a.tokens, a.full
    os.chdir(REPO)
    import sfhe  # noqa: F401
    from perseus._paths import resolve
    from perseus.impl import config, smcheb
    from perseus.impl.config import parse_softmax
    base = a.base or os.environ.get("BERT_CONFIGS", "configs/model/approximation/bert_base_n32_t128/configs.json")
    cfgs = config.load_configs(base)
    out = None
    for w in (a.weights or [""]):
        o = measure(cfgs, a.train, w, a.seed)
        out = o if out is None else {b: dict(smin=min(out[b]["smin"], o[b]["smin"]), smax=max(out[b]["smax"], o[b]["smax"]),
                                             dmin=min(out[b]["dmin"], o[b]["dmin"]), dmax=max(out[b]["dmax"], o[b]["dmax"]),
                                             n_over=out[b]["n_over"] + o[b]["n_over"], n_tok=out[b]["n_tok"] + o[b]["n_tok"],
                                             umax=max(out[b]["umax"], o[b]["umax"]))
                                     for b in range(12)}
    j = json.load(open(resolve(base)))
    kcs = tuple(sorted(set(smcheb.KC_DESIGN) | {128, TOKENS}))
    for b in range(12):
        sm = cfgs.block(b)[2]
        lo0, hi0 = smcheb.band_from_seed(sm.init_alpha, sm.init_beta)
        o = out[b]
        flag = (" SCORES OUT" if o["n_over"] else "") + (" D0 OUT" if (o["dmax"] > hi0 or o["dmin"] < lo0) else "")
        line = (f"layer {b:2d}: scores [{o['smin']:.3g}, {o['smax']:.3g}] vs clip [{sm.clip_lo:.3g}, {sm.clip_hi:.3g}] "
                f"({o['n_over']}/{o['n_tok']} query-heads out) | D0 [{o['dmin']:.3g}, {o['dmax']:.3g}] vs band [{lo0:.3g}, {hi0:.3g}]{flag}")
        g = j["softgelu"][f"transformer.h.{b}.mlp.act"]
        X = float(g.get("xmax", 0.0))
        line += f" | GELU max|u| {o['umax']:.3g} vs X {X:.3g}" + (" GELU OUT" if o["umax"] > X else "")
        if a.refit and a.gelu_margin and o["umax"] > X:
            line += refit_gelu(g, a.gelu_margin * o["umax"])
        if a.refit and a.clip_margin and o["n_over"]:
            line += widen_exp(j["softmax"][f"transformer.h.{b}.attn"], o, a.clip_margin)
        if a.refit:
            e = j["softmax"][f"transformer.h.{b}.attn"]
            c = parse_softmax(e)
            lo, hi = min(lo0, o["dmin"] / 2.0), max(hi0, o["dmax"] * a.margin)
            al, be = smcheb.cheb_seed(lo, hi)
            ks, tg = smcheb.design(al, be, c.log2delta2, float(e["recip_targets"][-1]), kcs)
            line += f" -> band [{lo:.3g}, {hi:.3g}], rounds {e['recip_iters_cheb']} -> {ks}"
            e.update(init_alpha=al, init_beta=be, recip_iters_cheb=ks, recip_targets=tg)
        print(line, flush=True)
    if a.refit:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(j, open(a.out, "w"), indent=1)
        print(f"[bert-sm-ranges] wrote {a.out}")


if __name__ == "__main__":
    main()
