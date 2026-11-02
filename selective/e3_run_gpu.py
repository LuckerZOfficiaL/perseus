#!/usr/bin/env python
"""E3 stage 4: the selective ViT prefill on the GPU through the planned path, N images with one fixed sensitive box.

  python selective/e3_run_gpu.py --plan bootstrap_placements/selective_e3/vit_sel_box4_sinc --n 3 --box 4 --check
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H
import e3_reference as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="bootstrap_placements/selective_e3/vit_sel_box4_sinc")
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs (upstream\'s --graphs)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--images", default="", help="directory of .npy (224,224,3) normalized arrays or images; default: the EuroSAT samples")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--box", type=int, default=4)
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--check", action="store_true", help="per-block error against the oracle")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e3/work"))
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e3/gpu_results.json"))
    ap.add_argument("--weights", default=None, help="a local HF checkpoint dir (fine-tuned weights) instead of the hub model")
    a = ap.parse_args()

    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=a.layers, cls=SelectivePrefill, plan=a.plan, weights=a.weights)
    V, store = h.V, h.store
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    from sfhe.vit_selective_planned import SelectivePlannedPrefill
    from perseus.examples.gpt2.prefill import report

    files = sorted(f for f in glob.glob(os.path.join(a.images or os.path.join(H.REPO, "scripts/repro/data/vit_eurosat"), "*"))
                   if f.lower().endswith((".npy", ".png", ".jpg", ".jpeg")))[:a.n]
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    imgs = []
    for f in files:
        px = np.load(f) if f.endswith(".npy") else V.pixels(f)
        rows = V.patch_rows(px)
        x = V.embed(store, rows)
        ref = R.two_stream(store, x, S, a.layers)
        full = V.exact_blocks(store, x, a.layers)
        imgs.append(dict(file=os.path.basename(f), rows_e=rows[ref["E_idx"]], pkv=ref["P_kv"], E=ref["E_hidden"],
                         E_idx=ref["E_idx"], full=full))
    E_idx = imgs[0]["E_idx"]
    n_e = len(E_idx)
    print(f"[gpu] {len(imgs)} images, {n_e} encrypted tokens (CLS + {len(S)} patches), {len(imgs[0]['pkv'][0][0])} "
          f"plaintext keys, plan {a.plan}", flush=True)

    cfg = plan_config(resolve(a.plan), graphs=a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = os.path.join(a.work_dir, f"sel_{os.getpid()}")
    run = SelectivePlannedPrefill(h.model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows,
                                  stc_cache=a.stc_cache, check_meta=False, log=lambda m: print(m, flush=True),
                                  ccmm_kappa=(256.0, 16.0), ccmm_sweep=(), positions=E_idx)
    t_set = time.time()
    ws = run.fix_ws([(im["rows_e"], im["pkv"]) for im in imgs])
    if ws:
        print(f"[gpu] route B: {len(ws)} per-image linears, fixed weight scales (b0.xqk {ws.get('b0.xqk')}, b0.xpv {ws.get('b0.xpv')})", flush=True)
    run.set_image(imgs[0]["pkv"], 0)
    run.prepare(imgs[0]["rows_e"])
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    print(f"[gpu] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {n_bts} refresh "
          f"sites + {n_fin[1]} PCMM/CCMM outputs ({n_fin[0]} finishes) = {n_bts + n_fin[1]} bts, registers {run.n_regs}",
          flush=True)
    if len(imgs) > 1:
        dyn = run.find_dynamic(imgs[1]["rows_e"], imgs[1]["pkv"])
        print(f"[gpu] per-image plaintexts: {sum(len(v) for v in dyn.values())} names over {len(dyn)} blocks "
              f"(block 0: {len(dyn.get(0, ()))})", flush=True)
    if a.compile_only:
        return 0
    run.setup(imgs[0]["rows_e"])
    free0 = run.fhe.mem_info()[0] / 2 ** 30
    print(f"[gpu] planned setup {time.time() - t_set:.1f}s; device free {free0:.2f} GB", flush=True)

    res = []
    for i, im in enumerate(imgs):
        n_staged, twin_s = run.stage_image(im["rows_e"], im["pkv"], img=i)
        vals = run.input_values(im["rows_e"])
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        wall = time.time() - t0
        hE = V.unscale(run.hidden())[:n_e]
        if a.check:
            for b in range(a.layers):
                hb = V.unscale(run.hidden(b))[:n_e]
                err = np.abs(hb - im["E"][b][:n_e]).max(1)
                print(f"[gpu] block {b} maxerr={err.max():.3g} (tok {int(err.argmax())}) |x|={np.abs(im['E'][b][:n_e]).max():.3g}",
                      flush=True)
        lg = V.logits(store, hE)[0]
        lo = V.logits(store, im["E"][-1])[0]
        lf = V.logits(store, im["full"][-1])[0]
        _, s_o = report([lg], [lo])
        r = dict(file=im["file"], wall=wall, twin_s=twin_s, staged=n_staged, argmax=int(lg.argmax()), oracle=int(lo.argmax()),
                 full=int(lf.argmax()), kl_oracle=s_o["medianKL"], max_logit_diff=float(np.abs(lg - lo).max()),
                 free_gb=round(run.fhe.mem_info()[0] / 2 ** 30, 2))
        res.append(r)
        print(f"[gpu] {r['file']}: wall={wall:.2f}s (twin re-record {twin_s:.0f}s, {n_staged} plaintexts restaged) "
              f"argmax enc {r['argmax']} oracle {r['oracle']} full {r['full']} KL(oracle||enc)={r['kl_oracle']:.3g} "
              f"|logit diff|={r['max_logit_diff']:.3g} free {r['free_gb']} GB", flush=True)
    walls = [r["wall"] for r in res]
    print(f"[gpu] SUMMARY images={len(res)} s/image={np.median(walls):.2f} (walls {' '.join(f'{w:.2f}' for w in walls)}) "
          f"bts/image={n_bts + n_fin[1]} argmax=oracle {sum(r['argmax'] == r['oracle'] for r in res)}/{len(res)} "
          f"=full {sum(r['argmax'] == r['full'] for r in res)}/{len(res)} medianKL={np.median([r['kl_oracle'] for r in res]):.3g}",
          flush=True)
    json.dump(dict(args=vars(a), n_e=n_e, S=S, bts=n_bts + n_fin[1], results=res, setup_s=time.time() - t_set),
              open(a.out, "w"), indent=1)
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
