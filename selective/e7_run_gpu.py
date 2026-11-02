#!/usr/bin/env python
"""E7 on the GPU: two-stream route B with `--inputs` images' encrypted tokens on ONE page, through the planned path.
Images are taken in order and grouped into pages; each input is scored against its own exact two-stream oracle.

  python selective/e7_run_gpu.py --plan bootstrap_placements/selective_e3/vit_sel_box4_b3 --inputs 3 --n 6 \
      [--images DIR] [--weights DIR] [--compile-only]
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
    ap.add_argument("--plan", required=True)
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs (upstream\'s --graphs)")
    ap.add_argument("--inputs", type=int, default=3, help="inputs per encrypted page")
    ap.add_argument("--n", type=int, default=6, help="images (whole pages only: rounded down to a multiple of --inputs)")
    ap.add_argument("--images", default="")
    ap.add_argument("--box", type=int, default=4)
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--chunk0", type=int, default=None, help="inputs per page on chunk 0 (default all); the rest on chunk 1")
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--compile-only", action="store_true", help="record + compile on the CPU, no device session")
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e7/work"))
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e7/gpu_results.json"))
    a = ap.parse_args()
    os.environ.setdefault("SEL_ROUTE", "pcmm")
    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=12, cls=SelectivePrefill, plan=a.plan, weights=a.weights)
    V, store = h.V, h.store
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    from sfhe.vit_batch_planned import BatchSelectivePlannedPrefill, page_pkv
    from perseus.examples.gpt2.prefill import report

    files = sorted(f for f in glob.glob(os.path.join(a.images or os.path.join(H.REPO, "scripts/repro/data/vit_eurosat"), "*"))
                   if f.lower().endswith((".npy", ".png", ".jpg", ".jpeg")))
    n = (min(a.n, len(files)) // a.inputs) * a.inputs
    files = files[:n]
    if a.compile_only and n < 2 * a.inputs:              # the dynamic-plaintext probe needs a second page
        files = (files * 2)[:2 * a.inputs]
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    ims = []
    for f in files:
        px = np.load(f) if f.endswith(".npy") else V.pixels(f)
        rows = V.patch_rows(px)
        x = V.embed(store, rows)
        ref = R.two_stream(store, x, S, 12)
        ims.append(dict(file=os.path.basename(f), rows_e=rows[ref["E_idx"]], pkv=ref["P_kv"], E=ref["E_hidden"],
                        E_idx=ref["E_idx"], full=V.exact_blocks(store, x, 12)))
    E_idx = ims[0]["E_idx"]
    n_e = len(E_idx)
    pages = []
    for p in range(len(ims) // a.inputs):
        grp = ims[p * a.inputs:(p + 1) * a.inputs]
        pages.append(dict(ims=grp, rows=np.concatenate([im["rows_e"] for im in grp]), pkv=page_pkv([im["pkv"] for im in grp])))
    print(f"[e7-gpu] {len(ims)} images on {len(pages)} pages of {a.inputs} inputs x {n_e} encrypted tokens, plan {a.plan}",
          flush=True)

    cfg = plan_config(resolve(a.plan), graphs=a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = os.path.join(a.work_dir, f"b{a.inputs}_{os.getpid()}")
    run = BatchSelectivePlannedPrefill(h.model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows,
                                       stc_cache=a.stc_cache, check_meta=False, log=lambda m: print(m, flush=True),
                                       ccmm_kappa=(256.0, 16.0), ccmm_sweep=(), positions=E_idx, groups=a.inputs,
                                       chunk0=a.chunk0)
    t_set = time.time()
    ws = run.fix_ws([(pg["rows"], pg["pkv"]) for pg in pages])
    print(f"[e7-gpu] {len(ws)} per-page linears, fixed weight scales (b0.xqk {ws.get('b0.xqk')}, b0.xpv {ws.get('b0.xpv')})",
          flush=True)
    run.set_image(pages[0]["pkv"], 0)
    run.prepare(pages[0]["rows"])
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    bts = n_bts + n_fin[1]
    print(f"[e7-gpu] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {n_bts} refresh "
          f"sites + {n_fin[1]} PCMM/CCMM outputs = {bts} bts per page, {bts / a.inputs:.0f} per image", flush=True)
    if len(pages) > 1:
        dyn = run.find_dynamic(pages[1]["rows"], pages[1]["pkv"])
        print(f"[e7-gpu] per-page plaintexts: {sum(len(v) for v in dyn.values())} names", flush=True)
    if a.compile_only:
        return 0
    run.setup(pages[0]["rows"])
    print(f"[e7-gpu] planned setup {time.time() - t_set:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB", flush=True)
    res = []
    sm = None
    for p, pg in enumerate(pages):
        n_staged, stage_s = run.stage_image(pg["rows"], pg["pkv"], img=p)
        vals = run.input_values(pg["rows"])
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        wall = time.time() - t0
        hE = V.unscale(run.hidden())
        for q, im in enumerate(pg["ims"]):
            lg = V.logits(store, hE[q * n_e:(q + 1) * n_e])[0]
            lo = V.logits(store, im["E"][-1])[0]
            lf = V.logits(store, im["full"][-1])[0]
            _, s_o = report([lg], [lo])
            r = dict(file=im["file"], page=p, slot=q, wall=wall / a.inputs, page_wall=wall, twin_s=stage_s, staged=n_staged,
                     argmax=int(lg.argmax()), oracle=int(lo.argmax()), full=int(lf.argmax()), kl_oracle=s_o["medianKL"],
                     max_logit_diff=float(np.abs(lg - lo).max()))
            res.append(r)
            print(f"[e7-gpu] page {p} input {q} {r['file']}: page wall={wall:.2f}s (stage {stage_s:.1f}s, {n_staged} uploads) "
                  f"argmax enc {r['argmax']} oracle {r['oracle']} full {r['full']} KL(oracle||enc)={r['kl_oracle']:.3g} "
                  f"|logit diff|={r['max_logit_diff']:.3g}", flush=True)
    walls = [r["page_wall"] for r in res[::a.inputs]]
    print(f"[e7-gpu] SUMMARY inputs/page={a.inputs} pages={len(walls)} images={len(res)} s/page={np.median(walls):.2f} "
          f"s/image={np.median(walls) / a.inputs:.2f} bts/page={bts} argmax=oracle {sum(r['argmax'] == r['oracle'] for r in res)}/{len(res)} "
          f"=full {sum(r['argmax'] == r['full'] for r in res)}/{len(res)} medianKL={np.median([r['kl_oracle'] for r in res]):.3g}",
          flush=True)
    json.dump(dict(args=vars(a), n_e=n_e, S=S, bts_page=bts, results=res, setup_s=time.time() - t_set),
              open(a.out, "w"), indent=1)
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
