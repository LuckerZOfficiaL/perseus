#!/usr/bin/env python
"""E6 on the GPU: late-inject-replace through the planned path, N images, one fixed private box, k encrypted layers.

  python selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k2_box4 --k 2 --n 3 [--weights DIR]
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
import e6_reference as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs (upstream\'s --graphs)")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--faces", default=None, help="E11: K_MIN,K_MAX face photos (e11_faces_common), private = their faces")
    ap.add_argument("--heldout", type=int, default=0, help="--faces: leave out the bucket's last 2 x HELDOUT photos")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--images", default="")
    ap.add_argument("--box", type=int, default=4)
    ap.add_argument("--at", default="3,3")
    ap.add_argument("--at2", default=None, help="a second private box (row,col), e.g. in the second 128-token page")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e6/work"))
    ap.add_argument("--compile-only", action="store_true", help="record + compile on the CPU, no device session")
    ap.add_argument("--direct-handover", action="store_true", help="build the per-image hand-over plaintexts directly, no twin pass")
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e6/gpu_results.json"))
    a = ap.parse_args()
    from sfhe.vit_late import LatePrefill
    h = H.setup(layers=12, cls=LatePrefill, plan=a.plan, weights=a.weights)
    V, store = h.V, h.store
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    from sfhe.vit_late_planned import LatePlannedPrefill
    from perseus.examples.gpt2.prefill import report
    files = sorted(f for f in glob.glob(os.path.join(a.images or os.path.join(H.REPO, "scripts/repro/data/vit_eurosat"), "*"))
                   if f.lower().endswith((".npy", ".png", ".jpg", ".jpeg")))[:a.n]
    r0, c0 = (int(v) for v in a.at.split(","))
    S = [(r0 + i) * 14 + (c0 + j) for i in range(a.box) for j in range(a.box)]
    if a.at2:
        r2, c2 = (int(v) for v in a.at2.split(","))
        S = sorted(set(S) | {(r2 + i) * 14 + (c2 + j) for i in range(a.box) for j in range(a.box)})
    per_image = None
    if a.faces:                                                  # E11: each photo its own private set (its faces)
        import e11_faces_common as F
        kmin, kmax = (int(v) for v in a.faces.split(","))
        sets = F.private_sets(kmax, kmin)
        sets = sets[:len(sets) - 2 * a.heldout][:a.n]
        files = [f for f, _lab, _S in sets]
        per_image = {f: (lab, S_) for f, lab, S_ in sets}
    imgs = []
    for f in files:
        px = np.load(f) if f.endswith(".npy") else V.pixels(f)
        if per_image:
            label, S = per_image[f]
        rows = V.patch_rows(px)
        x = V.embed(store, rows)
        ref = R.late_replace(store, x, S, a.k)
        full = V.exact_blocks(store, x, 12)
        page = np.zeros_like(rows)
        page[ref["tokS"]] = rows[ref["tokS"]]
        imgs.append(dict(file=os.path.relpath(f, F.ROOT) if per_image else os.path.basename(f), page=page,
                         hid=ref["handover"], ref=ref, full=full, S=list(S), label=label if per_image else None))
    print(f"[gpu-late] {len(imgs)} images, k={a.k}, {len(S)} private patches, plan {a.plan}", flush=True)
    cfg = plan_config(resolve(a.plan), graphs=a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = os.path.join(a.work_dir, f"late_{os.getpid()}")
    run = LatePlannedPrefill(h.model, resolve(a.plan), cfg, keys, work_dir=work, pt_rows=a.pt_rows, stc_cache=a.stc_cache,
                             check_meta=False, log=lambda m: print(m, flush=True), ccmm_kappa=(256.0, 16.0), ccmm_sweep=(),
                             positions=imgs[0]["S"], k=a.k, direct_handover=a.direct_handover)
    assert run.n_layers == a.k + 1                           # the recorded stages: hand-over + k blocks
    t_set = time.time()
    run.set_image(imgs[0]["hid"], 0)
    run.prepare(imgs[0]["page"])
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    print(f"[gpu-late] compiled {len(progs)} stages: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {n_bts} refresh sites + "
          f"{n_fin[1]} PCMM/CCMM outputs = {n_bts + n_fin[1]} bts, registers {run.n_regs}", flush=True)
    if len(imgs) > 1:
        run.set_image(imgs[0]["hid"], 0, imgs[1]["S"])                  # the probe image's own private set
        dyn = run.find_dynamic(imgs[1]["page"], imgs[1]["hid"])
        run.set_image(imgs[0]["hid"], 0, imgs[0]["S"])
        print(f"[gpu-late] per-image plaintexts: {sum(len(v) for v in dyn.values())} names", flush=True)
    if a.compile_only:
        return 0
    run.setup(imgs[0]["page"])
    print(f"[gpu-late] planned setup {time.time() - t_set:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB", flush=True)
    res = []
    for i, im in enumerate(imgs):
        n_staged, twin_s = run.stage_image(im["page"], im["hid"], img=i, positions=im["S"])
        vals = run.input_values(im["page"])
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        wall = time.time() - t0
        hE = V.unscale(run.hidden())[:197]
        lg = V.logits(store, hE)[0]
        lo = V.logits(store, im["ref"]["hidden"][-1])[0]
        lf = V.logits(store, im["full"][-1])[0]
        _, s_o = report([lg], [lo])
        r = dict(file=im["file"], wall=wall, twin_s=twin_s, staged=n_staged, argmax=int(lg.argmax()), oracle=int(lo.argmax()),
                 full=int(lf.argmax()), kl_oracle=s_o["medianKL"], max_logit_diff=float(np.abs(lg - lo).max()),
                 k=len(im["S"]), label=im["label"])
        res.append(r)
        print(f"[gpu-late] {r['file']}: wall={wall:.2f}s (twin {twin_s:.0f}s, {n_staged} plaintexts restaged) argmax enc {r['argmax']} "
              f"oracle {r['oracle']} full {r['full']} KL(oracle||enc)={r['kl_oracle']:.3g} |logit diff|={r['max_logit_diff']:.3g}", flush=True)
    walls = [r["wall"] for r in res]
    print(f"[gpu-late] SUMMARY k={a.k} images={len(res)} s/image={np.median(walls):.2f} (walls {' '.join(f'{w:.2f}' for w in walls)}) "
          f"bts/image={n_bts + n_fin[1]} argmax=oracle {sum(r['argmax'] == r['oracle'] for r in res)}/{len(res)}", flush=True)
    json.dump(dict(args=vars(a), S=S, bts=n_bts + n_fin[1], results=res), open(a.out, "w"), indent=1)
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
