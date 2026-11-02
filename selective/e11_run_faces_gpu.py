#!/usr/bin/env python
"""E11 on the GPU: face photos (private = the detected faces' patches) of one size bucket, packed several per page by lane
fit, ViT two-stream route B through the planned path (XPT_PAD=1). Each photo is scored against its exact two-stream
oracle, the full model and its label. The bucket's last 2 x HELDOUT photos (the calibration's) are left out.

  XPT_PAD=1 python selective/e11_run_faces_gpu.py --plan bootstrap_placements/selective_e3/vit_faces_s --k-min 1 --k-max 20 \
      --cap 21 --inputs 8 --chunk0 4 --heldout 24 [--n N] [--weights DIR]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H
import e8_harness as EH
import e8_run_gpu as R8
import e11_faces_common as F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs (upstream\'s --graphs)")
    ap.add_argument("--k-min", type=int, required=True)
    ap.add_argument("--k-max", type=int, required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--inputs", type=int, required=True)
    ap.add_argument("--chunk0", type=int, default=None)
    ap.add_argument("--heldout", type=int, required=True)
    ap.add_argument("--n", type=int, default=0, help="photos to score (0: the whole bucket minus the held-out ones)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=8)
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e11/work"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("SEL_ROUTE", "pcmm")
    assert os.environ.get("XPT_PAD") in ("1", "2"), "run with XPT_PAD=2 (padding lanes kept in range)"
    from sfhe.vit_selective import SelectivePrefill
    h = H.setup(layers=12, cls=SelectivePrefill, plan=a.plan, weights=a.weights)
    V, store, m = h.V, h.store, h.model
    m.n_e_cap = a.cap
    from perseus import _pc
    from perseus._paths import resolve
    from perseus.examples.bert.planned import plan_config
    from sfhe.vit_layout_planned import VitLayoutPlannedPrefill
    sets = F.private_sets(a.k_max, a.k_min)
    assert F.SET != "ages" or 2 * a.heldout == F.AGES_TAIL, "ages: --heldout AGES_TAIL / 2 (the train photos' tail)"
    sets = sets[:len(sets) - 2 * a.heldout]
    if a.n:
        sets = sets[:a.n]
    t0 = time.time()
    ins = F.inputs(V, store, sets)
    groups = EH.pack_pages(ins, a.inputs, a.chunk0 if a.chunk0 is not None else a.inputs, m.blocks.t)
    pages = [dict(ins=g, page=m.page([x["rows_e"] for x in g], [x["pos"] for x in g], [x["ref"]["P_kv"] for x in g],
                                     a.chunk0)) for g, _real in groups]
    print(f"[e11-gpu] {len(ins)} photos ({a.k_min}-{a.k_max} private patches) on {len(pages)} pages of {a.inputs} "
          f"(oracles {time.time() - t0:.0f}s), plan {a.plan}", flush=True)
    probe = pages[1]["page"] if len(pages) > 1 else pages[0]["page"]
    plan = resolve(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    work = os.path.join(a.work_dir, f"{os.path.basename(a.plan)}_{os.getpid()}")
    os.makedirs(work, exist_ok=True)
    run = VitLayoutPlannedPrefill(m, plan, plan_config(plan, graphs=a.graphs), keys, page=pages[0]["page"], work_dir=work, pt_rows=a.pt_rows,
                                  stc_cache=a.stc_cache, check_meta=False, log=lambda s: print(s, flush=True),
                                  ccmm_kappa=(256.0, 16.0), ccmm_sweep=())
    t_set = time.time()
    run.fix_ws([pg["page"] for pg in pages])
    run.prepare(pages[0]["page"][0])
    progs = [run.progs[lab] for lab in run.labels]
    bts = sum(p.n_bootstraps for p in progs) + sum(p.n_finishes[1] for p in progs)
    print(f"[e11-gpu] compiled: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {bts} bts per page", flush=True)
    dyn = run.find_dynamic(probe, pages[2]["page"] if len(pages) > 2 else None)
    grp = getattr(run, "dyn_group", {}) or {}
    print(f"[e11-gpu] per-page plaintexts: {sum(len(v) for v in dyn.values())} names in {len(set(grp.values())) or 'no'} groups",
          flush=True)
    R8._RUN, R8._PAGES = run, [pg["page"] for pg in pages]
    pts, ahead = {}, None
    if any(dyn.values()) and len(pages) > 1:
        t0 = time.time()
        if R8.ahead_on():
            ahead = R8.PagesAhead(R8._page_pts, list(range(1, len(pages))), a.workers)
            print(f"[e11-gpu] page plaintexts for {len(pages) - 1} pages on {a.workers} workers, read while the device runs",
                  flush=True)
        else:
            for i, names, vals in R8.page_pool(R8._page_pts, list(range(1, len(pages))), a.workers):
                pts[i] = (names, vals)
            print(f"[e11-gpu] page plaintexts for {len(pts)} pages on {a.workers} workers: {time.time() - t0:.0f}s", flush=True)
    run.setup(pages[0]["page"][0])
    run.staged_shape = run.shape()
    print(f"[e11-gpu] planned setup {time.time() - t_set:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB",
          flush=True)
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    res = []
    for p, pg in enumerate(pages):
        rows, _pos, layout, _pkv = pg["page"]
        t_w = time.time()
        got = ahead.get(p) if ahead is not None else ((p,) + pts.pop(p) if p in pts else None)
        wait_s = time.time() - t_w
        n_st, st_s = run.stage_page(pg["page"], R8._unpack_pts(*got[1:]) if got is not None else None)
        vals = run.input_values(rows)
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        wall = time.time() - t0
        hE = V.unscale(run.hidden())
        for q, (x, r0, lx) in enumerate(zip(pg["ins"], m.first_rows(layout), layout)):
            if x.get("filler"):
                continue
            lg = V.logits(store, hE[r0:r0 + lx[-2]])[0]
            lo = V.logits(store, x["ref"]["E_hidden"][-1])[0]
            kl = float((sm(lo) * (np.log(sm(lo) + 1e-30) - np.log(sm(lg) + 1e-30))).sum())
            r = dict(file=os.path.relpath(x["file"], F.ROOT), page=p, slot=q, k=len(x["S"]), label=x["label"], page_wall=wall,
                     stage_s=st_s, wait_s=wait_s, argmax=int(lg.argmax()), oracle=int(lo.argmax()), full=int(x["full"].argmax()), kl_oracle=kl,
                     max_logit_diff=float(np.abs(lg - lo).max()))
            res.append(r)
            print(f"[e11-gpu] page {p} slot {q} {r['file']} k={r['k']}: page wall {wall:.2f}s (stage {st_s:.1f}s) enc {r['argmax']} "
                  f"oracle {r['oracle']} full {r['full']} label {r['label']} KL {kl:.3g}", flush=True)
    if ahead is not None:
        ahead.close()
        print(f"[e11-gpu] page plaintexts recorded in {ahead.done_s or 0:.0f}s alongside the device; device waited "
              f"{sum({r['page']: r['wait_s'] for r in res}.values()):.0f}s for them", flush=True)
    walls = list({r["page"]: r["page_wall"] for r in res}.values())
    n = len(res)
    print(f"[e11-gpu] SUMMARY bucket {a.k_min}-{a.k_max} photos={n} pages={len(walls)} s/page={np.median(walls):.2f} "
          f"s/photo={sum(walls) / n:.2f} bts/page={bts} enc=oracle {sum(r['argmax'] == r['oracle'] for r in res)}/{n} "
          f"=full {sum(r['argmax'] == r['full'] for r in res)}/{n} labels enc {sum(r['argmax'] == r['label'] for r in res)} "
          f"oracle {sum(r['oracle'] == r['label'] for r in res)} full {sum(r['full'] == r['label'] for r in res)} "
          f"medianKL={np.median([r['kl_oracle'] for r in res]):.3g}", flush=True)
    json.dump(dict(args=vars(a), bts_page=bts, results=res), open(a.out, "w"), indent=1)
    run.close()


if __name__ == "__main__":
    R8.run_main(main)
