#!/usr/bin/env python
"""E8 on the GPU: two-stream BERT, `--inputs` SST-2 sentences' encrypted tokens per page, through the planned path.
Sentences are taken in index order and grouped into pages (a last partial page is filled with repeats, not scored); each
sentence is scored against its own exact two-stream oracle (e8_reference) and the full model.

  python selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g6 --inputs 6 --idx 0:60 \
      [--weights EXPORT_DIR] [--frac 0.3] [--workers 12] [--compile-only]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e8_harness as H

_RUN, _PAGES, _WORK = None, None, None
POOL_LOCK = os.environ.get("SFHE_POOL_LOCK", os.path.expanduser("~/data/selective-fhe/pool.lock"))
POOL_SLOTS = int(os.environ.get("SFHE_POOL_SLOTS", "2"))           # pools that may record at once on the host
POOL_EXTRA_GB = float(os.environ.get("SFHE_POOL_EXTRA_GB", "300"))  # a second slot only above this much MemAvailable


def _mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def acquire_pool_slot():
    """An open, flock-ed lock file: slot 0 is POOL_LOCK itself (what older runs hold), slot k > 0 POOL_LOCK.k, taken
    only while the host has POOL_EXTRA_GB available (a pool's workers grow by tens of GB each). Waits until one is free."""
    import fcntl
    t0, said = time.time(), False
    while True:
        for k in range(max(1, POOL_SLOTS)):
            if k and _mem_available_gb() < POOL_EXTRA_GB:
                break
            lk = open(POOL_LOCK if k == 0 else f"{POOL_LOCK}.{k}", "w")
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lk.close()
                continue
            if time.time() - t0 > 5:
                print(f"[pool] waited {time.time() - t0:.0f}s for a page-recording slot (got {k})", flush=True)
            return lk
        if not said:
            print("[pool] waiting for a page-recording slot", flush=True)
            said = True
        time.sleep(10)


def _few_threads(n=4):
    """Worker init: cap numpy's OpenBLAS pool (the GPU runners on the other cards share the CPU)."""
    import ctypes
    for line in open("/proc/self/maps"):
        path = line.split()[-1]
        if "openblas" in path.lower():
            lib = ctypes.CDLL(path)
            for sym in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads"):
                if hasattr(lib, sym):
                    getattr(lib, sym)(n)
                    return


def _page_pts(i):
    """Worker (forked before the device session): page i's dynamic plaintexts, each distinct array once (kept in RAM:
    the data quota has no room for ~0.2 GB per page)."""
    import hashlib
    pts = _RUN.page_plaintexts(_PAGES[i])
    uniq, names = {}, []
    for b, d in pts.items():
        for nm, v in d.items():
            h = hashlib.blake2b(v.tobytes(), digest_size=12).hexdigest()
            j = uniq.setdefault(h, (len(uniq), v))[0]
            names.append((b, nm, j))
    return i, names, [v for _j, v in sorted(uniq.values(), key=lambda x: x[0])]


def page_pool(fn, items, workers):
    """fn over items in forked workers (each a copy of the twin: ~65-100 GB once recording), one pool on the host at a time
    (a lock under ~/data/selective-fhe; several pools at once ran the host out of memory on 2026-10-29), failing at once if a
    worker dies (a plain Pool waits forever for a dead worker's task)."""
    import fcntl
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with acquire_pool_slot():
        with ProcessPoolExecutor(max(1, min(workers, len(items))), mp_context=mp.get_context("fork"),
                                 initializer=_few_threads) as ex:
            for fut in as_completed([ex.submit(fn, i) for i in items]):
                yield fut.result()


class PagesAhead:
    """page_pool's work started before the device session and read in page order while the device runs (the device
    starts as soon as page 1 is recorded instead of after the last page). A fork pool launches all its workers at the
    first submit (CPython: no fork after its manager thread starts), so every fork happens here, before the device
    session opens. The host's page-recording lock is held until the last page is recorded, not until the device ends.
    SFHE_PAGES_AHEAD=0: record every page first (page_pool), as before."""
    live = []                                     # pools to abort if the run dies (the __main__ guards below)

    def __init__(self, fn, items, workers):
        import fcntl
        import threading
        from concurrent.futures import ProcessPoolExecutor
        self.lk = acquire_pool_slot()
        self.t0, self.done_s, self.n = time.time(), None, len(items)
        self.ex = ProcessPoolExecutor(max(1, min(workers, len(items))), mp_context=mp.get_context("fork"),
                                      initializer=_few_threads)
        self._left, self._mu = len(items), threading.Lock()
        self.futs = {i: self.ex.submit(fn, i) for i in items}
        assert len(self.ex._processes) == self.ex._max_workers, "the pool forked lazily"
        for f in self.futs.values():
            f.add_done_callback(self._one_done)
        PagesAhead.live.append(self)

    def _one_done(self, _f):
        with self._mu:
            self._left -= 1
            if self._left == 0:
                self.done_s = time.time() - self.t0
                self._unlock()

    def _unlock(self):
        import fcntl
        if not self.lk.closed:
            fcntl.flock(self.lk, fcntl.LOCK_UN)
            self.lk.close()

    def get(self, i):
        """Page i's (i, names, vals) (waits for it), or None for a page the pool does not record."""
        f = self.futs.pop(i, None)
        return None if f is None else f.result()

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)
        with self._mu:
            self._unlock()
        if self in PagesAhead.live:
            PagesAhead.live.remove(self)

    def abort(self):
        """The run failed: stop the workers now (a plain shutdown waits for their pages) and free the host lock."""
        for p in list(getattr(self.ex, "_processes", {}).values()):
            p.terminate()
        self.ex.shutdown(wait=False, cancel_futures=True)
        with self._mu:
            self._unlock()


def run_main(main):
    """sys.exit(main()); if main raises: abort any page pool (else its workers outlive the error, holding the lock),
    print the traceback and leave at once (an orderly interpreter exit hung once with the device session open, keeping
    the card's memory for 48 min)."""
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        for a in list(PagesAhead.live):
            a.abort()
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def ahead_on():
    return os.environ.get("SFHE_PAGES_AHEAD", "1") not in ("", "0")


def _unpack_pts(names, vals):
    out = {}
    for b, nm, j in names:
        out.setdefault(int(b), {})[nm] = vals[j]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--graphs", action="store_true", help="planned: replay each block as CUDA graphs (upstream\'s --graphs)")
    ap.add_argument("--inputs", type=int, default=6, help="sentences per encrypted page")
    ap.add_argument("--idx", default="0:12", help="SST-2 validation indices a:b or a,b,c")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-e-cap", type=int, default=10)
    ap.add_argument("--chunk0", type=int, default=None, help="inputs per page on chunk 0 (default all); the rest on chunk 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=None, help="perseus export dir of the model (default: the shipped SST-2 BERT)")
    ap.add_argument("--workers", type=int, default=4,
                    help="CPU processes re-recording pages ahead of the device (each holds a twin copy: ~30+ GB)")
    ap.add_argument("--pt-rows", type=int, default=28)
    ap.add_argument("--stc-cache", type=int, default=16)
    ap.add_argument("--key-layout", choices=("dense", "packed"), default="packed")
    ap.add_argument("--compile-only", action="store_true", help="record + compile + page plaintexts on the CPU, no device")
    ap.add_argument("--private", default=None,
                    help="detector file (e11_ner_sst2.py): its private tokens; sentences without any are skipped (plaintext)")
    ap.add_argument("--pages", default=None, help="run only these pages (comma-separated); weight scales still over all")
    ap.add_argument("--check", action="store_true", help="per page: every block's output against each input's oracle")
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e8/work"))
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e8/gpu_results.json"))
    a = ap.parse_args()
    h = H.setup(layers=12, plan=a.plan, weights=a.weights, n_e_cap=a.n_e_cap)
    D, st, cl, m = h.D, h.store, h.cl, h.model
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from sfhe.bert_selective_planned import BertSelectivePlannedPrefill

    idx = (list(range(*[int(v) for v in a.idx.split(":")])) if ":" in a.idx else [int(v) for v in a.idx.split(",")])
    t_in = time.time()
    priv = None
    if a.private:
        priv = H.load_private(a.private)
        idx = [i for i in idx if priv.get(i)]
    ins = H.sentence_inputs(D, st, cl, idx, a.frac, a.seed, a.k_max, private=priv)
    if priv is not None:
        ins = [x for x in ins if x["S"]]
    from sfhe.dense_xpt import MB
    over = [x["i"] for x in ins if not 0 < len(x["ref"]["NS"]) <= MB * 64]
    if over:                                                             # a page takes 1..MB * 64 plaintext keys per input
        print(f"[e8-gpu] {len(over)} inputs left out: their plaintext keys do not fit one page's {MB * 64} "
              f"(indices {over[:10]}{' ...' if len(over) > 10 else ''})", flush=True)
        ins = [x for x in ins if x["i"] not in set(over)]
    big = [x["i"] for x in ins if len(x["pos"]) > a.n_e_cap]            # CLS + private + SEP over an input's lanes
    if big:                                                              # (deployed: these go fully encrypted)
        print(f"[e8-gpu] {len(big)} inputs left out: more than {a.n_e_cap} encrypted tokens "
              f"(indices {big[:10]}{' ...' if len(big) > 10 else ''})", flush=True)
        ins = [x for x in ins if x["i"] not in set(big)]
    G = a.inputs
    if priv is not None:                                                 # sizes vary: pack by lanes
        groups = H.pack_pages(ins, G, a.chunk0 if a.chunk0 is not None else G, m.blocks.t)
    else:
        groups = []
        for p in range(-(-len(ins) // G)):
            grp = ins[p * G:(p + 1) * G]
            groups.append((grp + [dict(ins[j % len(ins)], filler=True) for j in range(G - len(grp))], len(grp)))
    pages = [dict(ins=grp, real=real, page=m.page([x["rows_e"] for x in grp], [x["pos"] for x in grp],
                                                 [x["ref"]["P_kv"] for x in grp], a.chunk0)) for grp, real in groups]
    if len(pages) == 1:                                                  # find_dynamic needs a second page
        rot = ins[1:] + ins[:1] if len(ins) > 1 else ins
        grp = (rot * G)[:G]
        probe = m.page([x["rows_e"] for x in grp], [x["pos"] for x in grp], [x["ref"]["P_kv"] for x in grp], a.chunk0)
    else:
        probe = pages[1]["page"]
    print(f"[e8-gpu] {len(ins)} sentences on {len(pages)} pages of {G} (oracles {time.time() - t_in:.0f}s), plan {a.plan}; "
          f"page 0 layout {pages[0]['page'][2]}", flush=True)

    cfg = plan_config(a.plan, graphs=a.graphs)
    keys = _pc.KeySource.generate([], conj=True, layout=a.key_layout)
    work = os.path.join(a.work_dir, f"g{G}_{os.getpid()}")
    os.makedirs(work, exist_ok=True)
    run = BertSelectivePlannedPrefill(m, a.plan, cfg, keys, page=pages[0]["page"], work_dir=work, pt_rows=a.pt_rows,
                                      stc_cache=a.stc_cache, check_meta=False, log=lambda s: print(s, flush=True),
                                      ccmm_kappa=(256.0, 16.0), ccmm_sweep=())
    t_set = time.time()
    ws = run.fix_ws([pg["page"] for pg in pages])
    print(f"[e8-gpu] fixed weight scales b0.xqk {ws.get('b0.xqk')} b0.xpv {ws.get('b0.xpv')}", flush=True)
    run.prepare(pages[0]["page"][0])
    progs = [run.progs[lab] for lab in run.labels]
    n_bts = sum(p.n_bootstraps for p in progs)
    n_fin = [sum(p.n_finishes[i] for p in progs) for i in (0, 1)]
    bts = n_bts + n_fin[1]
    print(f"[e8-gpu] compiled {len(progs)} blocks: record {run.record_s:.1f}s compile {run.compile_s:.1f}s; {n_bts} refresh "
          f"sites + {n_fin[1]} PCMM/CCMM outputs = {bts} bts per page", flush=True)
    t0 = time.time()
    n_more = int(os.environ.get("SFHE_PROBES", "0"))                     # extra probe pages for the grouping
    more = list(range(3, min(3 + n_more, len(pages))))
    for q in range(len(pages[0]["page"][2])):                            # + the pages with each slot's extreme layout:
        for key in (lambda lx: lx[2], lambda lx: lx[1] + lx[2], lambda lx: lx[3]):   # most encrypted lanes, last lane,
            j = max(range(len(pages)), key=lambda k: key(pages[k]["page"][2][q]))    # most plaintext keys (E15m: masks
            if j > 2 and j not in more:                                  # that change only at nearly full lane groups
                more.append(j)                                           # looked static on the first probes)
    dyn = run.find_dynamic(probe, pages[2]["page"] if len(pages) > 2 else None,
                           more=[pages[j]["page"] for j in more])
    print(f"[e8-gpu] probe pages for the grouping: {[1, 2] + more}", flush=True)
    print(f"[e8-gpu] per-page plaintexts: {sum(len(v) for v in dyn.values())} names ({time.time() - t0:.0f}s)", flush=True)
    global _RUN, _PAGES, _WORK
    _RUN, _PAGES, _WORK = run, [pg["page"] for pg in pages], work
    pts_paths, ahead = {}, None
    todo = sorted(int(v) for v in a.pages.split(",")) if a.pages else list(range(len(pages)))
    if any(dyn.values()) and len(pages) > 1:
        t0 = time.time()
        need = [p for p in todo if p > 0]
        if ahead_on() and not a.compile_only:
            ahead = PagesAhead(_page_pts, need, a.workers)
            print(f"[e8-gpu] page plaintexts for {len(need)} pages on {a.workers} workers, read while the device runs",
                  flush=True)
        else:
            for i, names, vals in page_pool(_page_pts, need, a.workers):
                pts_paths[i] = (names, vals)
            nv = [len(v) for _n, v in pts_paths.values()]
            print(f"[e8-gpu] page plaintexts for {len(pts_paths)} pages on {a.workers} workers: {time.time() - t0:.0f}s; "
                  f"{np.mean(nv):.0f} distinct arrays per page ({sum(nv) * 2 ** 19 / 2 ** 30:.1f} GB in RAM)", flush=True)
    if a.compile_only:
        return 0
    run.setup(pages[0]["page"][0])
    run.staged_shape = run.shape()                                       # the session holds page 0's values
    print(f"[e8-gpu] planned setup {time.time() - t_set:.1f}s; device free {run.fhe.mem_info()[0] / 2 ** 30:.2f} GB",
          flush=True)
    res = []
    for p in todo:
        pg = pages[p]
        rows, _pos, layout, _pkv = pg["page"]
        t_w = time.time()
        got = ahead.get(p) if ahead is not None else ((p,) + pts_paths.pop(p) if p in pts_paths else None)
        wait_s = time.time() - t_w
        pts = _unpack_pts(*got[1:]) if got is not None else None
        n_staged, stage_s = run.stage_page(pg["page"], pts)
        vals = run.input_values(rows)
        t0 = time.time()
        run.run(vals=vals)
        run.fhe.device_sync()
        wall = time.time() - t0
        hE = run.hidden()
        if a.check:
            r0 = m.first_rows(layout)
            for b in range(m.n_layers):
                hb = run.hidden(b)
                e = [np.abs(hb[s0:s0 + m.entry(lx)[0]] - x["ref"]["E_hidden"][b]).max()
                     for s0, lx, x in zip(r0, layout, pg["ins"]) if not x.get("filler")]
                print(f"[e8-gpu] page {p} block {b} maxerr per input {' '.join(f'{v:.3g}' for v in e)}", flush=True)
        for q, (x, s0, lx) in enumerate(zip(pg["ins"], m.first_rows(layout), layout)):
            if x.get("filler"):
                continue
            n_e, n_p = m.entry(lx)
            lg = D.head(hE[s0], cl)
            lo = x["ref"]["logits"]
            r = dict(i=x["i"], page=p, slot=q, n=x["n"], n_e=n_e, n_p=n_p, label=x["label"], wall=wall / G, page_wall=wall,
                     stage_s=stage_s, wait_s=wait_s, staged=n_staged, logits=lg.tolist(), argmax=int(lg.argmax()), oracle=int(lo.argmax()),
                     full=int(x["full"].argmax()), redact=int(x["ref"]["logits_redact"].argmax()),
                     kl_oracle=D.kl2(lg, lo), max_logit_diff=float(np.abs(lg - lo).max()))
            res.append(r)
            print(f"[e8-gpu] page {p} input {q} sst2[{x['i']}] n={x['n']} |S|={len(x['S'])}: page wall={wall:.2f}s (stage "
                  f"{stage_s:.1f}s, {n_staged} items) argmax enc {r['argmax']} oracle {r['oracle']} full {r['full']} label "
                  f"{r['label']} KL={r['kl_oracle']:.3g} |logit diff|={r['max_logit_diff']:.3g}", flush=True)
    if ahead is not None:
        ahead.close()
        print(f"[e8-gpu] page plaintexts recorded in {ahead.done_s or 0:.0f}s alongside the device; device waited "
              f"{sum({r['page']: r['wait_s'] for r in res}.values()):.0f}s for them", flush=True)
    walls = list({r["page"]: r["page_wall"] for r in res}.values())
    n = len(res)
    print(f"[e8-gpu] SUMMARY inputs/page={G} pages={len(walls)} sentences={n} s/page={np.median(walls):.2f} "
          f"s/sentence={sum(walls) / n:.2f} bts/page={bts} argmax=oracle {sum(r['argmax'] == r['oracle'] for r in res)}/{n} "
          f"=full {sum(r['argmax'] == r['full'] for r in res)}/{n} labels enc {sum(r['argmax'] == r['label'] for r in res)} "
          f"oracle {sum(r['oracle'] == r['label'] for r in res)} full {sum(r['full'] == r['label'] for r in res)} "
          f"redact {sum(r['redact'] == r['label'] for r in res)} medianKL={np.median([r['kl_oracle'] for r in res]):.3g}",
          flush=True)
    json.dump(dict(args=vars(a), bts_page=bts, results=res, setup_s=time.time() - t_set), open(a.out, "w"), indent=1)
    run.close()
    return 0


if __name__ == "__main__":
    run_main(main)
