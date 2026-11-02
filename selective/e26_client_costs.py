#!/usr/bin/env python
"""E26: what the client pays and sends for one OWTE BERT page (measured, CPU only: Perseus's client never opens a
device). Records and compiles the drug-review two-stream page (one-way weights, 12 inputs per page) on the CPU like
`e8_run_gpu.py --compile-only`, then: the server's key manifest (`Session.key_manifest`, no device), the client's keygen
into a key bundle (size by kind, time), the block-0 input ciphertexts the client encrypts (count, level, wire bytes,
encryption time), and the last block's output ciphertexts it downloads and decrypts (count, level, bytes, time).

  python selective/e26_client_costs.py --plan bootstrap_placements/selective_e8/bert_drugscft_g12m --inputs 12 \
      --chunk0 6 --n-e-cap 15 --idx 0:200 --private ~/data/selective-fhe/e15/drugs_t128c_private.json \
      --weights ~/data/selective-fhe/exports/drugs_oneway/classic --out ~/data/selective-fhe/e26/client_bert_g12.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e8_harness as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--inputs", type=int, default=12)
    ap.add_argument("--chunk0", type=int, default=None)
    ap.add_argument("--n-e-cap", type=int, default=15)
    ap.add_argument("--idx", default="0:200")
    ap.add_argument("--private", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--work-dir", default=os.path.expanduser("~/data/selective-fhe/e26/work_client"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = H.setup(layers=12, plan=a.plan, weights=a.weights, n_e_cap=a.n_e_cap)
    D, st, cl, m = h.D, h.store, h.cl, h.model
    from perseus import _pc
    from perseus.examples.bert.planned import plan_config
    from sfhe.bert_selective_planned import BertSelectivePlannedPrefill
    idx = list(range(*[int(v) for v in a.idx.split(":")]))
    priv = H.load_private(a.private)
    idx = [i for i in idx if priv.get(i)]
    ins = [x for x in H.sentence_inputs(D, st, cl, idx, private=priv) if x["S"] and len(x["pos"]) <= a.n_e_cap]
    groups = H.pack_pages(ins, a.inputs, a.chunk0 if a.chunk0 is not None else a.inputs, m.blocks.t)
    grp, real = groups[0]
    page = m.page([x["rows_e"] for x in grp], [x["pos"] for x in grp], [x["ref"]["P_kv"] for x in grp], a.chunk0)
    cfg = plan_config(a.plan)
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    os.makedirs(a.work_dir, exist_ok=True)
    run = BertSelectivePlannedPrefill(m, a.plan, cfg, keys, page=page, work_dir=a.work_dir, pt_rows=28, stc_cache=16,
                                      check_meta=False, log=lambda s: print(s, flush=True), ccmm_kappa=(256.0, 16.0),
                                      ccmm_sweep=())
    run.fix_ws([page])
    run.prepare(page[0])
    out = dict(plan=a.plan, inputs_per_page=real, encrypted_tokens=[len(x["pos"]) for x in grp[:real]])
    # the key bundle the server asks the client for
    spec = _pc.Session.key_manifest(run.cfg, run.keys, **run.session_kwargs())
    js = json.loads(spec)
    sec = _pc.ClientSecret.generate("n32", seed=0)
    client = _pc.Client(sec)
    path = os.path.join(a.work_dir, "keys.pckb")
    t0 = time.time()
    kg = dict(client.keygen(spec, path, threads=a.threads))
    kg["wall_s"] = time.time() - t0
    info = _pc.bundle_info(path)
    out["keygen"] = dict(wall_s=kg["wall_s"], threads=a.threads, stats={k: v for k, v in kg.items()
                                                                        if isinstance(v, (int, float, str))},
                         bundle_bytes=info["bytes"], bytes_by_kind=info["bytes_by_kind"], entries=info["entries"],
                         spec_keys=[k for k in js if k not in ("contract", "params_hash")])
    os.remove(path)
    print(f"[e26-client] key bundle {info['bytes'] / 2**30:.2f} GiB {info['bytes_by_kind']}, keygen {kg['wall_s']:.1f}s",
          flush=True)
    # what the client encrypts: block 0's input slot vectors at their entry metas
    vals = run.input_values(page[0])
    metas = run.in_metas[0]
    enc_s, frames = [], []
    for v, (lv, dg) in zip(vals, metas):
        t = time.time()
        f = client.encrypt(np.asarray(v), int(lv))
        enc_s.append(time.time() - t)
        frames.append(f)
    out["upload"] = dict(ciphertexts=len(frames), levels=[int(lv) for lv, _d in metas], degs=[int(d) for _l, d in metas],
                         bytes=int(sum(len(f) for f in frames)), encrypt_s=float(sum(enc_s)),
                         per_input_bytes=float(sum(len(f) for f in frames)) / real, per_input_encrypt_s=float(sum(enc_s)) / real)
    # what it downloads: the last block's output ciphertexts (the page's encrypted rows), at their exit levels
    last = run.progs[run.labels[-1]]
    exit_levels = []
    for o in run.recs[len(run.labels) - 1].out_names:
        lv = None
        for attr in ("output_level", "output_meta"):
            if hasattr(last, attr):
                try:
                    r = getattr(last, attr)(last.output_index(o))
                    lv = int(r[0] if isinstance(r, (tuple, list)) else r)
                except Exception:
                    pass
        exit_levels.append(lv)
    dec_s, sizes = [], []
    for lv in exit_levels:
        f = client.encrypt(np.zeros(client.slots), int(lv if lv is not None else 0))
        sizes.append(len(f))
        t = time.time()
        client.decrypt(f)
        dec_s.append(time.time() - t)
    out["download"] = dict(ciphertexts=len(exit_levels), levels=exit_levels, bytes=int(sum(sizes)),
                           decrypt_s=float(sum(dec_s)), per_input_bytes=float(sum(sizes)) / real,
                           note="level None = exit level not exposed by the program: sized at the top level (upper bound)")
    out["program_attrs"] = [x for x in dir(last) if not x.startswith("_")]
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[e26-client] {json.dumps({k: v for k, v in out.items() if k != 'program_attrs'})[:3000]}", flush=True)


if __name__ == "__main__":
    main()
