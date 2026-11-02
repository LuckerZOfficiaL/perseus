#!/usr/bin/env python
"""E26: Sylph's setting on Llama-3.1-8B (Park et al. 2026): the prefix before the first private token in plaintext (its
K / V encrypted by the server as stored pages, as OWTE's public pages), everything from the first private token on
encrypted. Causal attention makes this the original model, so its accuracy is the full model's. Cost per prompt is
DERIVED from our MEASURED per-page timings (a page's cost does not depend on how full it is or which positions it
holds, E18: OWTE's 510-token page = the full prefill's page 4, 233.3 vs 233.0 s):
  - ~240-token drug prompts (kc256 configuration): encrypted page at index 0 / 1 = the full prefill's pages 0 / 1
    (75.2 s, 102.4 s), at index 2 = the two-stream page reading 2 stored pages (130.1 s); stored page 6.9 s.
  - 510-token e-mails (kc1024): page at index q = the full prefill's page q (gpu_fullL_p{q}, q <= 4); stored page =
    page 4's store encryption / 4.
Two placements of the suffix, the cheaper one kept: (a) as OWTE places its encrypted page, after all stored pages
(q = ceil(p0 / 128) stored, ceil(s / 128) encrypted pages), (b) 128-aligned true positions (pages floor(p0 / 128) ..
last encrypted, floor(p0 / 128) stored).

  python selective/e26_sylph_llama.py
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
E = f"{D}/e26"


def cost(n, p0, C, store):
    s = n - p0
    q, k = math.ceil(p0 / 128), math.ceil(s / 128)
    a = q * store + sum(C[j] for j in range(q, q + k))
    qb, last = p0 // 128, (n - 1) // 128
    b = qb * store + sum(C[j] for j in range(qb, last + 1))
    return min(a, b), (q, k) if a <= b else (qb, last - qb + 1)


def main():
    import e18_llama_ts as TS
    from transformers import AutoTokenizer
    tok = "meta-llama/Llama-3.1-8B-Instruct"                                    # TS.prompt loads it by name
    # measured page costs
    p0 = json.load(open(f"{D}/e18/p2/gpu_full_p0.json"))["results"]
    p1 = json.load(open(f"{D}/e18/p2/gpu_full_p1.json"))["results"]
    ts = json.load(open(f"{D}/e18/p2/gpu_ts.json"))["results"]
    C_short = [float(np.median([r["wall"] for r in p0[1:]])), float(np.median([r["wall"] - r["store_enc_s"] for r in p1[1:]])),
               float(np.median([r["wall"] - r["store_enc_s"] for r in ts]))]
    store_short = float(np.median([r["store_enc_s"] for r in ts])) / 2
    full_short = C_short[0] + C_short[1]
    page = {}
    for q in range(5):
        r = json.load(open(f"{D}/e18/p2/gpu_fullL_p{q}.json"))["results"][-1]
        page[q] = (r["wall"] - r["store_enc_s"], r["store_enc_s"])
    C_long = [page[q][0] for q in range(5)]
    store_long = page[4][1] / 4
    full_long = sum(C_long[:4])
    out = dict(measured_page_costs=dict(short_kc256=C_short, short_store_page=store_short, long_kc1024=C_long,
                                        long_store_page=store_long, full_short=full_short, full_long=full_long))
    # drug prompts: the OWTE set (E18's 20 + E26's 80)
    owte = {r["idx"]: r for f in [f"{D}/e18/p2/gpu_ts.json"] + sorted(glob.glob(f"{E}/llts_c*.json"))
            for r in json.load(open(f))["results"]}
    fix = f"{E}/llts_padq.json"
    if os.path.exists(fix):
        owte.update({r["idx"]: r for r in json.load(open(fix))["results"]})
    rows = []
    for idx, r in sorted(owte.items()):
        a = argparse.Namespace(idx=idx, split="validation", tokenizer=tok, owt=0, owt_page=1, enron=0, cat=1)
        ids, S, _letters, _label = TS.prompt(a)
        p = min(S) if S else len(ids) - 1                               # no private token: only the readout, as OWTE
        c, (q, k) = cost(len(ids), p, C_short, store_short)
        rows.append(dict(idx=idx, n=len(ids), first_private=p, suffix=len(ids) - p, stored=q, enc_pages=k, sylph_s=c,
                         owte_s=r["wall"], owte_enc=r["n_enc"]))
    out["drugs_240"] = rows
    # 510-token e-mails: the 20 measured OWTE prompts
    pr = json.load(open(f"{D}/e18/p2/prompts_enron_L510.json"))
    al = json.load(open(f"{D}/e18/p2/gpu_ts_e510_all20.json"))["results"]
    rows = []
    for r in al:
        P = pr[r["idx"]]
        p = min(P["S"]) if P["S"] else len(P["ids"]) - 1
        n = len(P["ids"])
        c, (q, k) = cost(n, p, C_long, store_long)
        rows.append(dict(idx=r["idx"], n=n, first_private=p, suffix=n - p, stored=q, enc_pages=k, sylph_s=c,
                         owte_s=r["wall"], owte_enc=r["n_enc"]))
    out["enron_510"] = rows
    summ = {}
    for key, full in (("drugs_240", full_short), ("enron_510", full_long)):
        R = out[key]
        sfx = np.array([x["suffix"] for x in R])
        summ[key] = dict(prompts=len(R), full_s=full, sylph_s_median=float(np.median([x["sylph_s"] for x in R])),
                         owte_s_median=float(np.median([x["owte_s"] for x in R])),
                         suffix_median=float(np.median(sfx)), suffix_p10=float(np.percentile(sfx, 10)),
                         suffix_p90=float(np.percentile(sfx, 90)), suffix_min=int(sfx.min()), suffix_max=int(sfx.max()),
                         enc_pages=dict(zip(*[v.tolist() for v in np.unique([x["enc_pages"] for x in R], return_counts=True)])),
                         owte_enc_median=float(np.median([x["owte_enc"] for x in R])),
                         note="sylph_s derived from measured per-page costs; owte_s measured (wall incl. store encryption)")
        print(f"[e26-sylph] {key}: {summ[key]}", flush=True)
    out["summary"] = summ
    json.dump(out, open(f"{E}/sylph_llama.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
