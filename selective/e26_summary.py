#!/usr/bin/env python
"""E26 summary: OWTE's batched cost (the paper's Table 1 rows, from paper/scripts/make_data.py) against full encryption
packed the same way (measured per layout with S = every content token, weighted by each task's share of inputs per
layout), the extended encrypted Llama set, the bootstrap fit and the client costs. Writes e26/summary.json.

  python selective/e26_summary.py
"""
import glob
import json
import os
import sys

import numpy as np

D = os.path.expanduser("~/data/selective-fhe")
E = f"{D}/e26"
sys.path.insert(0, os.path.expanduser("~/perseus-up/paper/scripts"))


def per_input(path):
    d = json.load(open(path))
    res = d["results"] if isinstance(d, dict) else d
    walls = {r["page"]: r["page_wall"] for r in res}
    return sum(walls.values()) / len(res), sum(r["argmax"] == r["oracle"] for r in res), len(res)


def packed_full(md):
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.expanduser("~/perseus-up/selective"))
    from b5_text_sanitize import TASKS as T
    from b5b_typed_sanitize import rows_of
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    c12 = [per_input(f"{E}/pf_{t}_12.json") for t in ("sst2", "hatex")]
    c4 = per_input(f"{E}/pf_drugsc_4.json")
    cost12 = float(np.mean([c for c, _e, _n in c12]))
    cost4 = c4[0]
    owte = {r[0]: r for r in md.cost_rows()}
    names = {"drugsc": "Drug reviews", "mts": "Medical notes", "sst2": "SST-2 reviews", "hatex": "HateXplain posts",
             "enron": "Enron e-mails"}
    out = dict(layout_cost=dict(s_per_input_12=cost12, s_per_input_4=cost4, full_128=md.FULL_BERT, full_256=md.FULL_256,
                                runs={os.path.basename(f): dict(zip(("s_per_input", "exact", "ran"), per_input(f)))
                                      for f in sorted(glob.glob(f"{E}/pf_*.json"))}), tasks={})
    lay = json.load(open(f"{E}/packed_full_layouts.json"))
    for task, name in names.items():
        _d, privf, L = T[task]
        priv = json.load(open(privf))["validation"]
        rows = rows_of(task, "validation")
        cs = []
        for i, (t, _l) in enumerate(rows):
            if not priv.get(str(i)):
                continue                                                  # Table 1 counts inputs with private content
            n = len(tok(t, truncation=True, max_length=L)["input_ids"])
            k = n - 2
            cs.append(cost12 if k <= 13 else cost4 if k <= 29 else md.FULL_BERT if n <= 128 else md.FULL_256)
        r = owte[name]
        full_ref = r[7]                                                   # Table 1's full-encryption cost per input
        extra = {}
        if task == "enron":                  # OWTE's fallback inputs at the cost of their own length (<= 128: one page)
            res = json.load(open(f"{D}/e12/enron_ft.json"))
            res = res["results"] if isinstance(res, dict) else res
            ran = {x["i"] for x in res}
            fb = [len(tok(t, truncation=True, max_length=L)["input_ids"]) for i, (t, _l) in enumerate(rows)
                  if priv.get(str(i)) and i not in ran]
            t_run = sum({x["page"]: x["page_wall"] for x in res}.values())
            t_fb = sum(md.FULL_BERT if n <= 128 else md.FULL_256 for n in fb)
            extra = dict(owte_s_fallback_by_length=(t_run + t_fb) / len(cs), fallback_inputs=len(fb),
                         fallback_le128=sum(n <= 128 for n in fb))
        out["tasks"][task] = dict(name=name, table1_inputs=len(cs), owte_s=r[6], full_s=full_ref,
                                  packed_full_s=float(np.mean(cs)), speedup_table1=full_ref / r[6],
                                  speedup_vs_packed=float(np.mean(cs)) / r[6], **extra, **lay[task])
    return out


def llama():
    base = json.load(open(f"{D}/e18/p2/gpu_ts.json"))["results"]
    more = sum((json.load(open(f))["results"] for f in sorted(glob.glob(f"{E}/llts_c*.json"))), [])
    allr = base + more
    seen, uniq = set(), []
    for r in allr:
        if r["idx"] not in seen:
            seen.add(r["idx"])
            uniq.append(r)
    fixf = f"{E}/llts_padq.json"                     # the SFHE_TS_PADQ=r fix: reruns replace the zero-padding results
    fix = json.load(open(fixf))["results"] if os.path.exists(fixf) else []
    fixed = {r["idx"]: r for r in fix}
    before = {r["idx"]: r for r in uniq if r["idx"] in fixed}
    uniq = [fixed.get(r["idx"], r) for r in uniq]
    fixrep = dict(rerun=len(fix), idx=sorted(fixed), letter_same=sum(r["letter_enc"] == r["letter_ref"] for r in fix),
                  kl_max=float(max((r["kl"] for r in fix), default=0)),
                  before_letter_same=sum(r["letter_enc"] == r["letter_ref"] for r in before.values()),
                  wall_median=float(np.median([r["wall"] for r in fix])) if fix else None)
    ok = [r for r in uniq if r["n_tokens"] >= 150]
    bad = [r for r in uniq if r["n_tokens"] < 150]
    calibrated = dict(prompts=len(ok), letter_same=sum(r["letter_enc"] == r["letter_ref"] for r in ok),
                      kl_max=float(max(r["kl"] for r in ok)), tokens_range=[min(r["n_tokens"] for r in ok),
                                                                            max(r["n_tokens"] for r in ok)])
    short = dict(prompts=len(bad), idx=[r["idx"] for r in bad], tokens=[r["n_tokens"] for r in bad],
                 letter_same=sum(r["letter_enc"] == r["letter_ref"] for r in bad),
                 maxerr=[round(r["maxerr"]) for r in bad],
                 note="after the SFHE_TS_PADQ=r fix (rows above replaced by the reruns); cause: the encrypted page's "
                      "padding query lanes attended every padding key and drifted out of the polynomial ranges")
    return dict(padq_fix=fixrep, calibrated_range=calibrated, short_second_page=short, prompts=len(uniq), letter_same=sum(r["letter_enc"] == r["letter_ref"] for r in uniq),
                argmax_same=sum(r["argmax_enc"] == r["argmax_ref"] for r in uniq),
                correct=sum(r["letter_enc"] == r["label"] for r in uniq),
                kl_median=float(np.median([r["kl"] for r in uniq])), kl_max=float(max(r["kl"] for r in uniq)),
                wall_median=float(np.median([r["wall"] for r in uniq])),
                store_median=float(np.median([r["store_enc_s"] for r in uniq])),
                compute_median=float(np.median([r["wall"] - r["store_enc_s"] for r in uniq])),
                tokens_range=[min(r["n_tokens"] for r in uniq), max(r["n_tokens"] for r in uniq)],
                enc_median=float(np.median([r["n_enc"] for r in uniq])))


def main():
    import make_data as md
    s = dict(packed_full=packed_full(md), llama_240=llama())
    for k, f in (("bts_fit", "bts_fit.json"), ("client_bert_g12", "client_bert_g12.json"), ("ckks", "ckks_params.json"),
                 ("sylph_llama", "sylph_llama.json")):
        if os.path.exists(f"{E}/{f}"):
            v = json.load(open(f"{E}/{f}"))
            s[k] = (v.get("fits", v) if k == "bts_fit" else v["summary"] if k == "sylph_llama"
                    else {kk: vv for kk, vv in v.items() if kk != "program_attrs"})
    json.dump(s, open(f"{E}/summary.json", "w"), indent=1)
    for t, v in s["packed_full"]["tasks"].items():
        print(f"{v['name']:18s} OWTE {v['owte_s']:6.2f}s  full {v['full_s']:6.2f}s ({v['speedup_table1']:5.1f}x)  "
              f"packed full {v['packed_full_s']:6.2f}s ({v['speedup_vs_packed']:5.1f}x)  share 12/4/full "
              f"{v['share_12']:.2f}/{v['share_4']:.2f}/{v['share_full']:.2f}  empty S {v['empty_private_share']:.2f}")
    print(json.dumps(s["llama_240"]))


if __name__ == "__main__":
    main()
