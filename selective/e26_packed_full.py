#!/usr/bin/env python
"""E26 (baseline, measured per layout): full encryption packed like OWTE. With every content token private (S = all),
two-stream computes the original model, and OWTE's page layouts pack several short inputs of one client into one page.
Writes, per text task, the private files of the packed full baseline (S = all content tokens but the last one: the
runner needs one plaintext key per input, and a page's program and cost do not depend on which lane tokens are public)
and the token-count distribution with the share of inputs per layout (12 / page up to 13 content tokens, 4 / page up
to 29, otherwise a full page), plus the share of inputs with an empty private set. CPU only.

  python selective/e26_packed_full.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e26")
TASKS = ("drugsc", "mts", "sst2", "hatex", "enron")


def layout(k):
    return 12 if k <= 13 else 4 if k <= 29 else 1


def main():
    from transformers import AutoTokenizer
    from b5_text_sanitize import TASKS as T
    from b5b_typed_sanitize import rows_of
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for task in TASKS:
        _d, privf, L = T[task]
        priv = json.load(open(privf))
        out = {}
        for split in ("validation", "train"):
            rows = rows_of(task, split)
            out[split] = {}
            for i, (t, _l) in enumerate(rows):
                n = len(tok(t, truncation=True, max_length=L)["input_ids"])
                out[split][str(i)] = list(range(1, max(n - 2, 2)))       # content tokens 1..n-2, the last one public
        json.dump(out, open(f"{OUT}/{task}_all_private.json", "w"))
        rows = rows_of(task, "validation")
        n = np.array([len(tok(t, truncation=True, max_length=L)["input_ids"]) for t, _l in rows])
        has = np.array([bool(priv["validation"].get(str(i))) for i in range(len(rows))])
        k = n - 2                                                        # content tokens = encrypted under full enc.
        lay = np.array([layout(v) for v in k[has]])
        stats[task] = dict(inputs=len(rows), with_private=int(has.sum()), empty_private_share=float(1 - has.mean()),
                           tokens_median=float(np.median(n[has])), tokens_p10=float(np.percentile(n[has], 10)),
                           tokens_p90=float(np.percentile(n[has], 90)), max_len=L,
                           share_12=float((lay == 12).mean()), share_4=float((lay == 4).mean()),
                           share_full=float((lay == 1).mean()))
        print(f"[e26] {task}: {stats[task]}", flush=True)
    json.dump(stats, open(f"{OUT}/packed_full_layouts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
