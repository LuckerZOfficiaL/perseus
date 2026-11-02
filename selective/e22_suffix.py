#!/usr/bin/env python
"""E22 (baseline, measured): encrypted suffix only (Park et al. 2026 / Sylph's setting: a public prefix in plaintext, the
private part encrypted, which must come last). On a text whose private words sit anywhere, the encrypted part is
everything from the first private word to the end. In our pipeline that is two-stream with the private set
S = [first private position, last content token]: the prefix never sees the suffix (as in a causal model), the suffix
sees everything. Writes the suffix detector files (both splits) next to the task's own and reports plaintext accuracy
(full / two-stream = suffix baseline / redaction of the suffix) for the given weights, and the encrypted-token counts.

  python selective/e22_suffix.py --task drugsc --models ~/data/selective-fhe/e15/drugs_oneway,~/data/selective-fhe/e15/drugs_bert
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e22")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("drugsc", "mts"))
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from b5_text_sanitize import TASKS
    from b5b_typed_sanitize import rows_of
    import e13_hatexplain as H13
    _data, privf, T = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    priv = json.load(open(privf))
    out = {}
    for split in ("validation", "train"):
        rows = rows_of(a.task, split)
        sp = {}
        for i, (t, _l) in enumerate(rows):
            P = [p for p in priv[split].get(str(i), []) if p > 0]
            if not P:
                sp[str(i)] = []
                continue
            n = len(tok(t, truncation=True, max_length=T)["input_ids"])
            sp[str(i)] = list(range(min(P), n - 1))                       # first private piece .. last content piece
        out[split] = sp
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/{a.task}_suffix_private.json"
    json.dump(out, open(f, "w"))
    k = np.array([len(v) for v in out["validation"].values() if v])
    kp = np.array([len(v) for v in priv["validation"].values() if v])
    print(f"[e22] {a.task}: {len(k)} validation texts with private words; encrypted tokens, ours median {np.median(kp):.0f} "
          f"(p90 {np.percentile(kp, 90):.0f}), suffix median {np.median(k):.0f} (p90 {np.percentile(k, 90):.0f}); suffix "
          f"<= 13: {(k <= 13).mean():.0%}, <= 29: {(k <= 29).mean():.0%}, <= 62: {(k <= 62).mean():.0%}", flush=True)
    data = {"validation": rows_of(a.task, "validation")}
    for m in [x for x in a.models.split(",") if x]:
        tag = os.path.basename(os.path.normpath(os.path.expanduser(m)))
        H13.accuracy(os.path.expanduser(m), tok, data, out, f"{OUT}/{a.task}_suffix_accuracy_{tag}.json", max_len=T,
                     focus=("class0", 0))


if __name__ == "__main__":
    main()
