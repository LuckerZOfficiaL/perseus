#!/usr/bin/env python
"""Baseline B4 (cost model, no device run): suffix-only selective encryption (Park et al. 2026, arXiv 2601.18511: a public
prefix in the clear, an encrypted suffix) on our tasks. With private tokens anywhere, a suffix-only scheme must encrypt
everything from the first private token to the end. Per text: the encrypted tokens under our arbitrary spans (CLS +
private + SEP) and under suffix-only (CLS + first private .. end); the share fitting the plan's per-input cap (the rest
go fully encrypted); and the expected device time per text from the measured page costs, packing inputs by lanes
(128 lanes per page, at most G inputs) and charging the full model's measured time to inputs over the cap.

  python selective/b4_suffix_only.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
# task: texts (None = SST-2 validation), detector file, tokens, per-input cap, inputs per page, page time (s), full (s)
TASKS = {
    "SST-2 names": (None, f"{D}/e11/ner_private.json", 128, 15, 24, 19.43, 14.40),
    "HateXplain identity mentions": (f"{D}/e13/hatexplain.json", f"{D}/e13/hatexplain_private.json", 128, 15, 24, 19.62, 14.40),
    "Enron personal data": (f"{D}/e14/enron_t256.json", f"{D}/e14/enron_t256_private.json", 256, 29, 3, 23.47, 39.76),
}


def cost(lanes, cap, G, page_s, full_s):
    """Mean seconds per text: inputs within the cap share pages by lanes (<= G per page), the rest run the full model."""
    lanes = np.asarray(lanes)
    ok = lanes <= cap
    per_page = np.minimum(G, 128 / max(lanes[ok].mean(), 1)) if ok.any() else 1
    return (ok.sum() * page_s / per_page + (~ok).sum() * full_s) / len(lanes), float(ok.mean())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    out = {}
    for name, (data, privf, T, cap, G, page_s, full_s) in TASKS.items():
        if data:
            rows = json.load(open(data))["validation"]
        else:
            from datasets import load_dataset
            ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
            rows = [[s, int(l)] for s, l in zip(ds["sentence"], ds["label"])]
        priv = json.load(open(privf))["validation"]
        ours, suffix, ntok = [], [], []
        for i, (text, _lab) in enumerate(rows):
            n = len(tok(text, truncation=True, max_length=T)["input_ids"])
            S = [p for p in priv.get(str(i), []) if 0 < p < n - 1]
            if not S:
                continue
            ours.append(len(S) + 2)
            suffix.append(n - min(S) + 1)               # CLS + tokens from the first private one through SEP
            ntok.append(n)
        c_ours, f_ours = cost(ours, cap, G, page_s, full_s)
        c_suf, f_suf = cost(suffix, cap, G, page_s, full_s)
        out[name] = dict(texts=len(ours), tokens=float(np.mean(ntok)), enc_ours=float(np.mean(ours)),
                         enc_suffix=float(np.mean(suffix)), fit_ours=f_ours, fit_suffix=f_suf, s_ours=c_ours,
                         s_suffix=c_suf, s_full=full_s)
        print(f"[b4] {name}: {len(ours)} texts, {np.mean(ntok):.0f} tokens; encrypted tokens ours {np.mean(ours):.1f} vs "
              f"suffix-only {np.mean(suffix):.1f}; within the cap {f_ours:.0%} vs {f_suf:.0%}; est. s/text ours "
              f"{c_ours:.2f} vs suffix-only {c_suf:.2f} (full {full_s})", flush=True)
    json.dump(out, open(f"{D}/e12/b4_suffix_only.json", "w"), indent=1)


if __name__ == "__main__":
    main()
