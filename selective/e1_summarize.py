#!/usr/bin/env python
"""Markdown tables from the E1 result JSONs: agreement with the full model, redaction vs one-way."""
import json
import sys


def table(path, xkey, xfmt):
    d = json.load(open(path))
    rows = d["rows"]
    print(f"\n### {d['model']}  (n = {rows[0]['n']}, full-model accuracy {rows[0]['acc_full']:.3f})\n")
    print(f"| Placement | Placeholder | {xkey} | Enc. frac | Acc redact | Acc one-way | Agree redact | Agree one-way | KL redact | KL one-way |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['placement']} | {r['placeholder']} | {xfmt(r)} | {r['frac_enc']:.2f} | {r['acc_redact']:.3f} | "
              f"{r['acc_oneway']:.3f} | {r['agree_redact']:.3f} | {r['agree_oneway']:.3f} | {r['kl_redact']:.3f} | {r['kl_oneway']:.3f} |")


for p in sys.argv[1:]:
    if "bert" in p:
        table(p, "k (mean)", lambda r: f"{r['frac']:.1f} ({r['mean_k']:.1f})")
    else:
        table(p, "k", lambda r: str(r["k"]))
