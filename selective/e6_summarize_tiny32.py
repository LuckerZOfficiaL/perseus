#!/usr/bin/env python
"""E6: score encrypted runs on the 32 Tiny-ImageNet validation images (labels: ImageNet-1k indices, labels.json).

Per run: encrypted top-1 = its exact plaintext version (the oracle), = the same weights' full model, label accuracy of the
encrypted run / the oracle / the full model, KL to the oracle, wall time.

  python selective/e6_summarize_tiny32.py [--labels DIR/labels.json] name=path.json [name=path.json ...]
"""
import json
import os
import sys

import numpy as np

LABELS = os.path.expanduser("~/data/selective-fhe/e3/images_tiny32/labels.json")


def main():
    args = sys.argv[1:]
    labels = LABELS
    if args and args[0].startswith("--labels"):
        flag = args.pop(0)
        labels = flag.split("=", 1)[1] if "=" in flag else args.pop(0)
    lab = json.load(open(os.path.expanduser(labels)))
    print("| Run | Images | Enc = plaintext version | Enc = full model | Label acc: enc / plaintext version / full | "
          "Median KL | Median s/image |")
    print("|---|---|---|---|---|---|---|")
    for arg in args:
        name, path = arg.split("=", 1)
        res = json.load(open(os.path.expanduser(path)))["results"]
        y = [lab[r["file"]] for r in res]
        n = len(res)
        enc_or = sum(r["argmax"] == r["oracle"] for r in res)
        enc_full = sum(r["argmax"] == r["full"] for r in res)
        acc = [sum(r[k] == t for r, t in zip(res, y)) for k in ("argmax", "oracle", "full")]
        print(f"| {name} | {n} | {enc_or}/{n} | {enc_full}/{n} | {acc[0]} / {acc[1]} / {acc[2]} | "
              f"{np.median([r['kl_oracle'] for r in res]):.2g} | {np.median([r['wall'] for r in res]):.1f} |")


if __name__ == "__main__":
    main()
