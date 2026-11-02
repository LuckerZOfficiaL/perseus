#!/usr/bin/env python
"""E1b comparison: original vs one-way fine-tune vs plain fine-tune (control), per placement and sensitive size.
Reports each model's own full accuracy, the one-way gap to it, and agreement with its own full predictions.

  python selective/e1b_compare.py vit orig=e1/vit_tinyimagenet.json oneway=e1b/vit_tinyimagenet_ft.json plain=e1b/vit_tinyimagenet_plainft.json
"""
import json
import os
import sys

ROOT = os.path.expanduser("~/data/selective-fhe/")


def main():
    kind, *specs = sys.argv[1:]
    models = {}
    for s in specs:
        name, path = s.split("=", 1)
        rows = json.load(open(os.path.join(ROOT, path)))["rows"]
        key = "k" if kind == "vit" else "frac"
        models[name] = {(r["placement"], r["placeholder"], r[key]): r for r in rows}
    ph = "drop" if kind == "vit" else "mask"
    sizes = [16, 49, 100, 144] if kind == "vit" else [0.1, 0.3, 0.5, 0.7]
    names = list(models)
    full = {n: next(iter(models[n].values()))["acc_full"] for n in names}
    print(f"{kind}: own full accuracy " + ", ".join(f"{n} {full[n]:.3f}" for n in names) + f"  (placeholder {ph})\n")
    print("| Placement | size | " + " | ".join(f"{n}: one-way / gap / agree" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for pl in (["salient", "box", "random", "nonsalient"] if kind == "vit" else ["salient", "span", "random", "nonsalient"]):
        for sz in sizes:
            cells = []
            for n in names:
                r = models[n].get((pl, ph, sz))
                cells.append("n/a" if r is None else f"{r['acc_oneway']:.3f} / {r['acc_full'] - r['acc_oneway']:+.3f} / {r['agree_oneway']:.3f}")
            print(f"| {pl} | {sz} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
