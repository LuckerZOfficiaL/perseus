#!/usr/bin/env python
"""Late injection (E5 json) against two-stream (E1 json) at the same sensitive sets: accuracy / agreement with the
full model per placement and size, for each number of encrypted layers.

  python selective/e5_compare.py vit e5/vit_tinyimagenet.json e1/vit_tinyimagenet.json [--ft e5b/vit_tinyimagenet_ft.json]
"""
import argparse
import json
import os

ROOT = os.path.expanduser("~/data/selective-fhe/")


def rows_of(path):
    return json.load(open(os.path.join(ROOT, path)))["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vit", "bert"])
    ap.add_argument("e5")
    ap.add_argument("e1")
    ap.add_argument("--ft", default=None, help="E5b json: the same sweep on the fine-tuned model")
    ap.add_argument("--metric", default="acc", choices=["acc", "agree"])
    a = ap.parse_args()
    size_key = "k" if a.kind == "vit" else "frac"
    ph5 = "drop" if a.kind == "vit" else "mask"
    ph1 = "drop" if a.kind == "vit" else "mask"
    e5 = {(r["placement"], r[size_key], r["k_late"]): r for r in rows_of(a.e5) if r["placeholder"] == ph5}
    e1 = {(r["placement"], r[size_key]): r for r in rows_of(a.e1) if r["placeholder"] == ph1}
    ft = {(r["placement"], r[size_key], r["k_late"]): r for r in rows_of(a.ft) if r["placeholder"] == ph5} if a.ft else {}
    klates = sorted({k for (_, _, k) in e5})
    full = next(iter(e5.values()))["acc_full"]
    m = a.metric
    m1 = "acc_oneway" if m == "acc" else "agree_oneway"
    print(f"{a.kind}: metric {m}; full model accuracy {full:.3f}. Late injection with k encrypted layers of 12; two-stream = E1 (12 layers, few tokens).\n")
    head = "| Placement | size | " + " | ".join(f"late k={k}" for k in klates) + " | two-stream |" + (" | ".join([""] + [f"late k={k} ft" for k in klates if k in {1, 2, 3, 4}]) + " |" if ft else "")
    print(head)
    print("|---|---|" + "---|" * (len(klates) + 1) + ("---|" * len([k for k in klates if k in {1, 2, 3, 4}]) if ft else ""))
    placements = ["salient", "box" if a.kind == "vit" else "span", "random", "nonsalient"]
    sizes = sorted({s for (_, s, _) in e5})
    for pl in placements:
        for sz in sizes:
            cells = []
            for k in klates:
                r = e5.get((pl, sz, k))
                cells.append("n/a" if r is None else f"{r[m]:.3f}")
            r1 = e1.get((pl, sz))
            cells.append("n/a" if r1 is None else f"{r1[m1]:.3f}")
            if ft:
                for k in klates:
                    if k in {1, 2, 3, 4}:
                        r = ft.get((pl, sz, k))
                        cells.append("n/a" if r is None else f"{r[m]:.3f}")
            print(f"| {pl} | {sz} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
