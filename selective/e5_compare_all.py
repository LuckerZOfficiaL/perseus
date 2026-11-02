#!/usr/bin/env python
"""The four methods side by side at matched sensitive sets: two-stream (E1 / E1b), late-inject-replace,
late-inject-additive, late-inject-attention (E5 / E5b / E5c), before and after their fine-tunes, per placement and
size. Each cell: accuracy (agreement with the full model in brackets).

  python selective/e5_compare_all.py vit --klate 2
  python selective/e5_compare_all.py bert --klate 2
"""
import argparse
import json
import os

ROOT = os.path.expanduser("~/data/selective-fhe/")


def load(path, **sel):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return {}
    rows = json.load(open(p))["rows"]
    return [r for r in rows if all(r.get(k) == v for k, v in sel.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["vit", "bert"])
    ap.add_argument("--klate", type=int, default=2)
    ap.add_argument("--sizes", default="")
    a = ap.parse_args()
    v = a.kind == "vit"
    sk = "k" if v else "frac"
    ph1 = "drop" if v else "mask"
    e1, e1b = f"e1/{'vit_tinyimagenet' if v else 'bert_sst2'}.json", f"e1b/{'vit_tinyimagenet' if v else 'bert_sst2'}_ft.json"
    base = 'vit_tinyimagenet' if v else 'bert_sst2'
    cols = {
        "two-stream": (load(e1, placeholder=ph1), "acc_oneway", "agree_oneway", {}),
        "two-stream ft": (load(e1b, placeholder=ph1), "acc_oneway", "agree_oneway", {}),
        "redaction": (load(f"e5/{base}.json", placeholder=ph1, k_late=0), "acc", "agree", {}),
        "replace": (load(f"e5/{base}.json", placeholder=ph1, k_late=a.klate), "acc", "agree", {}),
        "replace ft": (load(f"e5b/{base}_ft.json", placeholder=ph1, k_late=a.klate), "acc", "agree", {}),
        "additive": (load(f"e5/{base}_add.json", placeholder=ph1, k_late=a.klate), "acc", "agree", {}),
        "additive ft": (load(f"e5c/{base}_add_ft.json", placeholder=ph1, k_late=a.klate), "acc", "agree", {}),
        "attention ft": (load(f"e5c/{base}_attention_ft.json", placeholder=ph1, k_late=a.klate), "acc", "agree", {}),
    }
    index = {name: {(r["placement"], r[sk]): r for r in rows} for name, (rows, _, _, _) in cols.items()}
    full = next((r["acc_full"] for rows, *_ in cols.values() for r in rows), float("nan"))
    print(f"{a.kind}: late-inject at k_late = {a.klate} of 12 layers; full model {full:.3f}; cells = accuracy (agreement with the full model)\n")
    names = [n for n in cols if index[n]]
    print("| Placement | size | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    placements = ["salient", "box" if v else "span", "random", "nonsalient"]
    sizes = [int(x) if v else float(x) for x in a.sizes.split(",")] if a.sizes else ([16, 49, 100] if v else [0.1, 0.3, 0.5])
    for pl in placements:
        for sz in sizes:
            cells = []
            for n in names:
                r = index[n].get((pl, sz))
                _, ka, kg, _ = cols[n]
                cells.append("n/a" if r is None else f"{r[ka]:.3f} ({r[kg]:.2f})")
            print(f"| {pl} | {sz} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
