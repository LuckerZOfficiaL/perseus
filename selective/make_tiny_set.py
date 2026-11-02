#!/usr/bin/env python
"""A Tiny-ImageNet validation image set for the encrypted runs: N images drawn at random (seed) among those whose class
maps to ImageNet-1k (by WNID, as e1_oneway_vit.py), excluding the files of other sets; saved as tiny_<index>.png
(64 px, the runners resize to 224) with labels.json {file: ImageNet-1k index}.

  python selective/make_tiny_set.py --n 300 --seed 1 --out ~/data/selective-fhe/images_tiny300 \
      --exclude ~/data/selective-fhe/e3/images_tiny32
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--class-index", default=os.path.expanduser("~/data/selective-fhe/data/imagenet_class_index.json"))
    a = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet", split="valid")
    wnid2idx = {v[0]: int(k) for k, v in json.load(open(a.class_index)).items()}
    tiny2in = np.array([wnid2idx.get(w, -1) for w in ds.features["label"].names])
    labels = np.array(ds["label"])
    skip = set()
    for d in a.exclude:
        skip |= set(json.load(open(os.path.join(os.path.expanduser(d), "labels.json"))))
    ok = [i for i in np.where(tiny2in[labels] >= 0)[0] if f"tiny_{i:05d}.png" not in skip]
    sel = np.sort(np.random.default_rng(a.seed).choice(ok, size=min(a.n, len(ok)), replace=False))
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    lab = {}
    for i in sel.tolist():
        f = f"tiny_{i:05d}.png"
        ds[i]["image"].convert("RGB").save(os.path.join(out, f))
        lab[f] = int(tiny2in[labels[i]])
    json.dump(lab, open(os.path.join(out, "labels.json"), "w"), indent=1)
    print(f"{len(lab)} images ({len(set(lab.values()))} classes) -> {out}; excluded {len(skip)}")


if __name__ == "__main__":
    main()
