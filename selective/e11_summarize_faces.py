#!/usr/bin/env python
"""E11 faces summary: the encrypted bucket runs (e11_run_faces_gpu.py) side by side, with redaction recomputed in plaintext
on exactly the scored photos, and the cost per photo over the whole Imagenette validation set (photos without a face run in
plaintext, photos with 64+ private patches run the full encrypted model, 59.05 s).

  python selective/e11_summarize_faces.py --tag orig [--model google/vit-base-patch16-224]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FULL_S = float(os.environ.get("FULL_S", "36.57"))     # the full encrypted ViT on this upstream version (E12 baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="orig")
    ap.add_argument("--model", default="google/vit-base-patch16-224")
    ap.add_argument("--dir", default=os.path.expanduser("~/data/selective-fhe/e12"))
    ap.add_argument("--prefix", default="faces3", help="result files <prefix>_<bucket>_<tag>.json")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    import e11_faces_common as F
    D = a.dir
    vit = EV.Vit(ViTForImageClassification.from_pretrained(os.path.expanduser(a.model)).eval())
    det = json.load(open(F.FACES))
    rows, tot_time, all_res = [], 0.0, []
    for b in ("s", "m", "l"):
        f = f"{D}/{a.prefix}_{b}_{a.tag}.json"
        if not os.path.exists(f):
            continue
        res = json.load(open(f))["results"]
        red = 0
        for r in res:
            S = torch.tensor([det["detections"][r["file"]]["patches"]], dtype=torch.long)
            with torch.no_grad():
                lr, _ = vit.two_stream(EV.preprocess([Image.open(os.path.join(F.ROOT, r["file"]))]), S, "drop")
            r["redact"] = int(lr.argmax())
        walls = {r["page"]: r["page_wall"] for r in res}
        n = len(res)
        t = sum(walls.values())
        tot_time += t
        all_res += res
        acc = lambda k: sum(r[k] == r["label"] for r in res) / n
        rows.append((b, n, len(walls), t / n, sum(r["argmax"] == r["oracle"] for r in res), acc("full"), acc("argmax"),
                     acc("redact"), np.median([r["kl_oracle"] for r in res])))
    print("| Bucket | Photos | Pages | s/photo | Enc = plaintext two-stream | Acc full | Acc encrypted two-stream | Acc redaction | Median KL |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b, n, p, s, eq, af, ae, ar, kl in rows:
        print(f"| {b} | {n} | {p} | {s:.2f} | {eq}/{n} | {af:.3f} | {ae:.3f} | {ar:.3f} | {kl:.2g} |")
    n = len(all_res)
    acc = lambda k: sum(r[k] == r["label"] for r in all_res) / n
    print(f"| all | {n} | | {tot_time / n:.2f} | {sum(r['argmax'] == r['oracle'] for r in all_res)}/{n} | {acc('full'):.3f} | "
          f"{acc('argmax'):.3f} | {acc('redact'):.3f} | |")
    k = np.array([len(d["patches"]) for d in det["detections"].values()])
    n_xl = int((k >= 64).sum())
    per_face = tot_time / n
    blended_faces = (per_face * (len(k) - n_xl) + FULL_S * n_xl) / len(k)
    print(f"cost per photo with a face (incl. {n_xl} with 64+ private patches at {FULL_S} s): {blended_faces:.2f} s; over all "
          f"{det['n_photos']} photos (no face: plaintext): {blended_faces * len(k) / det['n_photos']:.2f} s; everything "
          f"encrypted: {FULL_S} s")


if __name__ == "__main__":
    main()
