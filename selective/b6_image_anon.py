#!/usr/bin/env python
"""Baseline B6 (accuracy only, plaintext): anonymise the private part of each face photo instead of encrypting it, then
run the ordinary model. Same photos and private patches as the encrypted face runs (faces5_{s,m,l}_<tag>.json); the
private patches' pixels of the 224 x 224 model input are replaced by
  black    the darkest value,
  gray     the mean value (0 after normalisation),
  blur     a Gaussian blur of the whole photo (sigma --sigma px),
  pixelate the patch mean (16 x 16 mosaic, one colour per private patch),
  inpaint  OpenCV's Telea inpainting from the surrounding pixels,
  genface  a synthetic face painted in by a diffusion inpainting model (b6b_gen_faces.py; when --gen-dir exists),
and compared with two-stream (encrypted, from the run files), token-drop redaction (ours) and the full model.
Accuracy, agreement with the full model, and the two-stream minus baseline accuracy with a paired bootstrap 95% CI.

  python selective/b6_image_anon.py --tag orig [--sigma 8]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = os.path.expanduser("~/data/selective-fhe/e12")
PATCH, GRID = 16, 14


def anonymise(img, patches, how, sigma):
    """img: (224, 224, 3) uint8; patches: private patch indices (row-major 14 x 14)."""
    import cv2
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for p in patches:
        r, c = divmod(int(p), GRID)
        mask[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 1
    out = img.copy()
    m = mask.astype(bool)
    if how == "black":
        out[m] = 0
    elif how == "gray":
        out[m] = 128
    elif how == "blur":
        out[m] = cv2.GaussianBlur(img, (0, 0), sigma)[m]
    elif how == "pixelate":
        for p in patches:
            r, c = divmod(int(p), GRID)
            blk = out[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
            blk[:] = blk.reshape(-1, 3).mean(0).astype(np.uint8)
    elif how == "inpaint":
        out = cv2.inpaint(img, mask * 255, 5, cv2.INPAINT_TELEA)
    else:
        raise ValueError(how)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="orig", choices=("orig", "ft"))
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--gen-dir", default=f"{E}/b6b_genface")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    import e11_faces_common as F
    from e12_report import ci
    det = json.load(open(F.FACES))["detections"]
    weights = "google/vit-base-patch16-224" if a.tag == "orig" else os.path.expanduser("~/data/selective-fhe/e1b/vit_oneway")
    model = ViTForImageClassification.from_pretrained(weights).eval()
    res = []
    for b in ("s", "m", "l"):
        d = json.load(open(f"{E}/faces5_{b}_{a.tag}.json"))
        res += d["results"]
    methods = ("black", "gray", "blur", "pixelate", "inpaint") + (("genface",) if os.path.isdir(a.gen_dir) else ())
    preds = {k: [] for k in methods}
    for r in res:
        img = np.asarray(Image.open(os.path.join(F.ROOT, r["file"])).convert("RGB").resize((224, 224), Image.BILINEAR))
        S = det[r["file"]]["patches"]
        batch = [Image.open(os.path.join(a.gen_dir, os.path.splitext(r["file"])[0] + ".png")).convert("RGB")
                 if k == "genface" else Image.fromarray(anonymise(img, S, k, a.sigma)) for k in methods]
        with torch.no_grad():
            lg = model(pixel_values=EV.preprocess(batch)).logits
        for k, row in zip(methods, lg):
            preds[k].append(int(row.argmax()))
    y = np.array([r["label"] for r in res])
    full, two = np.array([r["full"] for r in res]), np.array([r["argmax"] for r in res])
    out = {"n": len(res), "full": float((full == y).mean()), "two_stream": float((two == y).mean())}
    print(f"[b6] {a.tag}: {len(res)} face photos; full {out['full']:.3f}, two-stream (encrypted) {out['two_stream']:.3f}")
    for k in methods:
        p = np.array(preds[k])
        m, lo, hi = ci(two == y, p == y)
        out[k] = dict(acc=float((p == y).mean()), agree_full=float((p == full).mean()), two_minus=[m, lo, hi])
        print(f"[b6] {k:8s} acc {out[k]['acc']:.3f} agree-with-full {out[k]['agree_full']:.3f} | two-stream minus it "
              f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    json.dump(dict(out, preds=preds), open(a.out or f"{E}/b6_image_anon_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
