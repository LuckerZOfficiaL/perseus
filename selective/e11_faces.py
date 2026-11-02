#!/usr/bin/env python
"""E11 (realistic images): faces as the private part. The client's face detector (OpenCV YuNet) runs on each Imagenette
validation photo (320 px); every 16x16 patch of the 224 x 224 model input that overlaps a detected face box (grown by
--grow on each side) is private. Writes the private patch lists of the photos with at least one face, then the plaintext
accuracy (1000-way ImageNet top-1) of the full ViT-B/16, two-stream (E1's construction, "drop" placeholder) and redaction
on them, for each model given. CPU only.

  python selective/e11_faces.py --models google/vit-base-patch16-224,~/data/selective-fhe/e1b/vit_oneway
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.expanduser("~/data/selective-fhe/data/imagenette_dl/imagenette2-320/val")
YUNET = os.path.expanduser("~/data/selective-fhe/models/face_detection_yunet_2023mar.onnx")


def detect(files, score, grow):
    import cv2
    det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), score)
    out = {}
    for f in files:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        det.setInputSize((w, h))
        _n, faces = det.detect(im)
        if faces is None or len(faces) == 0:
            continue
        pat = set()
        for x, y, bw, bh in faces[:, :4]:
            x0, x1 = (x - grow * bw) * 224 / w, (x + bw * (1 + grow)) * 224 / w
            y0, y1 = (y - grow * bh) * 224 / h, (y + bh * (1 + grow)) * 224 / h
            for r in range(14):
                for c in range(14):
                    if c * 16 < x1 and (c + 1) * 16 > x0 and r * 16 < y1 and (r + 1) * 16 > y0:
                        pat.add(r * 14 + c)
        out[f] = dict(patches=sorted(pat), faces=int(len(faces)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/vit-base-patch16-224")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--grow", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e11/faces.json"))
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    wnid2idx = {v[0]: int(k) for k, v in json.load(open(os.path.expanduser(
        "~/data/selective-fhe/data/imagenet_class_index.json"))).items()}
    files = sorted(glob.glob(f"{ROOT}/*/*.JPEG"))
    found = detect(files, a.score, a.grow)
    k = np.array([len(v["patches"]) for v in found.values()])
    print(f"[e11-faces] {len(found)} of {len(files)} photos have a face; private patches per photo: median {np.median(k):.0f}, "
          f"mean {k.mean():.1f}, max {k.max()} (of 196); >= 64: {(k >= 64).sum()}", flush=True)
    res = dict(n_photos=len(files), detections={os.path.relpath(f, ROOT): v for f, v in found.items()}, models={})
    for mname in a.models.split(","):
        model = ViTForImageClassification.from_pretrained(os.path.expanduser(mname)).eval()
        vit = EV.Vit(model)
        hits = {"full": 0, "two_stream": 0, "redaction": 0}
        agree = {"two_stream": 0, "redaction": 0}
        for f, v in found.items():
            y = wnid2idx[f.split("/")[-2]]
            px = EV.preprocess([Image.open(f)])
            S = torch.tensor([v["patches"]], dtype=torch.long)
            with torch.no_grad():
                lf, _ = vit.full(px)
                lr, lo = vit.two_stream(px, S, "drop")
            pf, pr, po = int(lf.argmax()), int(lr.argmax()), int(lo.argmax())
            hits["full"] += pf == y
            hits["two_stream"] += po == y
            hits["redaction"] += pr == y
            agree["two_stream"] += po == pf
            agree["redaction"] += pr == pf
        n = len(found)
        r = {k_: v_ / n for k_, v_ in hits.items()} | {f"agree_{k_}": v_ / n for k_, v_ in agree.items()}
        res["models"][mname] = r
        print(f"[e11-faces] {os.path.basename(mname)}: accuracy full {r['full']:.4f} two-stream {r['two_stream']:.4f} "
              f"redaction {r['redaction']:.4f} | agreement with full: two-stream {r['agree_two_stream']:.4f} "
              f"redaction {r['agree_redaction']:.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
