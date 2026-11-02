#!/usr/bin/env python
"""E21 (breadth), image models on the face tasks, one result file per (model, task) with the same fields for every
model: full, two_stream (E1's construction, "drop"), redaction (face tokens dropped) and B6's anonymisers of the face
patches (black, gray, blur, pixelate, inpaint, genface), accuracy, agreement with the full model, and two-stream minus
each with a paired bootstrap 95% CI.
  faces   Imagenette validation photos with a detected face (E11 / E12: YuNet, 1-63 private patches), ImageNet top-1;
  ages    E15's IMDB-WIKI validation photos, the age group (3).

  python selective/e21_vision.py --task faces --model facebook/deit-base-patch16-224 --norm processor   (anon venv)
  python selective/e21_vision.py --task ages --model ~/data/selective-fhe/e21/vit/ages_deit-base-patch16-224
--norm: "half" ((x - 0.5) / 0.5, what our fine-tunes use) or "processor" (the checkpoint's own mean / std).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e21/vision")
METHODS = ("black", "gray", "blur", "pixelate", "inpaint", "genface")


def photos(task):
    """[(file, label, private patches, generated-face file)]."""
    if task == "ages":
        import e15_ages as E15
        val = json.load(open(f"{E15.OUT}/ages.json"))["validation"]
        det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
        return [(f"{E15.OUT}/{rel}", lab, det[rel]["patches"],
                 f"{E15.OUT}_genface/{os.path.splitext(rel)[0]}.png") for rel, lab, _age in val]
    import e11_faces_common as F
    gen = os.path.expanduser("~/data/selective-fhe/e12/b6b_genface")
    rows = [(f, lab, S, os.path.join(gen, os.path.splitext(os.path.relpath(f, F.ROOT))[0] + ".png"))
            for f, lab, S in F.private_sets(63, 1)]
    return [r for r in rows if os.path.exists(r[3])]           # the photos B6b painted a face into (E12's scored ones)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("faces", "ages"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", default="half", choices=("half", "processor"))
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoImageProcessor, ViTForImageClassification
    import e1_oneway_vit as EV
    from b6_image_anon import anonymise
    from e12_report import ci
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.path.expanduser(a.model)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if a.norm == "processor":
        p = AutoImageProcessor.from_pretrained(name)
        mean, std = tuple(p.image_mean), tuple(p.image_std)
    m_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    pre = lambda ims: (torch.from_numpy(np.stack([np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR))
                                                  for im in ims])).to(dev).permute(0, 3, 1, 2).float() / 255 - m_t) / s_t
    vit = EV.Vit(ViTForImageClassification.from_pretrained(name).eval().to(dev))
    rows = photos(a.task)
    P = {k: [] for k in ("full", "two_stream", "redaction") + METHODS}
    for f, _lab, S, gen in rows:
        im = Image.open(f).convert("RGB")
        px = pre([im])
        img = np.asarray(im.resize((224, 224), Image.BILINEAR))
        batch = [Image.open(gen) if k == "genface" else Image.fromarray(anonymise(img, S, k, a.sigma)) for k in METHODS]
        with torch.no_grad():
            lr, lo = vit.two_stream(px, torch.tensor([S], device=dev), "drop")
            P["full"].append(int(vit.full(px)[0].argmax()))
            P["two_stream"].append(int(lo.argmax()))
            P["redaction"].append(int(lr.argmax()))
            for k, row in zip(METHODS, vit.full(pre(batch))[0]):
                P[k].append(int(row.argmax()))
    y = np.array([lab for _f, lab, _S, _g in rows])
    P = {k: np.array(v) for k, v in P.items()}
    tag = a.tag or os.path.basename(os.path.normpath(name))
    res = dict(task=a.task, model=a.model, norm=a.norm, n=len(y), acc={k: float((v == y).mean()) for k, v in P.items()},
               agree={k: float((v == P["full"]).mean()) for k, v in P.items()})
    best = max(("redaction",) + METHODS, key=lambda k: res["acc"][k])
    res["best_hiding"] = best
    res["two_minus"] = {k: ci(P["two_stream"] == y, v == y) for k, v in P.items() if k != "two_stream"}
    print(f"[e21-vis] {a.task} {tag}: {len(y)} photos; " + ", ".join(f"{k} {v:.3f}" for k, v in res["acc"].items()) +
          f" | two-stream minus best hiding ({best}) {res['two_minus'][best][0]:+.3f} "
          f"[{res['two_minus'][best][1]:+.3f}, {res['two_minus'][best][2]:+.3f}]", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(res, preds={k: v.tolist() for k, v in P.items()}, labels=y.tolist()),
              open(f"{OUT}/{a.task}_{tag}.json", "w"))


if __name__ == "__main__":
    main()
