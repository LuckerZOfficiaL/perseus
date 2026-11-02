#!/usr/bin/env python
"""E16, who is in the photo: a zero-shot re-identification attack on each method's server view of E15's IMDB-WIKI age
photos (celebrities; the dataset names each one). The server scores the view against "a photo of <name>." for every
name among the validation photos with CLIP (openai/clip-vit-large-patch14-336, which knows many celebrities), and
SigLIP (google/siglip-so400m-patch14-384) as a second attacker. Views as e16_leakage.py (full; ours = the face patches
removed; blur / pixelate; the generated face of b6b_gen_faces.py; ours_d<r> = the face dilated by r patches). Reports
top-1 / top-5 identification with bootstrap 95% CIs; chance is 1 / number of names.

  python selective/e16_identity.py          (GPU, anon venv)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e16")


def feats(out):
    """get_*_features: a tensor, or (transformers 5) an output whose pooler_output is the projected embedding."""
    return (out if torch.is_tensor(out) else out.pooler_output).float()


def names_of(split, idx):
    """IMDB-WIKI names of the stream rows idx (the E15 photos are <split>/<stream index>.jpg)."""
    from datasets import load_dataset
    tok = open(os.path.expanduser("~/.credentials/hf_token")).read().strip()
    ds = load_dataset("systemk-ai/imdb-wiki", "default", split=split, streaming=True, token=tok).select_columns(["name"])
    want, out = set(idx), {}
    for j, r in enumerate(ds):
        if j in want:
            out[j] = r["name"]
            if len(out) == len(want):
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", default="full,ours,blur,pixelate,genface,ours_d1,ours_d2")
    ap.add_argument("--models", default="openai/clip-vit-large-patch14-336,google/siglip-so400m-patch14-384")
    ap.add_argument("--sigma", type=float, default=8.0)
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import e15_ages as E15
    from b6_image_anon import anonymise
    from e16_leakage import acc_ci, dilate
    val = json.load(open(f"{E15.OUT}/ages.json"))["validation"]
    det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    f = f"{OUT}/ages_names.json"
    if not os.path.exists(f):
        idx = [int(os.path.splitext(os.path.basename(r[0]))[0]) for r in val]
        json.dump({str(k): v for k, v in names_of("test", idx).items()}, open(f, "w"))
    names = json.load(open(f))
    who = [names[os.path.splitext(os.path.basename(r[0]))[0]] for r in val]
    cand = sorted(set(who))
    gold = np.array([cand.index(w) for w in who])
    print(f"[e16-id] {len(val)} photos, {len(cand)} distinct names (chance top-1 {1 / len(cand):.4f})", flush=True)
    gen_dir = os.path.expanduser(f"{E15.OUT}_genface")
    views = {}
    for v in a.views.split(","):
        ims = []
        for rel, _lab, _age in val:
            im = Image.open(f"{E15.OUT}/{rel}").convert("RGB")
            S = det[rel]["patches"]
            if v == "full":                                      # the model input's resolution, as every view
                ims.append(im.resize((224, 224), Image.BILINEAR).resize(im.size, Image.BILINEAR))
                continue
            if v == "genface":
                ims.append(Image.open(os.path.join(gen_dir, os.path.splitext(rel)[0] + ".png")).convert("RGB")
                           .resize(im.size, Image.BILINEAR))
                continue
            x = np.asarray(im.resize((224, 224), Image.BILINEAR))
            if v.startswith("ours"):
                x = anonymise(x, dilate(S, int(v[6:])) if v.startswith("ours_d") else S, "black", a.sigma)
            else:
                x = anonymise(x, S, v, a.sigma)
            ims.append(Image.fromarray(x).resize(im.size, Image.BILINEAR))
        views[v] = ims
    res = dict(n=len(val), names=len(cand), chance=1 / len(cand), models={})
    for mname in a.models.split(","):
        model = AutoModel.from_pretrained(mname, torch_dtype=torch.float16).cuda().eval()
        proc = AutoProcessor.from_pretrained(mname)
        with torch.no_grad():
            T = []
            for j in range(0, len(cand), 256):
                t = proc(text=[f"a photo of {n}." for n in cand[j:j + 256]], return_tensors="pt", padding="max_length",
                         truncation=True).to("cuda")
                T.append(torch.nn.functional.normalize(feats(model.get_text_features(**t)), dim=-1))
            T = torch.cat(T)
            res["models"][mname] = {}
            for v, ims in views.items():
                I = []
                for j in range(0, len(ims), 64):
                    px = proc(images=ims[j:j + 64], return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
                    I.append(torch.nn.functional.normalize(feats(model.get_image_features(pixel_values=px)), dim=-1))
                rank = (torch.cat(I) @ T.T).argsort(-1, descending=True).cpu().numpy()
                top1 = (rank[:, 0] == gold).astype(int)
                top5 = (rank[:, :5] == gold[:, None]).any(1).astype(int)
                m1, m5 = acc_ci(top1), acc_ci(top5)
                res["models"][mname][v] = dict(top1=m1, top5=m5)
                print(f"[e16-id] {mname.split('/')[-1]:28s} {v:8s} top-1 {m1[0]:.3f} [{m1[1]:.3f}, {m1[2]:.3f}]  "
                      f"top-5 {m5[0]:.3f} [{m5[1]:.3f}, {m5[2]:.3f}]", flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/identity_ages.json", "w"), indent=1)
    os._exit(0)                                                  # the streaming dataset's threads crash at exit


if __name__ == "__main__":
    main()
