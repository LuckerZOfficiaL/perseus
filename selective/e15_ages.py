#!/usr/bin/env python
"""E15 (a task whose answer depends on the private part, photos): age group of the person in an IMDB-WIKI photo (Rothe et
al. 2015/2018; HF systemk-ai/imdb-wiki, the full photos), three groups (under 30, 30-44, 45 and over). The private part
is the face, found by the client's detector as in E11 (OpenCV YuNet on the 320 px photo; every 16 x 16 patch of the
224 x 224 model input that overlaps the face box grown by 10% is private). Photos with exactly one detected face and at
most 63 private patches; class-balanced.

  --build                 stream the photos, detect faces, write ~/data/selective-fhe/e15/ages/{ages.json,faces.json}
  --finetune OUT          ordinary fine-tune of google/vit-base-patch16-224 with a 3-way head (GPU)
  --eval MODEL[,MODEL]    accuracy and agreement with the full model on the validation photos: full, two-stream (E1's
                          construction, "drop" placeholder), redaction (token drop) and B6's image anonymisers of the
                          face (black, gray, blur, pixelate, inpaint; a generated face when --gen-dir exists, made by
                          b6b_gen_faces.py --ages), each with the two-stream minus it paired bootstrap 95% CI

Needs datasets + cv2 (the sat venv; --eval: the anon venv).
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e15/ages")
GROUPS = ["under 30", "30-44", "45 and over"]


def group(age):
    return 0 if age < 30 else 1 if age < 45 else 2


def build(a):
    import io
    from datasets import Image as ImageFeature, load_dataset
    from PIL import Image
    import e11_faces as E11
    tokf = os.path.expanduser("~/.credentials/hf_token")
    data, dets = {}, {}
    for split, name, quota in (("test", "validation", a.val), ("train", "train", a.train)):
        ds = load_dataset("systemk-ai/imdb-wiki", "default", split=split, streaming=True,
                          token=open(tokf).read().strip()).select_columns(["age", "second_face_score", "image"])
        ds = ds.cast_column("image", ImageFeature(decode=False))         # decoded below: a few files are truncated
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        counts, items, seen = Counter(), [], 0
        for j, r in enumerate(ds):
            if all(counts[k] >= quota for k in range(len(GROUPS))):
                break
            seen += 1
            age, second = r["age"], r["second_face_score"]
            if age is None or not 15 <= age <= 80 or (second is not None and not math.isnan(second)):
                continue
            lab = group(age)
            if counts[lab] >= quota:
                continue
            try:
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            except OSError:
                continue
            im.thumbnail((320, 320))
            rel = f"{name}/{j}.jpg"
            f = f"{OUT}/{rel}"
            im.save(f, quality=95)
            d = E11.detect([f], a.score, a.grow).get(f)
            if d is None or d["faces"] != 1 or len(d["patches"]) > 63:
                os.remove(f)
                continue
            items.append([rel, lab, int(age)])
            dets[rel] = d
            counts[lab] += 1
            if len(items) % 500 == 0:
                print(f"[e15-ages] {name}: {len(items)} kept of {seen} streamed {dict(counts)}", flush=True)
        data[name] = items
        k = np.array([len(dets[r[0]]["patches"]) for r in items])
        print(f"[e15-ages] {name}: {len(items)} photos of {seen} streamed, per group {dict(counts)}; private patches "
              f"median {np.median(k):.0f}, <= 20: {(k <= 20).mean():.0%}", flush=True)
    json.dump(dict(data, groups=GROUPS), open(f"{OUT}/ages.json", "w"))
    json.dump(dict(detections=dets, score=a.score, grow=a.grow), open(f"{OUT}/faces.json", "w"))


def finetune(a):
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = json.load(open(f"{OUT}/ages.json"))["train"]
    model = ViTForImageClassification.from_pretrained(a.base, num_labels=len(GROUPS),
                                                      ignore_mismatched_sizes=True).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(train) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        idx = rng.choice(len(train), a.batch, replace=False)
        ims = [Image.open(f"{OUT}/{train[i][0]}") for i in idx]
        ims = [im.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else im for im in ims]
        y = torch.tensor([train[i][1] for i in idx]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(pixel_values=EV.preprocess(ims).cuda()).logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"[e15-ages] step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval().save_pretrained(os.path.expanduser(a.finetune))


def evaluate(a):
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    from b6_image_anon import anonymise
    from e12_report import ci
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = json.load(open(f"{OUT}/ages.json"))["validation"]
    det = json.load(open(f"{OUT}/faces.json"))["detections"]
    y = np.array([r[1] for r in val])
    gen_dir = os.path.expanduser(a.gen_dir)
    methods = ("black", "gray", "blur", "pixelate", "inpaint") + (("genface",) if os.path.isdir(gen_dir) else ())
    for mname in a.eval.split(","):
        tag = os.path.basename(os.path.normpath(mname))
        vit = EV.Vit(ViTForImageClassification.from_pretrained(os.path.expanduser(mname)).eval().to(dev))
        P = {k: [] for k in ("full", "two_stream", "redaction") + methods}
        for rel, _lab, _age in val:
            im = Image.open(f"{OUT}/{rel}").convert("RGB")
            px = EV.preprocess([im]).to(dev)
            S = torch.tensor([det[rel]["patches"]], dtype=torch.long, device=dev)
            img = np.asarray(im.resize((224, 224), Image.BILINEAR))
            batch = [Image.open(os.path.join(gen_dir, os.path.splitext(rel)[0] + ".png")).convert("RGB")
                     if k == "genface" else Image.fromarray(anonymise(img, det[rel]["patches"], k, a.sigma))
                     for k in methods]
            with torch.no_grad():
                lr, lo = vit.two_stream(px, S, "drop")
                P["full"].append(int(vit.full(px)[0].argmax()))
                P["two_stream"].append(int(lo.argmax()))
                P["redaction"].append(int(lr.argmax()))
                for k, row in zip(methods, vit.full(EV.preprocess(batch).to(dev))[0]):
                    P[k].append(int(row.argmax()))
        P = {k: np.array(v) for k, v in P.items()}
        print(f"[e15-ages] {tag}: {len(y)} photos; accuracy " +
              ", ".join(f"{k} {(v == y).mean():.3f}" for k, v in P.items()) + " | agreement with full: " +
              ", ".join(f"{k} {(v == P['full']).mean():.3f}" for k, v in P.items() if k != "full"), flush=True)
        out = {}
        for k, v in P.items():
            out[k] = dict(acc=float((v == y).mean()), agree_full=float((v == P["full"]).mean()))
            if k not in ("full", "two_stream"):
                m, lo_, hi = ci(P["two_stream"] == y, v == y)
                out[k]["two_minus"] = [m, lo_, hi]
                print(f"[e15-ages] {tag} {k:9s} two-stream minus it {m:+.3f} [{lo_:+.3f}, {hi:+.3f}]", flush=True)
        json.dump(dict(out, n=len(y), preds={k: v.tolist() for k, v in P.items()}),
                  open(f"{os.path.dirname(OUT)}/ages_eval_{tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--finetune", default=None)
    ap.add_argument("--base", default="google/vit-base-patch16-224", help="--finetune: the pretrained ViT (E21: other sizes)")
    ap.add_argument("--eval", default=None)
    ap.add_argument("--val", type=int, default=300, help="validation photos per age group")
    ap.add_argument("--train", type=int, default=3000, help="train photos per age group")
    ap.add_argument("--score", type=float, default=0.7)
    ap.add_argument("--grow", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=8.0, help="--eval: blur sigma (px), as B6")
    ap.add_argument("--gen-dir", default=f"{OUT}_genface", help="--eval: generated faces (b6b_gen_faces.py --ages)")
    a = ap.parse_args()
    if a.build:
        build(a)
    if a.finetune:
        finetune(a)
    if a.eval:
        evaluate(a)


if __name__ == "__main__":
    main()
