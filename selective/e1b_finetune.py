#!/usr/bin/env python
"""E1b: fine-tune ViT-B/16 or BERT-base under the one-way attention rule.

Each step samples a placement and a sensitive set size, runs the frozen original model as a teacher
(full logits, and the saliency the "salient" placement needs) and the student's one-way read-out,
and minimizes cross-entropy on the label plus KL(teacher || student). Size 0 is in the mix so the
plain model is preserved. The plaintext stream uses the placeholder that won in E1 (drop for ViT,
[MASK] for BERT). Saves HF weights to --out; evaluate with e1_oneway_{vit,bert}.py --model <out>.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["vit", "bert"], required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=0, help="0: 48 for vit, 32 for bert")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--kl", type=float, default=1.0, help="weight of the distillation term")
    ap.add_argument("--train-n", type=int, default=16000, help="vit: Tiny-ImageNet train images to use")
    ap.add_argument("--vit-model", default=None, help="vit: the HF model to fine-tune (default: google/vit-base-patch16-224)")
    ap.add_argument("--images", default=None, help="vit: a photo folder built by e15_ages.py (ages.json + faces.json, "
                                                    "split 'train') instead of Tiny-ImageNet; its face sets join the "
                                                    "placements ('detector')")
    ap.add_argument("--ks", default="0,4,9,16,25,36,49,64,100", help="vit: sensitive sizes sampled per step (0 = plain control)")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.5,0.7", help="bert: sensitive fractions sampled per step (0 = plain control)")
    ap.add_argument("--bert-model", default=None, help="bert: the HF model to fine-tune (default: the SST-2 BERT)")
    ap.add_argument("--texts", default=None, help="bert: training texts in the SFHE_TEXT_DATA format (split 'train') "
                                                   "instead of SST-2 train")
    ap.add_argument("--private", default=None, help="bert: a detector file (split 'train'); its private sets join the "
                                                     "placements ('detector')")
    ap.add_argument("--max-len", type=int, default=64, help="bert: tokens per training text")
    ap.add_argument("--plain", action="store_true", help="bert: an ordinary fine-tune (full model, cross-entropy only), "
                                                        "e.g. a task head on bert-base-uncased")
    ap.add_argument("--num-labels", type=int, default=2, help="bert --plain: classes of a new head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--class-index", default=os.path.expanduser("~/data/selective-fhe/data/imagenet_class_index.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.batch = a.batch or (48 if a.model == "vit" else 32)
    a.out = a.out or os.path.expanduser(f"~/data/selective-fhe/e1b/{a.model}_oneway")
    return a


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def step_log(step, a, loss_ce, loss_kl, agree, tag, t0):
    print(f"[step {step:5d}/{a.steps}] ce {loss_ce:.3f} kl {loss_kl:.3f} agree-with-teacher {agree:.3f} {tag} "
          f"lr {lr_at(step, a):.1e} {time.time() - t0:.0f}s"
          + (f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB" if a.device.startswith("cuda") else ""), flush=True)


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_vit(a):
    import e1_oneway_vit as E
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageClassification
    dev = a.device
    model = AutoModelForImageClassification.from_pretrained(a.vit_model or E.MODEL).to(dev)
    teacher = freeze(copy.deepcopy(model))
    S, T = E.Vit(model), E.Vit(teacher)

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    side = E.GRID * E.PATCH
    faces = None
    if a.images:                                          # e15_ages.py photos; their face sets, bucketed by size
        root = os.path.expanduser(a.images)
        rows = json.load(open(f"{root}/ages.json"))["train"]
        det = json.load(open(f"{root}/faces.json"))["detections"]
        ims = [Image.open(f"{root}/{r[0]}") for r in rows]
        y_all = torch.as_tensor([r[1] for r in rows])
        faces = [det[r[0]]["patches"] for r in rows]
        buckets = {}
        for i, f in enumerate(faces):
            buckets.setdefault(len(f), []).append(i)
    else:
        ds = load_dataset("zh-plus/tiny-imagenet", split="train")
        names = ds.features["label"].names
        wnid2idx = {v[0]: int(k) for k, v in json.load(open(a.class_index)).items()}
        tiny2in = np.array([wnid2idx.get(w, -1) for w in names])
        labels = np.array(ds["label"])
        ok = np.where(tiny2in[labels] >= 0)[0]
        sel = np.sort(rng.choice(ok, size=min(a.train_n, len(ok)), replace=False))
        ims = ds.select(sel.tolist())["image"]
        y_all = torch.as_tensor(tiny2in[labels[sel]])
    imgs = np.empty((len(ims), side, side, 3), dtype=np.uint8)
    for i, im in enumerate(ims):
        imgs[i] = np.asarray(im.convert("RGB").resize((side, side), Image.BILINEAR))
    imgs = torch.from_numpy(imgs)
    print(f"[data] {len(imgs)} training images ready in {time.time() - t0:.0f}s", flush=True)

    ks = [int(k) for k in a.ks.split(",")]
    placements = ["random", "box", "salient"] + (["detector"] if faces else [])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        k, pl = random.choice(ks), random.choice(placements)
        if pl == "detector":                              # a batch of photos whose faces cover the same number of patches
            b = buckets[len(faces[random.randrange(len(faces))])]
            idx = torch.as_tensor(rng.choice(b, min(a.batch, len(b)), replace=False))
            k = len(faces[int(idx[0])])
        else:
            idx = torch.randint(0, len(imgs), (a.batch,))
        px = ((imgs[idx].to(dev).permute(0, 3, 1, 2).float() / 255.0) - 0.5) / 0.5
        y = y_all[idx].to(dev)
        with torch.no_grad():
            t_logits, sal = T.full(px)
        if pl == "detector":
            Sset = torch.as_tensor([faces[int(i)] for i in idx], dtype=torch.long, device=dev)
        else:
            Sset = E.choose_S(pl, k, sal, gen)
        _, lo = S.two_stream(px, Sset, "drop")
        loss_ce = F.cross_entropy(lo, y)
        loss_kl = F.kl_div(lo.log_softmax(1), t_logits.log_softmax(1), log_target=True, reduction="batchmean")
        loss = loss_ce + a.kl * loss_kl
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            agree = (lo.argmax(1) == t_logits.argmax(1)).float().mean().item()
            step_log(step, a, loss_ce.item(), loss_kl.item(), agree, f"{pl} k={k}", t0)
    model.eval()
    model.save_pretrained(a.out)
    print(f"[done] saved {a.out} after {time.time() - t0:.0f}s", flush=True)


def train_bert(a):
    import e1_oneway_bert as E
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = a.device
    name = a.bert_model or E.MODEL
    tok = AutoTokenizer.from_pretrained(name)
    if a.plain:
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=a.num_labels).to(dev)
        S, T = E.Bert(model), None
    else:
        model = AutoModelForSequenceClassification.from_pretrained(name).to(dev)
        teacher = freeze(copy.deepcopy(model))
        S, T = E.Bert(model), E.Bert(teacher)
    if a.texts:
        rows = json.load(open(os.path.expanduser(a.texts)))["train"]
        sents, labels = [t for t, _ in rows], torch.as_tensor([lab for _, lab in rows])
    else:
        ds = load_dataset("nyu-mll/glue", "sst2", split="train")
        sents, labels = ds["sentence"], torch.as_tensor(ds["label"])
    priv = json.load(open(os.path.expanduser(a.private)))["train"] if a.private else None
    print(f"[data] {len(sents)} training texts from {a.texts or 'SST-2'}; model {name}", flush=True)
    mask_id, sep_id = tok.mask_token_id, tok.sep_token_id

    fracs = [float(f) for f in a.fracs.split(",")]
    placements = ["random", "span", "salient"] + (["detector"] if priv else [])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        idx = rng.sample(range(len(sents)), a.batch)
        enc = tok([sents[i] for i in idx], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        ids, valid = enc["input_ids"].to(dev), enc["attention_mask"].bool().to(dev)
        y = labels[idx].to(dev)
        content = valid & (ids != sep_id)
        content[:, 0] = False
        if a.plain:
            lo, _ = S.full(ids, valid)
            loss_ce = F.cross_entropy(lo, y)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a)
            opt.zero_grad(set_to_none=True)
            loss_ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0 or step == a.steps - 1:
                acc = (lo.argmax(1) == y).float().mean().item()
                step_log(step, a, loss_ce.item(), 0.0, acc, "plain (agree = batch accuracy)", t0)
            continue
        frac, pl = rng.choice(fracs), rng.choice(placements)
        with torch.no_grad():
            t_logits, sal = T.full(ids, valid)
        if pl == "detector":                              # the detector's own private sets (texts without any: plain)
            Sset = torch.zeros_like(content)
            for r, i in enumerate(idx):
                for q in priv.get(str(i), []):
                    if q < Sset.shape[1]:
                        Sset[r, q] = True
            Sset &= content
            frac = "det"
        else:
            Sset = E.choose_S(pl, frac, sal, content, gen)
        _, lo = S.two_stream(ids, valid, Sset, "mask", mask_id, sep_id)
        loss_ce = F.cross_entropy(lo, y)
        loss_kl = F.kl_div(lo.log_softmax(1), t_logits.log_softmax(1), log_target=True, reduction="batchmean")
        loss = loss_ce + a.kl * loss_kl
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            agree = (lo.argmax(1) == t_logits.argmax(1)).float().mean().item()
            step_log(step, a, loss_ce.item(), loss_kl.item(), agree, f"{pl} f={frac}", t0)
    model.eval()
    model.config.output_attentions = False          # the HateXplain checkpoint sets it; transformers >= 5 refuses to save it
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[done] saved {a.out} after {time.time() - t0:.0f}s", flush=True)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "train_config.json"), "w"), indent=1)
    (train_vit if a.model == "vit" else train_bert)(a)


if __name__ == "__main__":
    main()
