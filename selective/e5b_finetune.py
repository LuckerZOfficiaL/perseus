#!/usr/bin/env python
"""E5b: fine-tune ViT-B/16 or BERT-base under the late-injection rule (E5), k_late sampled per step from --klates so
one model serves several hand-over depths; teacher = the frozen original, loss = CE + KL as in e1b_finetune.py.
Evaluate with e5_late_inject_{vit,bert}.py --model <out>."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1b_finetune as B1


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["vit", "bert"], required=True)
    ap.add_argument("--klates", default="1,2,3,4")
    ap.add_argument("--inject", default="replace", choices=["replace", "add", "attention"])
    ap.add_argument("--fusion-lr", type=float, default=1e-4, help="attention: learning rate of the new cross-attention blocks")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--train-n", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--class-index", default=os.path.expanduser("~/data/selective-fhe/data/imagenet_class_index.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.batch = a.batch or (32 if a.model == "vit" else 32)
    a.out = a.out or os.path.expanduser(f"~/data/selective-fhe/e5b/{a.model}_late_{a.inject}")
    return a


def train_vit(a):
    import e1_oneway_vit as E
    import e5_late_inject_vit as E5
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoModelForImageClassification
    dev = a.device
    model = AutoModelForImageClassification.from_pretrained(E.MODEL).to(dev)
    teacher = B1.freeze(copy.deepcopy(model))
    S_, T_ = E.Vit(model), E.Vit(teacher)
    ds = load_dataset("zh-plus/tiny-imagenet", split="train")
    names = ds.features["label"].names
    wnid2idx = {v[0]: int(k) for k, v in json.load(open(a.class_index)).items()}
    tiny2in = np.array([wnid2idx.get(w, -1) for w in names])
    labels = np.array(ds["label"])
    ok = np.where(tiny2in[labels] >= 0)[0]
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(ok, size=min(a.train_n, len(ok)), replace=False))
    sub = ds.select(sel.tolist())
    side = E.GRID * E.PATCH
    imgs = np.empty((len(sub), side, side, 3), dtype=np.uint8)
    for i, im in enumerate(sub["image"]):
        imgs[i] = np.asarray(im.convert("RGB").resize((side, side), Image.BILINEAR))
    imgs = torch.from_numpy(imgs)
    y_all = torch.as_tensor(tiny2in[labels[sel]])
    ks = [0, 4, 9, 16, 25, 36, 49, 64, 100]
    klates = [int(v) for v in a.klates.split(",")]
    placements = ["random", "box", "salient"]
    fusion = None
    groups = [dict(params=list(model.parameters()), lr=a.lr)]
    if a.inject == "attention":
        from e5_fusion import CrossFusion
        fusion = CrossFusion(model.config.hidden_size, model.config.num_attention_heads, klates).to(dev).train()
        groups.append(dict(params=list(fusion.parameters()), lr=a.fusion_lr))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, len(imgs), (a.batch,))
        px = ((imgs[idx].to(dev).permute(0, 3, 1, 2).float() / 255.0) - 0.5) / 0.5
        y = y_all[idx].to(dev)
        k, pl, kl = random.choice(ks), random.choice(placements), random.choice(klates)
        with torch.no_grad():
            t_logits, sal = T_.full(px)
        Sset = E.choose_S(pl, k, sal, gen)
        lo = E5.late_inject(S_, px, Sset, kl, "drop", a.inject, fusion)
        loss_ce = F.cross_entropy(lo, y)
        loss_kl = F.kl_div(lo.log_softmax(1), t_logits.log_softmax(1), log_target=True, reduction="batchmean")
        loss = loss_ce + a.kl * loss_kl
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = B1.lr_at(step, a) * lr0 / a.lr
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            B1.step_log(step, a, loss_ce.item(), loss_kl.item(), (lo.argmax(1) == t_logits.argmax(1)).float().mean().item(),
                        f"{pl} k={k} k_late={kl}", t0)
    model.eval(); model.save_pretrained(a.out)
    if fusion is not None:
        fusion.save(os.path.join(a.out, "fusion.pt"))
    print(f"[done] saved {a.out} after {time.time() - t0:.0f}s", flush=True)


def train_bert(a):
    import e1_oneway_bert as E
    import e5_late_inject_bert as E5
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = a.device
    tok = AutoTokenizer.from_pretrained(E.MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(E.MODEL).to(dev)
    teacher = B1.freeze(copy.deepcopy(model))
    S_, T_ = E.Bert(model), E.Bert(teacher)
    ds = load_dataset("nyu-mll/glue", "sst2", split="train")
    sents, labels = ds["sentence"], torch.as_tensor(ds["label"])
    mask_id, sep_id = tok.mask_token_id, tok.sep_token_id
    fracs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    klates = [int(v) for v in a.klates.split(",")]
    placements = ["random", "span", "salient"]
    fusion = None
    groups = [dict(params=list(model.parameters()), lr=a.lr)]
    if a.inject == "attention":
        from e5_fusion import CrossFusion
        fusion = CrossFusion(model.config.hidden_size, model.config.num_attention_heads, klates).to(dev).train()
        groups.append(dict(params=list(fusion.parameters()), lr=a.fusion_lr))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model.train()
    t0 = time.time()
    for step in range(a.steps):
        idx = rng.sample(range(len(sents)), a.batch)
        enc = tok([sents[i] for i in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
        ids, valid = enc["input_ids"].to(dev), enc["attention_mask"].bool().to(dev)
        y = labels[idx].to(dev)
        content = valid & (ids != sep_id); content[:, 0] = False
        frac, pl, kl = rng.choice(fracs), rng.choice(placements), rng.choice(klates)
        with torch.no_grad():
            t_logits, sal = T_.full(ids, valid)
        Sset = E.choose_S(pl, frac, sal, content, gen)
        lo = E5.late_inject(S_, ids, valid, Sset, kl, "mask", mask_id, a.inject, fusion)
        loss_ce = F.cross_entropy(lo, y)
        loss_kl = F.kl_div(lo.log_softmax(1), t_logits.log_softmax(1), log_target=True, reduction="batchmean")
        loss = loss_ce + a.kl * loss_kl
        for g, lr0 in zip(opt.param_groups, base_lrs):
            g["lr"] = B1.lr_at(step, a) * lr0 / a.lr
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            B1.step_log(step, a, loss_ce.item(), loss_kl.item(), (lo.argmax(1) == t_logits.argmax(1)).float().mean().item(),
                        f"{pl} f={frac} k_late={kl}", t0)
    model.eval(); model.save_pretrained(a.out); tok.save_pretrained(a.out)
    if fusion is not None:
        fusion.save(os.path.join(a.out, "fusion.pt"))
    print(f"[done] saved {a.out} after {time.time() - t0:.0f}s", flush=True)


def main():
    a = parse()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "train_config.json"), "w"), indent=1)
    (train_vit if a.model == "vit" else train_bert)(a)


if __name__ == "__main__":
    main()
