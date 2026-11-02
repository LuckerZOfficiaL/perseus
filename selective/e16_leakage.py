#!/usr/bin/env python
"""E16 (what the server learns): attribute-inference attackers trained on each method's server view. The server holds
the task model, so each attacker starts from it and is fine-tuned on the TRAIN split's views, then scored on the
validation views; an attacker that sees nothing (full encryption) is at the majority rate.

Views (what the server receives):
  full          the whole input (no protection),
  ours          two-stream: the public part, with the private tokens' positions (text: [MASK] at each private word
                piece; photos: the private patches blacked out, the same information as dropping them),
  typed         text, Presidio-style placeholders (<MEDICATION>, <RELIGION>, ...),
  blur / pixelate   photos, B6's anonymisers of the face patches,
  ours_w<r>     text, also the r neighbouring words on each side encrypted (E4's margin),
  ours_s        text, every sentence with a private word encrypted,
  ours_d<r>     photos, the private patches dilated by r patches.
Attributes: drugsc, the condition (the task label; E4's fixed detector) or with --attr drug the reviewed drug itself
(content recovery, the 30 most frequent drugs); hatex, the identity category of the hidden
mention (ethnicity / religion / sexuality, posts whose mentions share one); ages, the age group. Reports accuracy with a
bootstrap 95% CI, the majority rate, and the encrypted tokens per input with E15's cost (s / input).

  python selective/e16_leakage.py --task drugsc --model ~/data/selective-fhe/e15/drugs_bert      (GPU, sat venv)
  python selective/e16_leakage.py --task hatex --model Hate-speech-CNERG/bert-base-uncased-hatexplain
  python selective/e16_leakage.py --task ages --model ~/data/selective-fhe/e15/vit_age           (GPU, anon venv)
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e16")
TEXT_COST = ((0, 0.0), (13, 1.78), (29, 5.51))                  # E15 drugsc: 12 per page, then 4 per page
FULL_BERT = 14.40
FACE_COST = ((0, 0.0), (20, 2.51), (31, 5.54), (63, 10.76))     # E15 ages, s per photo by private patches
FULL_VIT = 36.57
HATEX_KINDS = ("ethnicity", "religion", "sexuality")


def cost(k, table, full):
    for kmax, s in table:
        if k <= kmax:
            return s
    return full


def acc_ci(correct, n_boot=10000, seed=0):
    c = np.asarray(correct, float)
    idx = np.random.default_rng(seed).integers(0, len(c), (n_boot, len(c)))
    m = c[idx].mean(1)
    return float(c.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- text

def drug_names_of(split):
    """drugsc: the reviewed drug of each text of a split (drugs.com's drugName, matched on the text's start)."""
    from datasets import load_dataset
    import e15_drugs as E15
    ds = load_dataset("lewtun/drug-reviews")["test" if split == "validation" else "train"]
    return {E15.clean(r)[:200]: d for r, d in zip(ds["review"], ds["drugName"])}


def text_split(task, split, tok, T, n, seed, attr="label"):
    """[(ids per view, attribute, encrypted pieces per view)] for up to n inputs with private tokens (attr "drug":
    drugsc's reviewed drug, a name)."""
    from b5_text_sanitize import TASKS
    from b5b_typed_sanitize import detect, replace, PLACEHOLDER
    data, privf, _T = TASKS[task]
    rows = json.load(open(data))[split]
    priv = json.load(open(privf))[split]
    idx = [i for i in range(len(rows)) if priv.get(str(i))]
    rng = np.random.default_rng(seed)
    idx = sorted(rng.permutation(idx)[:n].tolist()) if n and len(idx) > n else idx
    texts = [rows[i][0] for i in idx]
    spans = detect(task, texts)
    drug = drug_names_of(split) if attr == "drug" else None
    out = []
    for i, t, sp in zip(idx, texts, spans):
        if task == "hatex":
            kinds = Counter(k for _s, _e, k in sp)
            if len(kinds) != 1 or next(iter(kinds)) not in HATEX_KINDS:
                continue
            attr = HATEX_KINDS.index(next(iter(kinds)))
        elif drug is not None:
            attr = drug.get(t[:200])
            if attr is None:
                continue
        else:
            attr = int(rows[i][1])
        enc = tok(t, truncation=True, max_length=T, return_offsets_mapping=True)
        ids, off, wid = enc["input_ids"], enc["offset_mapping"], enc.word_ids()
        body = [p for p in range(len(ids)) if wid[p] is not None]
        P = {p for p in priv[str(i)] if p < len(ids)}
        pw = {wid[p] for p in P}
        sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*", t)]
        hot = [(s, e) for s, e in sents if any(s <= off[p][0] < e for p in P)]
        sets = {"full": set(), "ours": P, "ours_s": {p for p in body if any(s <= off[p][0] < e for s, e in hot)} | P,
                "all": set(body)}
        for r in (1, 2):
            sets[f"ours_w{r}"] = {p for p in body if any(abs(wid[p] - w) <= r for w in pw)} | P
        v = {}
        for name, S in sets.items():
            v[name] = ([tok.mask_token_id if p in S else x for p, x in enumerate(ids)], len(S))
        typed = replace(t, sp, lambda _s, k: PLACEHOLDER.get(k, f"<{k.upper()}>"))
        v["typed"] = (tok(typed, truncation=True, max_length=T)["input_ids"], 0)
        out.append((v, attr))
    return out


def train_text(model_name, n_cls, train, val, view, a, dev):
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(a.seed)
    m = AutoModelForSequenceClassification.from_pretrained(os.path.expanduser(model_name), num_labels=n_cls,
                                                           ignore_mismatched_sizes=True).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(train) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    pad = lambda seqs: torch.nn.utils.rnn.pad_sequence([torch.tensor(s) for s in seqs], batch_first=True).to(dev)
    for step in range(steps):
        b = rng.choice(len(train), a.batch, replace=False)
        ids = pad([train[j][0][view][0] for j in b])
        y = torch.tensor([train[j][1] for j in b], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(m(input_ids=ids, attention_mask=(ids != 0)).logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    pred = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(0, len(val), 64):
            ids = pad([x[0][view][0] for x in val[j:j + 64]])
            pred += m(input_ids=ids, attention_mask=(ids != 0)).logits.argmax(-1).tolist()
    return [int(p == x[1]) for p, x in zip(pred, val)]


def run_text(a, dev):
    from transformers import AutoTokenizer
    from b5_text_sanitize import TASKS
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    T = TASKS[a.task][2]
    train = text_split(a.task, "train", tok, T, a.n_train, a.seed, a.attr)
    val = text_split(a.task, "validation", tok, T, 0, a.seed, a.attr)
    n_cls = len(HATEX_KINDS) if a.task == "hatex" else 10
    if a.attr == "drug":                                         # the top drugs of the train split, as classes
        top = [d for d, _c in Counter(x[1] for x in train).most_common(a.top_drugs)]
        train = [(v, top.index(d)) for v, d in train if d in top]
        val = [(v, top.index(d)) for v, d in val if d in top]
        n_cls = len(top)
    maj = Counter(x[1] for x in val).most_common(1)[0][1] / len(val)
    print(f"[e16] {a.task}: {len(train)} train / {len(val)} validation inputs with private content, {n_cls} classes, "
          f"majority {maj:.3f}", flush=True)
    res = dict(task=a.task, attr=a.attr, n_train=len(train), n_val=len(val), majority=maj, views={})
    for view in a.views.split(","):
        k = [x[0][view][1] for x in val]
        s = float(np.mean([cost(kk, TEXT_COST, FULL_BERT) for kk in k]))
        if view == "all":
            res["views"][view] = dict(acc=maj, ci=[maj, maj], tokens=float(np.mean(k)), s=FULL_BERT)
            continue
        c = train_text(a.model, n_cls, train, val, view, a, dev)
        m, lo, hi = acc_ci(c)
        res["views"][view] = dict(acc=m, ci=[lo, hi], tokens=float(np.mean(k)), s=s, correct=c)
        print(f"[e16] {a.task} {view:8s} attacker accuracy {m:.3f} [{lo:.3f}, {hi:.3f}]  encrypted pieces "
              f"{np.mean(k):5.1f}  s/input {s:5.2f}", flush=True)
    return res


# ---------------------------------------------------------------- photos

def dilate(patches, r):
    m = np.zeros((14, 14), bool)
    for p in patches:
        m[divmod(int(p), 14)] = True
    for _ in range(r):
        q = np.pad(m, 1)
        m = np.max([q[1 + dr:15 + dr, 1 + dc:15 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    return np.flatnonzero(m.ravel()).tolist()


def photo_split(rows, det, root, views, sigma):
    """Per view: uint8 (N, 224, 224, 3) images and encrypted patches per photo."""
    from PIL import Image
    from b6_image_anon import anonymise
    imgs = {v: np.empty((len(rows), 224, 224, 3), np.uint8) for v in views}
    k = {v: np.zeros(len(rows), int) for v in views}
    for i, (rel, _lab, _age) in enumerate(rows):
        im = np.asarray(Image.open(f"{root}/{rel}").convert("RGB").resize((224, 224), Image.BILINEAR))
        S = det[rel]["patches"]
        for v in views:
            if v == "full":
                imgs[v][i], k[v][i] = im, 0
            elif v.startswith("ours"):
                P = dilate(S, int(v[6:])) if v.startswith("ours_d") else S
                imgs[v][i], k[v][i] = anonymise(im, P, "black", sigma), len(P)
            elif v == "all":
                imgs[v][i], k[v][i] = 0, 196
            else:
                imgs[v][i], k[v][i] = anonymise(im, S, v, sigma), 0
    return imgs, k


def to_px(u8, dev):
    return ((torch.from_numpy(u8).to(dev).permute(0, 3, 1, 2).float() / 255.0) - 0.5) / 0.5


def train_photo(model_name, Xtr, ytr, Xva, yva, a, dev):
    from transformers import ViTForImageClassification
    torch.manual_seed(a.seed)
    m = ViTForImageClassification.from_pretrained(os.path.expanduser(model_name)).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(Xtr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    for step in range(steps):
        b = np.sort(rng.choice(len(Xtr), a.batch, replace=False))
        x = to_px(Xtr[b], dev)
        flip = torch.from_numpy(rng.random(len(b)) < 0.5).to(dev)
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        y = torch.from_numpy(ytr[b]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(m(pixel_values=x).logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    m.eval()
    pred = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(0, len(Xva), 64):
            pred += m(pixel_values=to_px(Xva[j:j + 64], dev)).logits.argmax(-1).tolist()
    return [int(p == y) for p, y in zip(pred, yva)]


def run_ages(a, dev):
    import e15_ages as E15
    meta = json.load(open(f"{E15.OUT}/ages.json"))
    det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    views = a.views.split(",")
    tr, va = meta["train"], meta["validation"]
    Xtr, _k = photo_split(tr, det, E15.OUT, [v for v in views if v != "all"], a.sigma)
    Xva, kva = photo_split(va, det, E15.OUT, views, a.sigma)
    ytr, yva = np.array([r[1] for r in tr]), np.array([r[1] for r in va])
    maj = Counter(yva.tolist()).most_common(1)[0][1] / len(yva)
    print(f"[e16] ages: {len(tr)} train / {len(va)} validation photos, majority {maj:.3f}", flush=True)
    res = dict(task="ages", n_train=len(tr), n_val=len(va), majority=maj, views={})
    for v in views:
        s = float(np.mean([cost(kk, FACE_COST, FULL_VIT) for kk in kva[v]]))
        if v == "all":
            res["views"][v] = dict(acc=maj, ci=[maj, maj], tokens=196.0, s=FULL_VIT)
            continue
        c = train_photo(a.model, Xtr[v], ytr, Xva[v], yva, a, dev)
        m, lo, hi = acc_ci(c)
        res["views"][v] = dict(acc=m, ci=[lo, hi], tokens=float(kva[v].mean()), s=s, correct=c)
        print(f"[e16] ages {v:8s} attacker accuracy {m:.3f} [{lo:.3f}, {hi:.3f}]  encrypted patches "
              f"{kva[v].mean():5.1f}  s/photo {s:5.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("drugsc", "hatex", "ages"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--views", default=None)
    ap.add_argument("--n-train", type=int, default=0, help="text: train inputs (0 = all with private content)")
    ap.add_argument("--attr", default="label", choices=("label", "drug"), help="drugsc: the condition (label) or the "
                    "reviewed drug (content recovery: the --top-drugs most frequent drugs as classes)")
    ap.add_argument("--top-drugs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    if a.task == "ages":
        a.views = a.views or "full,ours,blur,pixelate,ours_d1,ours_d2,ours_d4,all"
        a.lr = a.lr or 3e-5
        res = run_ages(a, dev)
    else:
        a.views = a.views or "full,ours,typed,ours_w1,ours_w2,ours_s,all"
        a.lr = a.lr or 2e-5
        res = run_text(a, dev)
    res["args"] = vars(a)
    json.dump(res, open(f"{OUT}/attr_{a.task}{'' if a.attr == 'label' else '_' + a.attr}.json", "w"))


if __name__ == "__main__":
    main()
