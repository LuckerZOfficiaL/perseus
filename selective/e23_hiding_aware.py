#!/usr/bin/env python
"""E23 (blind-review follow-up), plaintext, encoders and ViTs on the tasks where the private part decides the answer
(drugsc, mts, ages). One model per (task, backbone, condition), trained with the backbone's ordinary fine-tune recipe
(E15 / E21: same base checkpoint, steps, batch, learning rate, schedule) on transformed training inputs:
  hiding-aware   text: mask / delete / typed (B5 / B5b transforms of the private word pieces or spans);
                 photos: black / gray / blur / pixelate (B6 anonymiser of the face patches) and drop (the face patches
                 removed from attention, the "redaction" view);
  private        the private part plus the readout only (text: [CLS] + the private word pieces + [SEP]; photos: CLS +
                 the face patches, every other patch removed from attention),
evaluated on the same transform of the validation inputs with private content, against OWTE (the one-way weights'
two-stream answer, as in the grid) with a paired bootstrap 95% interval. Models train on 90% of the train split (fixed
permutation); the other 10% ("held") gets log-probabilities for the late-fusion weight (e23_summary.py). Per-input
log-probabilities on held and validation are saved, the checkpoint is not.

  python selective/e23_hiding_aware.py --task drugsc --backbone bert-base --cond typed      (perseus venv)
  python selective/e23_hiding_aware.py --task ages --backbone deit-small --cond drop          (anon venv)
  python selective/e23_hiding_aware.py --cache-typed                                         (CPU: detector spans)
"""
import argparse
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
D = os.path.expanduser("~/data/selective-fhe")
OUT = f"{D}/e23"
TEXT = {  # backbone: (base checkpoint, {task: (steps, one-way weights)})
    "bert-base": ("bert-base-uncased", {"drugsc": (1500, f"{D}/e15/drugs_oneway"), "mts": (600, f"{D}/e15m/mts_oneway")}),
    "bert-large": ("bert-large-uncased", {"drugsc": (1500, f"{D}/e21/bert/drugsc_large_oneway"),
                                          "mts": (1500, f"{D}/e21/bert/mts_large_oneway")}),
}
VISION = {"vit-base": "google/vit-base-patch16-224", "vit-large": "google/vit-large-patch16-224",
          "deit-base": "facebook/deit-base-patch16-224", "deit-small": "facebook/deit-small-patch16-224"}
TEXT_CONDS = ("mask", "delete", "typed", "private")
IMAGE_CONDS = ("black", "gray", "blur", "pixelate", "drop", "private")


def split_train(n, seed=0):
    p = np.random.default_rng(seed).permutation(n)
    k = n // 10
    return np.sort(p[k:]), np.sort(p[:k])


def lr_at(step, steps, lr, warmup=100):                      # e1b_finetune.py's schedule
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


# ------------------------------------------------------------------ text
def typed_cache(task):
    return f"{OUT}/typed_spans_{task}.json"


def cache_typed():
    """Detector spans (with kinds) of every train and validation text, for the typed placeholders (B5b's detector)."""
    from b5b_typed_sanitize import detect, rows_of
    os.makedirs(OUT, exist_ok=True)
    for task in ("drugsc", "mts"):
        if os.path.exists(typed_cache(task)):
            continue
        out = {}
        for split in ("validation", "train"):
            t0 = time.time()
            out[split] = detect(task, [r[0] for r in rows_of(task, split)])
            print(f"[e23] {task} {split}: {len(out[split])} texts detected in {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(typed_cache(task), "w"))


def text_views(task, cond, tok, split, idx):
    """Token-id sequences of the given rows under the condition (None: the full text), labels, and the private sets."""
    from b5_text_sanitize import TASKS
    from b5b_typed_sanitize import PLACEHOLDER, replace, rows_of
    _d, privf, T = TASKS[task]
    rows = rows_of(task, split)
    priv = json.load(open(privf))[split]
    spans = json.load(open(typed_cache(task)))[split] if cond == "typed" else None
    seqs, y, S_all = [], [], []
    for i in idx:
        text, lab = rows[i]
        ids = tok(text, truncation=True, max_length=T)["input_ids"]
        S = sorted(p for p in priv.get(str(i), []) if 0 < p < len(ids) - 1)
        s = set(S)
        if cond is None:
            out = ids
        elif cond == "mask":
            out = [tok.mask_token_id if j in s else t for j, t in enumerate(ids)]
        elif cond == "delete":
            out = [t for j, t in enumerate(ids) if j not in s]
        elif cond == "typed":
            sp = [tuple(x) for x in spans[i]]
            out = tok(replace(text, sp, lambda _s, k: PLACEHOLDER.get(k, f"<{k.upper()}>")), truncation=True,
                      max_length=T)["input_ids"]
        elif cond == "private":
            out = [ids[0]] + [ids[j] for j in S] + [ids[-1]]
        else:
            raise ValueError(cond)
        seqs.append(out)
        y.append(int(lab))
        S_all.append(S)
    return seqs, np.array(y), S_all


def pad(seqs, pad_id, dev):
    L = max(map(len, seqs))
    ids = torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], device=dev)
    valid = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=dev).bool()
    return ids, valid


def text_val_rows(task, tok, cond):
    """Validation rows scored: those with private tokens (the grid's subset); typed: also the same positions on
    re-detection (B5b's subset)."""
    from b5_text_sanitize import TASKS
    from b5b_typed_sanitize import positions, rows_of
    _d, privf, T = TASKS[task]
    rows = rows_of(task, "validation")
    priv = json.load(open(privf))["validation"]
    keep = [i for i in range(len(rows)) if priv.get(str(i))]
    if cond == "typed":
        sp = json.load(open(typed_cache(task)))["validation"]
        keep = [i for i in keep if positions(tok, rows[i][0], [tuple(x) for x in sp[i]], T) == priv[str(i)]]
    return keep


def owte_text(task, backbone, tok, dev):
    """OWTE (one-way weights, two-stream, [MASK] placeholder) and full predictions on the validation rows with private
    tokens, cached (the grid's e21_text numbers, recomputed per input for the paired intervals)."""
    f = f"{OUT}/owte_{task}_{backbone}.json"
    if os.path.exists(f):
        return json.load(open(f))
    from transformers import AutoModelForSequenceClassification
    import e1_oneway_bert as E
    model = AutoModelForSequenceClassification.from_pretrained(TEXT[backbone][1][task][1]).eval().to(dev)
    bert = E.Bert(model)
    keep = text_val_rows(task, tok, None)
    seqs, y, S = text_views(task, None, tok, "validation", keep)
    two, full = [], []
    with torch.no_grad():
        for b in range(0, len(seqs), 32):
            ids, valid = pad(seqs[b:b + 32], tok.pad_token_id, dev)
            Sm = torch.zeros_like(valid)
            for r, s in enumerate(S[b:b + 32]):
                Sm[r, s] = True
            _, one = bert.two_stream(ids, valid, Sm, "mask", tok.mask_token_id, tok.sep_token_id)
            two += one.argmax(1).tolist()
            full += bert.full(ids, valid)[0].argmax(1).tolist()
    res = dict(rows=keep, labels=y.tolist(), two_stream=two, full=full)
    json.dump(res, open(f, "w"))
    return res


def run_text(a, dev):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import e1_oneway_bert as E
    from b5b_typed_sanitize import rows_of
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    base, tasks = TEXT[a.backbone]
    steps = a.steps or tasks[a.task][0]
    ref = owte_text(a.task, a.backbone, tok, dev)
    tr, held = split_train(len(rows_of(a.task, "train")))
    seqs, y, _ = text_views(a.task, a.cond, tok, "train", tr)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=10).to(dev)
    bert = E.Bert(model)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = rng.sample(range(len(seqs)), 32)
        ids, valid = pad([seqs[i] for i in idx], tok.pad_token_id, dev)
        lo, _ = bert.full(ids, valid)
        loss = F.cross_entropy(lo, torch.as_tensor(y[idx], device=dev))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 2e-5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"[e23] step {step}/{steps} ce {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()

    def logp(sq):
        out = []
        with torch.no_grad():
            for b in range(0, len(sq), 64):
                ids, valid = pad(sq[b:b + 64], tok.pad_token_id, dev)
                out.append(bert.full(ids, valid)[0].float().log_softmax(1).cpu())
        return torch.cat(out).numpy()

    keep = text_val_rows(a.task, tok, a.cond)
    vs, vy, _ = text_views(a.task, a.cond, tok, "validation", keep)
    hs, hy, _ = text_views(a.task, a.cond, tok, "train", held)
    return finish(a, logp(vs), vy, keep, logp(hs), hy, ref)


# ------------------------------------------------------------------ photos
def ages_rows(split):
    import e15_ages as E15
    rows = json.load(open(f"{E15.OUT}/ages.json"))[split]
    det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    return E15.OUT, rows, det


def load_imgs(root, rows):
    from PIL import Image
    out = np.empty((len(rows), 224, 224, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        out[i] = np.asarray(Image.open(f"{root}/{r[0]}").convert("RGB").resize((224, 224), Image.BILINEAR))
    return out


def image_view(imgs, faces, cond, sigma=8.0):
    if cond in ("drop", "private", None):
        return imgs
    from b6_image_anon import anonymise
    return np.stack([anonymise(im, S, cond, sigma) for im, S in zip(imgs, faces)])


def keep_mask(faces, cond, B, flips, dev):
    """Allowed keys (CLS + kept patches) for drop / private; None for pixel-level conditions."""
    if cond not in ("drop", "private"):
        return None
    m = torch.zeros(B, 197, dtype=torch.bool, device=dev)
    for r, (S, fl) in enumerate(zip(faces, flips)):
        S = [(k // 14) * 14 + (13 - k % 14) for k in S] if fl else S
        m[r, [k + 1 for k in S]] = True
    if cond == "drop":
        m = ~m
    m[:, 0] = True
    return m


def vit_logits(vit, px, allowed):
    import e1_oneway_vit as EV
    if allowed is None:
        return vit.full(px)[0]
    x = vit.emb(px)
    mask = allowed[:, None, None, :]
    for L in vit.layers:
        q, k, v = vit.qkv(L, L.layernorm_before(x))
        x = x + vit.attn_out(L, F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
        x = x + vit.mlp(L, L.layernorm_after(x))
    return vit.head(vit.lnf(x)[:, 0])


def owte_vision(backbone):
    r = json.load(open(f"{D}/e21/vision/ages_{backbone}_oneway.json"))
    return dict(rows=list(range(r["n"])), labels=r["labels"], two_stream=r["preds"]["two_stream"],
                full=r["preds"]["full"])


def run_vision(a, dev):
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    root, train, det = ages_rows("train")
    _r, val, _d = ages_rows("validation")
    ref = owte_vision(a.backbone)
    tr, held = split_train(len(train))
    faces = [det[r[0]]["patches"] for r in train]
    t0 = time.time()
    imgs = image_view(load_imgs(root, [train[i] for i in tr]), [faces[i] for i in tr], a.cond)
    ytr = np.array([train[i][1] for i in tr])
    ftr = [faces[i] for i in tr]
    print(f"[e23] {len(imgs)} training photos ({a.cond}) ready in {time.time() - t0:.0f}s", flush=True)
    model = ViTForImageClassification.from_pretrained(VISION[a.backbone], num_labels=3,
                                                      ignore_mismatched_sizes=True).to(dev).train()
    vit = EV.Vit(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    batch, epochs = 48, 3
    steps = a.steps or epochs * len(train) // batch          # the E15 recipe's step count (full train split)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-5, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    to_px = lambda arr: ((torch.from_numpy(np.ascontiguousarray(arr)).to(dev).permute(0, 3, 1, 2).float() / 255) - 0.5) / 0.5
    for step in range(steps):
        idx = rng.choice(len(imgs), batch, replace=False)
        flips = rng.random(batch) < 0.5
        arr = np.stack([imgs[i][:, ::-1] if f else imgs[i] for i, f in zip(idx, flips)])
        allowed = keep_mask([ftr[i] for i in idx], a.cond, batch, flips, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(pixel_values=to_px(arr)).logits if allowed is None else vit_logits(vit, to_px(arr), allowed)
            loss = F.cross_entropy(lo.float(), torch.as_tensor(ytr[idx], device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"[e23] step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()

    def logp(rows):
        fs = [det[r[0]]["patches"] for r in rows]
        ims = image_view(load_imgs(root, rows), fs, a.cond)
        out = []
        with torch.no_grad():
            for b in range(0, len(rows), 64):
                allowed = keep_mask(fs[b:b + 64], a.cond, len(fs[b:b + 64]), [False] * 64, dev)
                out.append(vit_logits(vit, to_px(ims[b:b + 64]), allowed).float().log_softmax(1).cpu())
        return torch.cat(out).numpy()

    vy = np.array([r[1] for r in val])
    hrows = [train[i] for i in held]
    return finish(a, logp(val), vy, list(range(len(val))), logp(hrows), np.array([r[1] for r in hrows]), ref)


# ------------------------------------------------------------------ shared
def finish(a, lp_val, y_val, rows, lp_held, y_held, ref):
    from e12_report import ci
    pos = {r: k for k, r in enumerate(ref["rows"])}
    k = [pos[r] for r in rows]
    yr = np.array(ref["labels"])[k]
    assert (yr == y_val).all(), "label order differs from the OWTE reference"
    pred = lp_val.argmax(1)
    two = np.array(ref["two_stream"])[k]
    m, lo, hi = ci(two == y_val, pred == y_val)
    res = dict(task=a.task, backbone=a.backbone, cond=a.cond, seed=a.seed, n=len(y_val),
               acc=float((pred == y_val).mean()), owte=float((two == y_val).mean()),
               full=float((np.array(ref["full"])[k] == y_val).mean()), owte_minus=[m, lo, hi],
               held_acc=float((lp_held.argmax(1) == y_held).mean()),
               rows=list(map(int, rows)), labels=y_val.tolist(), logp_val=np.round(lp_val, 5).tolist(),
               held_labels=y_held.tolist(), logp_held=np.round(lp_held, 5).tolist())
    od = a.out or OUT
    os.makedirs(od, exist_ok=True)
    f = f"{od}/{a.task}_{a.backbone}_{a.cond}{'' if a.seed == 0 else f'_s{a.seed}'}.json"
    json.dump(res, open(f, "w"))
    print(f"SUMMARY {a.task} {a.backbone} {a.cond}: acc {res['acc']:.3f} (n {res['n']}), OWTE {res['owte']:.3f}, "
          f"OWTE minus it {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("drugsc", "mts", "ages"))
    ap.add_argument("--backbone", choices=sorted(TEXT) + sorted(VISION))
    ap.add_argument("--cond", choices=TEXT_CONDS + IMAGE_CONDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="0: the recipe's steps (smoke tests only)")
    ap.add_argument("--out", default=None, help="result directory (default e23)")
    ap.add_argument("--cache-typed", action="store_true")
    a = ap.parse_args()
    if a.cache_typed:
        return cache_typed()
    dev = "cuda"
    (run_vision if a.task == "ages" else run_text)(a, dev)


if __name__ == "__main__":
    main()
