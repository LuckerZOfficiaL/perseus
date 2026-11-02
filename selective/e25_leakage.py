#!/usr/bin/env python
"""E25 (privacy analyses for the blind reviews): E16's attackers on more server views, and the leakage-vs-cost frontier.

Text (drugsc condition / drug, hatex group), attackers as E16 (start from the server's task model, fine-tuned on the
TRAIN split's views, scored on validation). Views:
  full, ours (= the "mask" hiding view: [MASK] at each private piece; same ids), delete (private pieces removed),
  typed, ours_w1 / ours_w3 (r neighbouring words encrypted too), ours_s (every sentence with a private word), all.
Photos (age group), views:
  full, ours (= "black": private patches blacked out, the same information as dropping them), ours_b1.25 / 1.5 / 2 / 3 (the
  face's bounding box in patch units enlarged by the factor around its centre), ours_ub (upper body: 3x the face width,
  down to 4x the face height below its top), all; genface (B6b's generated face) on a --genface-train subset of the train
  photos (its generated train images in e25/ages_genface_train), with full / ours on the same subset for a matched row.
--owte: plaintext OWTE accuracy of the one-way weights at every frontier level (drugsc: H13.accuracy with the level's
private set; ages: e1_oneway_vit two-stream, drop), with the per-input cost from the measured per-layout costs
(e16_leakage TEXT_COST / FACE_COST, more than the largest layout = the full model): DERIVED costs, not new runs.

  python selective/e25_leakage.py --task drugsc --model ~/data/selective-fhe/e15/drugs_bert           (sat venv, GPU)
  python selective/e25_leakage.py --task drugsc --attr drug --model ...
  python selective/e25_leakage.py --task hatex --model Hate-speech-CNERG/bert-base-uncased-hatexplain --views full,ours,delete,typed
  python selective/e25_leakage.py --task ages --model ~/data/selective-fhe/e15/vit_age                (anon venv, GPU)
  python selective/e25_leakage.py --task ages --genface --model ...
  python selective/e25_leakage.py --task drugsc --owte / --task ages --owte
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
import e16_leakage as E16
from e16_leakage import FACE_COST, FULL_BERT, FULL_VIT, TEXT_COST, acc_ci, cost

OUT = os.path.expanduser("~/data/selective-fhe/e25")
D = os.path.expanduser("~/data/selective-fhe")
TEXT_LEVELS = ("ours", "ours_w1", "ours_w3", "ours_s", "all")
FACE_LEVELS = ("ours", "ours_b1.25", "ours_b1.5", "ours_b2", "ours_b3", "ours_ub", "all")


# ---------------------------------------------------------------- text

def text_sets(t, ids, off, wid, P):
    body = [p for p in range(len(ids)) if wid[p] is not None]
    pw = {wid[p] for p in P}
    sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*", t)]
    hot = [(s, e) for s, e in sents if any(s <= off[p][0] < e for p in P)]
    sets = {"ours": set(P), "ours_s": {p for p in body if any(s <= off[p][0] < e for s, e in hot)} | set(P),
            "all": set(body)}
    for r in (1, 3):
        sets[f"ours_w{r}"] = {p for p in body if any(abs(wid[p] - w) <= r for w in pw)} | set(P)
    return sets


def text_split(task, split, tok, T, n, seed, attr="label"):
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
    drug = E16.drug_names_of(split) if attr == "drug" else None
    out = []
    for i, t, sp in zip(idx, texts, spans):
        if task == "hatex":
            kinds = Counter(k for _s, _e, k in sp)
            if len(kinds) != 1 or next(iter(kinds)) not in E16.HATEX_KINDS:
                continue
            y = E16.HATEX_KINDS.index(next(iter(kinds)))
        elif drug is not None:
            y = drug.get(t[:200])
            if y is None:
                continue
        else:
            y = int(rows[i][1])
        enc = tok(t, truncation=True, max_length=T, return_offsets_mapping=True)
        ids, off, wid = enc["input_ids"], enc["offset_mapping"], enc.word_ids()
        P = {p for p in priv[str(i)] if p < len(ids)}
        v = {"full": (ids, 0)}
        for name, S in text_sets(t, ids, off, wid, P).items():
            v[name] = ([tok.mask_token_id if p in S else x for p, x in enumerate(ids)], len(S))
        v["delete"] = ([x for p, x in enumerate(ids) if p not in P], 0)
        typed = replace(t, sp, lambda _s, k: PLACEHOLDER.get(k, f"<{k.upper()}>"))
        v["typed"] = (tok(typed, truncation=True, max_length=T)["input_ids"], 0)
        out.append((v, y))
    return out


def run_text(a, dev):
    from transformers import AutoTokenizer
    from b5_text_sanitize import TASKS
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    T = TASKS[a.task][2]
    train = text_split(a.task, "train", tok, T, a.n_train, a.seed, a.attr)
    val = text_split(a.task, "validation", tok, T, 0, a.seed, a.attr)
    n_cls = len(E16.HATEX_KINDS) if a.task == "hatex" else 10
    if a.attr == "drug":
        top = [d for d, _c in Counter(x[1] for x in train).most_common(a.top_drugs)]
        train = [(v, top.index(d)) for v, d in train if d in top]
        val = [(v, top.index(d)) for v, d in val if d in top]
        n_cls = len(top)
    maj = Counter(x[1] for x in val).most_common(1)[0][1] / len(val)
    print(f"[e25] {a.task}/{a.attr}: {len(train)} train / {len(val)} validation, {n_cls} classes, majority {maj:.3f}",
          flush=True)
    res = dict(task=a.task, attr=a.attr, n_train=len(train), n_val=len(val), majority=maj, views={},
               cost_note="s/input DERIVED from measured per-layout page costs (e16_leakage TEXT_COST), not new runs")
    for view in a.views.split(","):
        k = [x[0][view][1] for x in val]
        s = float(np.mean([cost(kk, TEXT_COST, FULL_BERT) for kk in k]))
        if view == "all":
            res["views"][view] = dict(acc=maj, ci=[maj, maj], tokens=float(np.mean(k)), s=FULL_BERT)
            continue
        c = E16.train_text(a.model, n_cls, train, val, view, a, dev)
        m, lo, hi = acc_ci(c)
        res["views"][view] = dict(acc=m, ci=[lo, hi], tokens=float(np.mean(k)), s=s, correct=c)
        print(f"[e25] {a.task} {view:9s} attacker {m:.3f} [{lo:.3f}, {hi:.3f}]  encrypted pieces {np.mean(k):5.1f}  "
              f"s/input {s:5.2f}", flush=True)
    return res


def owte_text(a):
    """Plaintext two-stream accuracy of the one-way weights with each frontier level's private set (validation)."""
    from transformers import AutoTokenizer
    from b5_text_sanitize import TASKS
    import e13_hatexplain as H13
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    data_f, privf, T = TASKS[a.task]
    data = json.load(open(data_f))
    priv = json.load(open(privf))["validation"]
    levels = {L: {} for L in TEXT_LEVELS}
    K = {L: [] for L in TEXT_LEVELS}
    for i, (t, _y) in enumerate(data["validation"]):
        P0 = priv.get(str(i), [])
        enc = tok(t, truncation=True, max_length=T, return_offsets_mapping=True)
        if not P0:
            for L in TEXT_LEVELS:
                levels[L][str(i)] = []
            continue
        sets = text_sets(t, enc["input_ids"], enc["offset_mapping"], enc.word_ids(), {p for p in P0 if p < len(enc["input_ids"])})
        for L in TEXT_LEVELS:
            levels[L][str(i)] = sorted(sets[L])
            K[L].append(len(sets[L]))
    res = dict(task=a.task, model=a.owte_model, levels={},
               cost_note="s/input DERIVED from measured per-layout page costs (e16_leakage TEXT_COST), not new runs")
    for L in TEXT_LEVELS:
        path = f"{OUT}/owte_{a.task}_{L}.json"
        H13.accuracy(os.path.expanduser(a.owte_model), tok, data, {"validation": levels[L]}, path, max_len=T,
                     focus=("c0", 0))
        r = json.load(open(path))["with_private"]
        s = float(np.mean([cost(k, TEXT_COST, FULL_BERT) for k in K[L]]))
        res["levels"][L] = dict(owte=r["two_stream"], full=r["full"], redaction=r["redaction"], n=r["n"],
                                tokens=float(np.mean(K[L])), s=s)
        print(f"[e25] owte {a.task} {L:8s} acc {r['two_stream']:.3f} (full {r['full']:.3f})  pieces {np.mean(K[L]):5.1f}  "
              f"s/input {s:5.2f}", flush=True)
    return res


# ---------------------------------------------------------------- photos

def grow_box(patches, kind):
    """Private patches of a frontier level from the face patches: bounding box (patch units) scaled about its centre."""
    if kind == "ours":
        return sorted(int(p) for p in patches)
    if kind == "all":
        return list(range(196))
    rr = [int(p) // 14 for p in patches]
    cc = [int(p) % 14 for p in patches]
    r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1      # [r0, r1) x [c0, c1)
    h, w = r1 - r0, c1 - c0
    if kind == "ours_ub":
        R0, R1 = r0, r0 + 4 * h
        C0, C1 = (c0 + c1) / 2 - 1.5 * w, (c0 + c1) / 2 + 1.5 * w
    else:
        f = float(kind[len("ours_b"):])
        R0, R1 = (r0 + r1) / 2 - f * h / 2, (r0 + r1) / 2 + f * h / 2
        C0, C1 = (c0 + c1) / 2 - f * w / 2, (c0 + c1) / 2 + f * w / 2
    R0, R1, C0, C1 = max(0, int(np.floor(R0))), min(14, int(np.ceil(R1))), max(0, int(np.floor(C0))), min(14, int(np.ceil(C1)))
    S = {r * 14 + c for r in range(R0, R1) for c in range(C0, C1)} | {int(p) for p in patches}
    return sorted(S)


def load_img(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR))


def photo_views(rows, det, views, sigma, genface_dir=None):
    from b6_image_anon import anonymise
    import e15_ages as E15
    imgs = {v: np.empty((len(rows), 224, 224, 3), np.uint8) for v in views if v != "all"}
    k = {v: np.zeros(len(rows), int) for v in views}
    for i, (rel, _lab, _age) in enumerate(rows):
        im = load_img(f"{E15.OUT}/{rel}")
        S = det[rel]["patches"]
        for v in views:
            if v == "full":
                imgs[v][i] = im
            elif v.startswith("ours"):
                P = grow_box(S, v)
                imgs[v][i], k[v][i] = anonymise(im, P, "black", sigma), len(P)
            elif v == "all":
                k[v][i] = 196
            elif v == "genface":
                imgs[v][i] = load_img(os.path.join(genface_dir, os.path.splitext(rel)[0] + ".png"))
            else:
                imgs[v][i] = anonymise(im, S, v, sigma)
    return imgs, k


def gen_faces(rows, det, out, seed=0):
    """B6b's generative face replacement (same model and settings) for the given rows."""
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    import e15_ages as E15
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting",
                                                          torch_dtype=torch.float16, variant="fp16",
                                                          safety_checker=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for k, (rel, _l, _a) in enumerate(rows):
        dst = os.path.join(out, os.path.splitext(rel)[0] + ".png")
        if os.path.exists(dst):
            continue
        img = Image.open(f"{E15.OUT}/{rel}").convert("RGB").resize((224, 224), Image.BILINEAR)
        mask = np.zeros((224, 224), dtype=np.uint8)
        for p in det[rel]["patches"]:
            r, c = divmod(int(p), 14)
            mask[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = 255
        gen = pipe(prompt="a photo of a person's face, natural, realistic", negative_prompt="blurry, deformed, cartoon, painting",
                   image=img.resize((512, 512), Image.BICUBIC), mask_image=Image.fromarray(mask).resize((512, 512), Image.NEAREST),
                   height=512, width=512, num_inference_steps=30, guidance_scale=7.5,
                   generator=torch.Generator("cuda").manual_seed(seed + 100000 + k)).images[0]
        gen = np.asarray(gen.resize((224, 224), Image.BILINEAR))
        o = np.where(mask[..., None] > 0, gen, np.asarray(img))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(o.astype(np.uint8)).save(dst)
        if (k + 1) % 250 == 0:
            print(f"[e25] genface train {k + 1}/{len(rows)}", flush=True)


def run_ages(a, dev):
    import e15_ages as E15
    meta = json.load(open(f"{E15.OUT}/ages.json"))
    det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    tr, va = meta["train"], meta["validation"]
    views = a.views.split(",")
    gdir_tr = f"{OUT}/ages_genface_train"
    if a.genface:
        sub = np.sort(np.random.default_rng(a.seed).permutation(len(tr))[:a.genface_train])
        tr = [tr[i] for i in sub]
        gen_faces(tr, det, gdir_tr, a.seed)
    Xtr, _ = photo_views(tr, det, [v for v in views if v != "all"], a.sigma, gdir_tr)
    Xva, kva = photo_views(va, det, views, a.sigma, f"{D}/e15/ages_genface")
    ytr, yva = np.array([r[1] for r in tr]), np.array([r[1] for r in va])
    maj = Counter(yva.tolist()).most_common(1)[0][1] / len(yva)
    print(f"[e25] ages: {len(tr)} train / {len(va)} validation photos, majority {maj:.3f}", flush=True)
    res = dict(task="ages", n_train=len(tr), n_val=len(va), majority=maj, views={},
               cost_note="s/photo DERIVED from measured per-layout page costs (e16_leakage FACE_COST), not new runs")
    for v in views:
        s = float(np.mean([cost(kk, FACE_COST, FULL_VIT) for kk in kva[v]]))
        if v == "all":
            res["views"][v] = dict(acc=maj, ci=[maj, maj], tokens=196.0, s=FULL_VIT)
            continue
        c = E16.train_photo(a.model, Xtr[v], ytr, Xva[v], yva, a, dev)
        m, lo, hi = acc_ci(c)
        res["views"][v] = dict(acc=m, ci=[lo, hi], tokens=float(kva[v].mean()), s=s, correct=c)
        print(f"[e25] ages {v:9s} attacker {m:.3f} [{lo:.3f}, {hi:.3f}]  encrypted patches {kva[v].mean():5.1f}  "
              f"s/photo {s:5.2f}", flush=True)
    return res


def owte_ages(a, dev):
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    import e15_ages as E15
    va = json.load(open(f"{E15.OUT}/ages.json"))["validation"]
    det = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    vit = EV.Vit(ViTForImageClassification.from_pretrained(os.path.expanduser(a.owte_model)).eval().to(dev))
    y = np.array([lab for _r, lab, _a in va])
    pred = {L: [] for L in FACE_LEVELS}
    full, K = [], {L: [] for L in FACE_LEVELS}
    for rel, _l, _a in va:
        px = EV.preprocess([Image.open(f"{E15.OUT}/{rel}").convert("RGB")]).to(dev)
        with torch.no_grad():
            f = vit.full(px)[0]
            full.append(int(f.argmax()))
            for L in FACE_LEVELS:
                S = grow_box(det[rel]["patches"], L)
                K[L].append(len(S))
                if len(S) >= 196:                                # everything encrypted: the full model
                    pred[L].append(int(f.argmax()))
                else:
                    _lr, lo = vit.two_stream(px, torch.tensor([S], device=dev), "drop")
                    pred[L].append(int(lo.argmax()))
    res = dict(task="ages", model=a.owte_model, full=float((np.array(full) == y).mean()), levels={},
               cost_note="s/photo DERIVED from measured per-layout page costs (e16_leakage FACE_COST), not new runs")
    for L in FACE_LEVELS:
        s = float(np.mean([cost(k, FACE_COST, FULL_VIT) for k in K[L]]))
        res["levels"][L] = dict(owte=float((np.array(pred[L]) == y).mean()), tokens=float(np.mean(K[L])), s=s,
                                over_63=int(sum(k > 63 for k in K[L])))
        print(f"[e25] owte ages {L:9s} acc {res['levels'][L]['owte']:.3f} (full {res['full']:.3f})  patches "
              f"{np.mean(K[L]):5.1f}  s/photo {s:5.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("drugsc", "hatex", "ages"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--views", default=None)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--attr", default="label", choices=("label", "drug"))
    ap.add_argument("--top-drugs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--genface", action="store_true", help="ages: genface on a train subset, with matched full / ours")
    ap.add_argument("--genface-train", type=int, default=3000)
    ap.add_argument("--owte", action="store_true")
    ap.add_argument("--owte-model", default=None)
    ap.add_argument("--device", default="cuda", help="cpu: run on the CPU (tensors kept on the CPU in code)")
    a = ap.parse_args()
    dev = a.device
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("E25_THREADS", "48")))
    os.makedirs(OUT, exist_ok=True)
    if a.owte:
        if a.task == "ages":
            a.owte_model = a.owte_model or f"{D}/e15/ages_oneway"
            res = owte_ages(a, dev)
        else:
            a.owte_model = a.owte_model or f"{D}/e15/drugs_oneway"
            res = owte_text(a)
        json.dump(res, open(f"{OUT}/owte_frontier_{a.task}.json", "w"), indent=1)
        return
    if a.task == "ages":
        a.views = a.views or ("full,ours,genface" if a.genface else ",".join(("full", "blur") + FACE_LEVELS))
        a.lr = a.lr or 3e-5
        res = run_ages(a, dev)
        tag = "ages_genface" if a.genface else "ages"
    else:
        a.views = a.views or ",".join(("full", "delete", "typed") + TEXT_LEVELS)
        a.lr = a.lr or 2e-5
        res = run_text(a, dev)
        tag = a.task + ("" if a.attr == "label" else "_" + a.attr)
    res["args"] = vars(a)
    json.dump(res, open(f"{OUT}/attr_{tag}.json", "w"))


if __name__ == "__main__":
    main()
