#!/usr/bin/env python
"""Baseline B5b (accuracy only, plaintext): the stronger text sanitisers. Same texts, models and private positions as B5
(b5_text_sanitize.py). The client's detector is re-run to recover each private span's kind (the detector files keep
positions only); texts whose re-detected positions differ from the stored ones are left out (counted). Methods:
  typed       each private span replaced by a typed placeholder, as Presidio's default anonymiser does (<PERSON>,
              <ORGANIZATION>, <LOCATION>, <MISC>, <EMAIL_ADDRESS>, <PHONE_NUMBER>; HateXplain: the identity category,
              <RELIGION>, <ETHNICITY>, ...),
  surrogate   each private span replaced by another real span of the same kind (pseudonymisation), drawn from the spans
              the detector finds in the task's train texts (HateXplain: another term of the same identity category),
  custext<eps>  CusText (Chen et al. 2023): each private word piece replaced by one of its K nearest whole-word tokens
              (input-embedding distance), sampled with probability ~ exp(eps * u / 2), u the similarity rescaled to [0, 1],
  emb<r>      the private word pieces' input embeddings perturbed with the metric-LDP noise of Split-and-Denoise (Mai et
              al. 2023; density ~ exp(-eta ||z||) in d = 768 dimensions), without its client-side denoiser; the expected
              noise norm d / eta is r times the mean distance between two words' embeddings (r in --emb-ratios).
Reports accuracy, agreement with the full model, and two-stream minus each method (paired bootstrap 95% CI), on the texts
with private tokens.

  python selective/b5b_typed_sanitize.py --task sst2|hatex|hatexval|enron|drugs|drugsc|mts --model HF_DIR_OR_ID
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
from b5_text_sanitize import D, TASKS

PLACEHOLDER = {"PER": "<PERSON>", "ORG": "<ORGANIZATION>", "LOC": "<LOCATION>", "MISC": "<MISC>",
               "EMAIL": "<EMAIL_ADDRESS>", "PHONE": "<PHONE_NUMBER>", "DRUG": "<MEDICATION>", "Medication": "<MEDICATION>",
               "Disease_disorder": "<CONDITION>", "CONDITION": "<CONDITION>", "Therapeutic_procedure": "<PROCEDURE>",
               "Diagnostic_procedure": "<PROCEDURE>"}


def rows_of(task, split):
    data = TASKS[task][0]
    if data:
        return json.load(open(data))[split]
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    return [[s, int(lab)] for s, lab in zip(ds["sentence"], ds["label"])]


def merge(sp):
    """Overlapping spans joined; the joined span keeps the kind of its longest member."""
    out = []
    for s, e, k in sorted(sp):
        if out and s < out[-1][1]:
            s0, e0, k0 = out[-1]
            out[-1] = (s0, max(e, e0), k if e - s > e0 - s0 else k0)
        else:
            out.append((s, e, k))
    return out


def detect(task, texts):
    """Per text: merged [(start, end, kind)] private character spans, as the task's detector (E11 / E13 / E14) finds them."""
    if task == "mts":
        import e15_mts as M
        sp = M.detector(procedures=True)(texts)
    elif task.startswith("drugs"):
        import e15_drugs as E15
        sp = E15.detector(conditions=task == "drugsc")(texts)
    elif task.startswith("hatex"):
        from e13_hatexplain import WORD
        sp = [[(m.start(), m.end(), WORD[m.group()]) for m in re.finditer(r"[a-z]+", t.lower()) if m.group() in WORD]
              for t in texts]
    else:
        from transformers import pipeline
        tagger = pipeline("token-classification", model="dslim/bert-base-NER-uncased", aggregation_strategy="simple",
                          device=-1)
        if task == "enron":
            import e14_enron as E14
            sp = E14.detect(texts, tagger, 0.5)
        else:
            sp = [[(int(e["start"]), int(e["end"]), e["entity_group"]) for e in es if e["score"] >= 0.5]
                  for es in tagger(texts, batch_size=64)]
    return [merge(s) for s in sp]


def positions(tok, text, sp, T):
    enc = tok(text, truncation=True, max_length=T, return_offsets_mapping=True)
    return [j for j, (c0, c1) in enumerate(enc["offset_mapping"])
            if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1, _ in sp)]


def replace(text, sp, fn):
    for s, e, k in reversed(sp):
        text = text[:s] + fn(text[s:e], k) + text[e:]
    return text


def predict(bert, pad, seqs, noisy=None, eta=None, rng=None):
    """Full-model labels of the token sequences; `noisy`: per sequence, positions whose word embedding gets d_chi noise."""
    L = max(map(len, seqs))
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
    valid = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs]).bool()
    if noisy is None:
        return bert.full(ids, valid)[0].argmax(1).tolist()
    we = bert.emb.word_embeddings(ids)
    for r, pos in enumerate(noisy):
        if pos:
            z = rng.standard_normal((len(pos), we.shape[-1]))
            z *= (rng.gamma(we.shape[-1], 1 / eta, len(pos)) / np.linalg.norm(z, axis=1))[:, None]
            we[r, pos] += torch.from_numpy(z).to(we)
    x = bert.emb(inputs_embeds=we)
    for layer in bert.layers:
        x = bert.layer(layer, x, valid[:, None, None, :])
    return bert.readout(x[:, 0]).argmax(1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--eps", default="1,3", help="CusText epsilons")
    ap.add_argument("--K", type=int, default=20, help="CusText candidate set size")
    ap.add_argument("--emb-ratios", default="4,8,16")
    ap.add_argument("--pool", type=int, default=5000, help="train texts the surrogates are drawn from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import e1_oneway_bert as E
    from e12_report import ci
    _data, privf, T = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    rows = rows_of(a.task, "validation")
    priv = json.load(open(privf))["validation"]
    sp_all = detect(a.task, [r[0] for r in rows])
    keep = [i for i in range(len(rows)) if priv.get(str(i))]
    same = [i for i in keep if positions(tok, rows[i][0], sp_all[i], T) == priv[str(i)]]
    print(f"[b5b] {a.task}: {len(keep)} texts with private tokens, {len(same)} with the same positions on re-detection",
          flush=True)
    rng = np.random.default_rng(a.seed)
    if a.task.startswith("hatex"):
        from e13_hatexplain import TERMS
        pool = {c: ws.split() for c, ws in TERMS.items()}
    else:
        tr = [r[0] for r in rows_of(a.task, "train")[:a.pool]]
        pool = {}
        for t, sp in zip(tr, detect(a.task, tr)):
            for s, e, k in sp:
                pool.setdefault(k, []).append(t[s:e])
    print(f"[b5b] surrogate pool: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(pool.items())), flush=True)
    fallback = Counter()

    def typed(_s, k):
        return PLACEHOLDER.get(k, f"<{k.upper()}>")

    def surrogate(s, k):
        cand = [w for w in pool.get(k, []) if w.lower() != s.lower()]
        if not cand:
            fallback[k] += 1
            return typed(s, k)
        return cand[rng.integers(len(cand))]

    model = AutoModelForSequenceClassification.from_pretrained(os.path.expanduser(a.model)).eval().double()
    bert = E.Bert(model)
    emb = model.bert.embeddings.word_embeddings.weight.detach()
    words = torch.tensor(sorted(i for t, i in tok.get_vocab().items() if t.isalpha() and not t.startswith("##")))
    W = emb[words]
    scale = float(torch.cdist(W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))[:2000]],
                              W[:2000]).mean())
    gen = torch.Generator().manual_seed(a.seed)
    eps_list = [float(e) for e in a.eps.split(",") if e]
    ratios = [float(r) for r in a.emb_ratios.split(",") if r]
    etas = {r: emb.shape[1] / (r * scale) for r in ratios}
    near = {}

    def custext(tid, eps):
        if tid not in near:
            d = torch.cdist(emb[tid][None], W)[0]
            dk, ik = d.topk(a.K, largest=False)
            near[tid] = (words[ik], 1 - (dk - dk.min()) / (dk.max() - dk.min()).clamp_min(1e-12))
        cand, u = near[tid]
        return int(cand[torch.multinomial(torch.softmax(eps * u / 2, 0), 1, generator=gen)])

    methods = ["typed", "surrogate"] + [f"custext{e:g}" for e in eps_list] + [f"emb{r:g}" for r in ratios]
    preds = {m: [] for m in ["full", "two_stream"] + methods}
    y, shown = [], 0
    for b0 in range(0, len(same), a.batch):
        idx = same[b0:b0 + a.batch]
        enc = [tok(rows[i][0], truncation=True, max_length=T)["input_ids"] for i in idx]
        S = [priv[str(i)] for i in idx]
        y += [int(rows[i][1]) for i in idx]
        texts = {"typed": [replace(rows[i][0], sp_all[i], typed) for i in idx],
                 "surrogate": [replace(rows[i][0], sp_all[i], surrogate) for i in idx]}
        if shown < 2:
            print(f"[b5b] example: {rows[idx[0]][0][:160]!r}\n      typed: {texts['typed'][0][:160]!r}\n"
                  f"      surrogate: {texts['surrogate'][0][:160]!r}", flush=True)
            shown += 1
        variants = {"full": enc}
        variants.update({m: [tok(t, truncation=True, max_length=T)["input_ids"] for t in ts] for m, ts in texts.items()})
        for e in eps_list:
            variants[f"custext{e:g}"] = [[custext(t, e) if j in set(s) else t for j, t in enumerate(ids)]
                                         for ids, s in zip(enc, S)]
        with torch.no_grad():
            for m, seqs in variants.items():
                preds[m] += predict(bert, tok.pad_token_id, seqs)
            for r in ratios:
                preds[f"emb{r:g}"] += predict(bert, tok.pad_token_id, enc, S, etas[r], rng)
            L = max(len(s) for s in enc)
            ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in enc])
            valid = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in enc]).bool()
            Sm = torch.zeros_like(valid)
            for r, s in enumerate(S):
                Sm[r, s] = True
            _, one = bert.two_stream(ids, valid, Sm, "mask", tok.mask_token_id, tok.sep_token_id)
            preds["two_stream"] += one.argmax(1).tolist()
    y = np.array(y)
    P = {m: np.array(v) for m, v in preds.items()}
    out = dict(task=a.task, model=a.model, texts_with_private=len(keep), n=len(y), full=float((P["full"] == y).mean()),
               two_stream=float((P["two_stream"] == y).mean()), K=a.K, emb_eta={f"emb{r:g}": etas[r] for r in ratios},
               word_distance=scale, surrogate_fallback=dict(fallback))
    print(f"[b5b] {a.task} {os.path.basename(os.path.normpath(a.model))}: {len(y)} texts; full {out['full']:.3f}, "
          f"two-stream {out['two_stream']:.3f}", flush=True)
    for m in methods:
        mm, lo, hi = ci(P["two_stream"] == y, P[m] == y)
        out[m] = dict(acc=float((P[m] == y).mean()), agree_full=float((P[m] == P["full"]).mean()), two_minus=[mm, lo, hi])
        print(f"[b5b] {m:10s} acc {out[m]['acc']:.3f} agree-with-full {out[m]['agree_full']:.3f} | two-stream minus it "
              f"{mm:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    tag = os.path.basename(os.path.normpath(a.model)).replace("/", "_")
    json.dump(out, open(a.out or f"{D}/e12/b5b_{a.task}_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
