#!/usr/bin/env python
"""Baselines B5 / B5b, the privacy side: how often each word-level sanitiser hands the server the original private word
piece. Same texts, private positions and models (input embeddings) as b5_text_sanitize.py / b5b_typed_sanitize.py.
Per private word piece:
  random     1 / |whole-word vocabulary| if the piece is a whole word, else 0,
  dp<eps>    SanText-style substitution: the exact probability that the sampled replacement is the original,
  custext<eps>  CusText: the same, over the piece's K nearest whole words,
  emb<r>     embedding noise: the nearest-neighbour inversion (the noisy embedding's closest vocabulary embedding, the
             attack of Song and Raghunathan 2020) returns the original; Monte Carlo, --draws per piece.
mask / delete / typed / surrogate never send the piece (0); encryption sends nothing readable (0). Reports the mean over
the private word pieces.

  python selective/b5c_leakage.py --task sst2|hatex|hatexval|enron --model HF_DIR_OR_ID
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b5_text_sanitize import D, TASKS
from b5b_typed_sanitize import rows_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--eps", default="1,3,10", help="SanText-style epsilons (b5)")
    ap.add_argument("--custext-eps", default="1,3")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--emb-ratios", default="4,8,16")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _data, privf, T = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    rows = rows_of(a.task, "validation")
    priv = json.load(open(privf))["validation"]
    tids = []
    for i in range(len(rows)):
        if priv.get(str(i)):
            ids = tok(rows[i][0], truncation=True, max_length=T)["input_ids"]
            tids += [ids[p] for p in priv[str(i)] if 0 < p < len(ids) - 1]
    tids = torch.tensor(tids)
    model = AutoModelForSequenceClassification.from_pretrained(os.path.expanduser(a.model)).eval().double()
    emb = model.bert.embeddings.word_embeddings.weight.detach()
    words = torch.tensor(sorted(i for t, i in tok.get_vocab().items() if t.isalpha() and not t.startswith("##")))
    W = emb[words]
    scale = float(torch.cdist(W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))[:2000]],
                              W[:2000]).mean())
    pos = {int(w): j for j, w in enumerate(words)}           # whole word -> its index in W
    uniq, inv = tids.unique(return_inverse=True)
    out = dict(task=a.task, model=a.model, pieces=len(tids), whole_word=float(np.mean([int(t) in pos for t in tids])),
               random=float(np.mean([int(t) in pos for t in tids])) / len(words))
    d_all = torch.cdist(emb[uniq], W)                         # (unique pieces, whole words)
    has = torch.tensor([int(t) in pos for t in uniq])
    col = torch.tensor([pos.get(int(t), 0) for t in uniq])
    for e in [float(x) for x in a.eps.split(",") if x]:
        p = torch.softmax(-e * (d_all / scale) / 2, 1).gather(1, col[:, None])[:, 0] * has
        out[f"dp{e:g}"] = float(p[inv].mean())
    dk, ik = d_all.topk(a.K, 1, largest=False)
    u = 1 - (dk - dk.min(1, keepdim=True).values) / (dk.max(1, keepdim=True).values - dk.min(1, keepdim=True).values).clamp_min(1e-12)
    for e in [float(x) for x in a.custext_eps.split(",") if x]:
        p = (torch.softmax(e * u / 2, 1) * (ik == col[:, None])).sum(1) * has
        out[f"custext{e:g}"] = float(p[inv].mean())
    rng = np.random.default_rng(a.seed)
    d = emb.shape[1]
    for r in [float(x) for x in a.emb_ratios.split(",") if x]:
        eta = d / (r * scale)
        hit = []
        for _ in range(a.draws):
            z = rng.standard_normal((len(tids), d))
            z *= (rng.gamma(d, 1 / eta, len(tids)) / np.linalg.norm(z, axis=1))[:, None]
            noisy = emb[tids] + torch.from_numpy(z)
            nn = torch.cat([torch.cdist(noisy[k:k + 2048], emb).argmin(1) for k in range(0, len(noisy), 2048)])
            hit.append((nn == tids).double().mean())
        out[f"emb{r:g}"] = float(np.mean(hit))
    print(f"[b5c] {a.task} {os.path.basename(os.path.normpath(a.model))}: {len(tids)} private word pieces "
          f"({out['whole_word']:.0%} whole words); original recovered: " +
          ", ".join(f"{k} {v:.3f}" for k, v in out.items() if k not in ("task", "model", "pieces", "whole_word")), flush=True)
    tag = os.path.basename(os.path.normpath(a.model)).replace("/", "_")
    json.dump(out, open(f"{D}/e12/b5c_{a.task}_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
