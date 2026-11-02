#!/usr/bin/env python
"""E28 privacy axis (CPU): per method and level, the share of private word pieces the server recovers exactly from what
it receives. Substitution (SanText-style, CusText): the sent token is the original. Embedding noise: the nearest
vocabulary embedding of the noisy embedding is the original (the nearest-neighbour inversion of Song & Raghunathan 2020 /
Qu et al. 2021). Same samplers as B5 / B5b, on the validation private tokens of drugsc and mts, with each backbone's
one-way weights. Adds "recovered" to every test-time point of e28/summary.json.

  python selective/e28_recovery.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e23_hiding_aware as H

E = os.path.expanduser("~/data/selective-fhe/e28")


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    summ = json.load(open(f"{E}/summary.json"))
    for key, res in summ["results"].items():
        task, bk = key.split("/")
        emb = AutoModelForSequenceClassification.from_pretrained(H.TEXT[bk][1][task][1]).bert.embeddings \
            .word_embeddings.weight.detach().double()
        words = torch.tensor(sorted(i for t, i in tok.get_vocab().items() if t.isalpha() and not t.startswith("##")))
        W = emb[words]
        scale = float(torch.cdist(W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))[:2000]],
                                  W[:2000]).mean())
        keep = H.text_val_rows(task, tok, None)
        seqs, _y, S = H.text_views(task, None, tok, "validation", keep)
        toks = torch.tensor([ids[p] for ids, s in zip(seqs, S) for p in s])
        u_toks, inv = toks.unique(return_inverse=True)
        D = torch.cdist(emb[u_toks], W)                               # unique private tokens x whole-word vocabulary
        gen = torch.Generator().manual_seed(0)
        rng = np.random.default_rng(0)
        for m in ("dp", "custext", "emb"):
            for p in res["methods"][m]["test_time"]:
                lv = p["level"]
                if m == "dp":
                    P = torch.softmax(-lv * (D / scale) / 2, 1)[inv]
                    sent = words[torch.multinomial(P, 1, generator=gen)[:, 0]]
                    rec = (sent == toks).double().mean()
                elif m == "custext":
                    dk, ik = D.topk(20, 1, largest=False)
                    u = 1 - (dk - dk.min(1, keepdim=True)[0]) / (dk.max(1, keepdim=True)[0] - dk.min(1, keepdim=True)[0]).clamp_min(1e-12)
                    P = torch.softmax(lv * u / 2, 1)[inv]
                    sent = words[ik[inv].gather(1, torch.multinomial(P, 1, generator=gen))[:, 0]]
                    rec = (sent == toks).double().mean()
                else:
                    eta = emb.shape[1] / (lv * scale)
                    z = rng.standard_normal((len(toks), emb.shape[1]))
                    z *= (rng.gamma(emb.shape[1], 1 / eta, len(toks)) / np.linalg.norm(z, axis=1))[:, None]
                    noisy = emb[toks] + torch.from_numpy(z)
                    best = torch.cat([torch.cdist(noisy[b:b + 512], emb).argmin(1)     # inversion over the whole vocabulary
                                      for b in range(0, len(toks), 512)])
                    rec = (best == toks).double().mean()
                p["recovered"] = float(rec)
            print(f"[e28] {key} {m}: " + ", ".join(f"{p['level']:g}:{p['recovered']:.2f}" for p in res["methods"][m]["test_time"]),
                  flush=True)
    summ["methods_note"] = ("recovered = share of validation private word pieces the server gets back exactly (sent token "
                            "equals the original, or nearest vocabulary embedding of the noisy embedding is the original)")
    json.dump(summ, open(f"{E}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
