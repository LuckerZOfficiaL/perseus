#!/usr/bin/env python
"""E28 (blind-review follow-up), plaintext: the text hiding baselines that have a privacy parameter, trained on their own
noisy inputs at one level (the test-time sweep over levels is B5 / B5b with longer --eps / --emb-ratios lists).
  dp<eps>       SanText-style metric-DP substitution (B5): replacement ~ exp(-eps * d / 2), d the input-embedding
                distance over the whole-word vocabulary, normalised to unit mean distance;
  custext<eps>  CusText (B5b): one of the K = 20 nearest whole words, ~ exp(eps * u / 2), u the similarity in [0, 1];
  emb<r>        Split-and-Denoise's metric-LDP embedding noise on the private word pieces (B5b), expected norm r times
                the mean word distance, without the denoiser; fresh noise every training step.
Substitutions are sampled with the base checkpoint's input embeddings (one draw per text, seed 0); the model is trained
with the backbone's ordinary recipe on 90% of the train split (as E23) and scored on the same transform of the validation
texts with private tokens, against OWTE (E23's per-input reference) with a paired bootstrap 95% CI.

  python selective/e28_privacy_levels.py --task drugsc --backbone bert-base --method dp --level 10
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e23_hiding_aware as H

OUT = os.path.expanduser("~/data/selective-fhe/e28")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("drugsc", "mts"))
    ap.add_argument("--backbone", required=True, choices=sorted(H.TEXT))
    ap.add_argument("--method", required=True, choices=("dp", "custext", "emb"))
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import e1_oneway_bert as E
    from b5b_typed_sanitize import rows_of
    from e12_report import ci
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    base, tasks = H.TEXT[a.backbone]
    steps = a.steps or tasks[a.task][0]
    ref = H.owte_text(a.task, a.backbone, tok, dev)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=10).to(dev)
    bert = E.Bert(model)
    emb = model.bert.embeddings.word_embeddings.weight.detach().clone().cpu()
    words = torch.tensor(sorted(i for t, i in tok.get_vocab().items() if t.isalpha() and not t.startswith("##")))
    W = emb[words]
    scale = float(torch.cdist(W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))[:2000]],
                              W[:2000]).mean())
    gen = torch.Generator().manual_seed(0)
    cache = {}

    def sub(tid):
        if a.method == "dp":
            if tid not in cache:
                cache[tid] = torch.softmax(-a.level * torch.cdist(emb[tid][None], W)[0] / scale / 2, 0)
            return int(words[torch.multinomial(cache[tid], 1, generator=gen)])
        if tid not in cache:
            d = torch.cdist(emb[tid][None], W)[0]
            dk, ik = d.topk(a.K, largest=False)
            cache[tid] = (words[ik], 1 - (dk - dk.min()) / (dk.max() - dk.min()).clamp_min(1e-12))
        cand, u = cache[tid]
        return int(cand[torch.multinomial(torch.softmax(a.level * u / 2, 0), 1, generator=gen)])

    def views(split, idx):
        seqs, y, S = H.text_views(a.task, None, tok, split, idx)
        if a.method != "emb":
            seqs = [[sub(t) if j in set(s) else t for j, t in enumerate(ids)] for ids, s in zip(seqs, S)]
        return seqs, y, S

    eta = emb.shape[1] / (a.level * scale)
    nrng = np.random.default_rng(0)

    def forward(seqs, S):
        ids, valid = H.pad(seqs, tok.pad_token_id, dev)
        if a.method != "emb":
            return bert.full(ids, valid)[0]
        we = bert.emb.word_embeddings(ids)
        noise = torch.zeros_like(we)
        for r, pos in enumerate(S):
            if pos:
                z = nrng.standard_normal((len(pos), we.shape[-1]))
                z *= (nrng.gamma(we.shape[-1], 1 / eta, len(pos)) / np.linalg.norm(z, axis=1))[:, None]
                noise[r, pos] = torch.from_numpy(z).to(we)
        x = bert.emb(inputs_embeds=we + noise)
        for layer in bert.layers:
            x = bert.layer(layer, x, valid[:, None, None, :])
        return bert.readout(x[:, 0])

    tr, _held = H.split_train(len(rows_of(a.task, "train")))
    t0 = time.time()
    seqs, y, S = views("train", tr)
    print(f"[e28] {len(seqs)} training texts ({a.method}{a.level:g}) ready in {time.time() - t0:.0f}s", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    rng = random.Random(0)
    torch.manual_seed(0)
    model.train()
    for step in range(steps):
        idx = rng.sample(range(len(seqs)), 32)
        loss = F.cross_entropy(forward([seqs[i] for i in idx], [S[i] for i in idx]), torch.as_tensor(y[idx], device=dev))
        for g in opt.param_groups:
            g["lr"] = H.lr_at(step, steps, 2e-5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"[e28] step {step}/{steps} ce {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    keep = H.text_val_rows(a.task, tok, None)
    vs, vy, vS = views("validation", keep)
    pred = []
    with torch.no_grad():
        for b in range(0, len(vs), 64):
            pred += forward(vs[b:b + 64], vS[b:b + 64]).argmax(1).tolist()
    pred = np.array(pred)
    pos = {r: k for k, r in enumerate(ref["rows"])}
    two = np.array(ref["two_stream"])[[pos[r] for r in keep]]
    assert (np.array(ref["labels"])[[pos[r] for r in keep]] == vy).all()
    m, lo, hi = ci(two == vy, pred == vy)
    res = dict(task=a.task, backbone=a.backbone, method=a.method, level=a.level, trained=True, n=len(vy),
               acc=float((pred == vy).mean()), owte=float((two == vy).mean()), owte_minus=[m, lo, hi],
               eta=eta if a.method == "emb" else None, word_distance=scale)
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/trained_{a.task}_{a.backbone}_{a.method}{a.level:g}.json"
    json.dump(res, open(f, "w"), indent=1)
    print(f"SUMMARY {a.task} {a.backbone} trained {a.method}{a.level:g}: acc {res['acc']:.3f}, OWTE {res['owte']:.3f}, "
          f"OWTE minus it {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {f}", flush=True)


if __name__ == "__main__":
    main()
