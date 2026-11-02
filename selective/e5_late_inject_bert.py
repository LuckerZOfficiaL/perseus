#!/usr/bin/env python
"""E5 pilot: late injection on BERT-base SST-2 (plaintext simulation). Sensitive tokens are [MASK] (or dropped from
attention) through the first L - k layers; before the last k layers their hidden states are replaced by their raw
embeddings (word + position + type) and the last k layers run with full attention. Same data and S sets as E1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_oneway_bert as E


def late_inject(bert, ids, valid, S, k_late, placeholder, mask_id, inject="replace", fusion=None):
    B, T = ids.shape
    L = len(bert.layers)
    x_full = bert.emb(input_ids=ids)
    idsP = ids.clone()
    if placeholder == "mask":
        idsP[S] = mask_id
    P = bert.emb(input_ids=idsP)
    key_ok = valid & (~S if placeholder == "drop" else torch.ones_like(S))
    mask = key_ok[:, None, None, :]
    for i, Lr in enumerate(bert.layers):
        if i == L - k_late:
            if inject == "attention" and k_late in fusion.klates:
                keys = S | ~S.any(1)[:, None]                   # a row without secrets attends anything (its delta is dropped)
                P = torch.where(S[..., None], P + fusion(k_late, P, x_full, keys), P)
            elif inject == "add":
                P = torch.where(S[..., None], x_full + P, P)
            else:
                P = torch.where(S[..., None], x_full, P)
            mask = valid[:, None, None, :]
        P = bert.layer(Lr, P, mask)
    return bert.readout(P[:, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=872)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", default="0.1,0.2,0.3,0.5,0.7")
    ap.add_argument("--klate", default="0,1,2,3,4,6,12")
    ap.add_argument("--placements", default="random,span,salient,nonsalient")
    ap.add_argument("--placeholders", default="mask,drop")
    ap.add_argument("--inject", default="replace", choices=["replace", "add", "attention"])
    ap.add_argument("--fusion", default="")
    ap.add_argument("--model", default=E.MODEL)
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e5/bert_sst2.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    bert = E.Bert(model)
    fusion = None
    if a.inject == "attention":
        from e5_fusion import CrossFusion
        fusion = CrossFusion.load(a.fusion, a.device)
    ids, valid, y = E.load_data(a.n, tok)
    ids, valid, y = ids.to(a.device), valid.to(a.device), y.to(a.device)
    n = ids.shape[0]
    mask_id, sep_id = tok.mask_token_id, tok.sep_token_id
    content = valid & (ids != sep_id)
    content[:, 0] = False
    full_logits, sal = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch):
            lg, s = bert.full(ids[i:i + a.batch], valid[i:i + a.batch])
            full_logits.append(lg); sal.append(s)
    full_logits, sal = torch.cat(full_logits), torch.cat(sal)
    full_pred = full_logits.argmax(1)
    acc_full = (full_pred == y).float().mean().item()
    logp_full = full_logits.log_softmax(1)
    print(f"[data] {n} sentences; full model accuracy {acc_full:.4f}", flush=True)
    rows = []
    gen = torch.Generator(device=a.device)
    for placement in a.placements.split(","):
        for frac in [float(v) for v in a.fracs.split(",")]:
            for placeholder in a.placeholders.split(","):
                for kl in [int(v) for v in a.klate.split(",")]:
                    if fusion is not None and kl not in fusion.klates and 0 < kl < 12:
                        continue
                    po, ks = [], []
                    for bi, i in enumerate(range(0, n, a.batch)):
                        gen.manual_seed(a.seed * 1000003 + int(frac * 1000) * 1009 + bi)
                        S = E.choose_S(placement, frac, sal[i:i + a.batch], content[i:i + a.batch], gen)
                        with torch.no_grad():
                            po.append(late_inject(bert, ids[i:i + a.batch], valid[i:i + a.batch], S, kl, placeholder, mask_id, a.inject, fusion))
                        ks.append(S.sum(1))
                    lo, ks = torch.cat(po), torch.cat(ks)
                    pred = lo.argmax(1)
                    kl_ = (logp_full.exp() * (logp_full - lo.log_softmax(1))).sum(1).mean().item()
                    row = dict(placement=placement, placeholder=placeholder, inject=a.inject, frac=frac, mean_k=ks.float().mean().item(),
                               k_late=kl, frac_enc_layers=kl / 12, n=n, acc_full=acc_full, acc=(pred == y).float().mean().item(),
                               agree=(pred == full_pred).float().mean().item(), kl=kl_)
                    rows.append(row)
                    print(f"[{placement:10s} {placeholder:4s} f={frac:.1f} k_late={kl:2d}] acc {row['acc']:.3f} agree {row['agree']:.3f} KL {row['kl']:.3f}", flush=True)
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    json.dump(dict(config=vars(a), model=a.model, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] {len(rows)} configs in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
