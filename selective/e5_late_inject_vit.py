#!/usr/bin/env python
"""E5 pilot: late injection on ViT-B/16 (plaintext simulation, no FHE).

The sensitive patches S are replaced by a placeholder (gray patch, or dropped from attention) and the whole image runs
through the first L - k layers in the clear. Before the last k layers the S tokens' hidden states are REPLACED by their
raw embeddings (patch embedding + position), and the last k layers run with full attention over all tokens (the part
that would be encrypted). Read-out: CLS. k = 0 is redaction, k = 12 is the full model.

Same data, placements and sizes as E1 (e1_oneway_vit.py); the two-stream numbers for the same S come from E1's JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_oneway_vit as E


def late_inject(vit, px, S, k_late, placeholder, inject="replace", fusion=None):
    """S: (B,k) patch indices; returns logits of the late-injection model."""
    B, k = S.shape
    dev = px.device
    L = len(vit.layers)
    x_full = vit.emb(px)
    isS = torch.zeros(B, E.NP, dtype=torch.bool, device=dev)
    if k:
        isS.scatter_(1, S, True)
    tokS = S + 1
    if placeholder == "gray":
        m = isS.view(B, 1, E.GRID, 1, E.GRID, 1).expand(B, 3, E.GRID, E.PATCH, E.GRID, E.PATCH).reshape(B, 3, E.GRID * E.PATCH, E.GRID * E.PATCH)
        P = vit.emb(px.masked_fill(m, 0.0))
        maskP = None
    else:
        P = x_full.clone()
        allowed = torch.ones(B, E.NT, dtype=torch.bool, device=dev)
        if k:
            allowed.scatter_(1, tokS, False)
        maskP = allowed[:, None, None, :]
    for i, Lr in enumerate(vit.layers):
        if i == L - k_late:                                   # the hand-over: raw embeddings replace the placeholders
            if k and (inject != "attention" or k_late in fusion.klates):
                idx = tokS[..., None].expand(B, k, vit.D)
                raw = torch.gather(x_full, 1, idx)
                if inject == "add":                           # late-inject-additive: keep the context, add the secret
                    raw = raw + torch.gather(P, 1, idx)
                elif inject == "attention":                   # late-inject-attention: the context reads the secrets
                    h = torch.gather(P, 1, idx)
                    raw = h + fusion(k_late, h, raw)
                P = P.scatter(1, idx, raw)                    # out of place: autograd-safe
            maskP = None
        q, kk, v = vit.qkv(Lr, Lr.layernorm_before(P))
        a = F.scaled_dot_product_attention(q, kk, v, attn_mask=maskP)
        P = P + vit.attn_out(Lr, a)
        P = P + vit.mlp(Lr, Lr.layernorm_after(P))
    if k_late == 0 and k:                                     # redaction never injects
        pass
    return vit.head(vit.lnf(P)[:, 0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="9,16,25,49,100", help="|S| values")
    ap.add_argument("--klate", default="0,1,2,3,4,6,12", help="encrypted (last) layers")
    ap.add_argument("--placements", default="random,box,salient,nonsalient")
    ap.add_argument("--placeholders", default="drop,gray")
    ap.add_argument("--inject", default="replace", choices=["replace", "add", "attention"], help="the hand-over variant")
    ap.add_argument("--fusion", default="", help="attention: the CrossFusion weights (e5b_finetune.py --inject attention)")
    ap.add_argument("--model", default=E.MODEL)
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e5/vit_tinyimagenet.json"))
    ap.add_argument("--class-index", default=os.path.expanduser("~/data/selective-fhe/data/imagenet_class_index.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForImageClassification
    t0 = time.time()
    model = AutoModelForImageClassification.from_pretrained(a.model).to(a.device).eval()
    vit = E.Vit(model)
    fusion = None
    if a.inject == "attention":
        from e5_fusion import CrossFusion
        fusion = CrossFusion.load(a.fusion, a.device)
    px, y, classes, _ = E.load_data(a.n, a.seed, a.class_index)
    px, y, classes = px.to(a.device), y.to(a.device), classes.to(a.device)
    n = px.shape[0]
    full_logits, sal = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch):
            lg, s = vit.full(px[i:i + a.batch])
            full_logits.append(E.restricted(lg, classes)); sal.append(s)
    full_logits, sal = torch.cat(full_logits), torch.cat(sal)
    full_pred = full_logits.argmax(1)
    y_r = (y[:, None] == classes[None, :]).float().argmax(1)
    acc_full = (full_pred == y_r).float().mean().item()
    logp_full = full_logits.log_softmax(1)
    print(f"[data] {n} images; full model top-1 {acc_full:.4f}", flush=True)
    rows = []
    gen = torch.Generator(device=a.device)
    for placement in a.placements.split(","):
        for k in [int(v) for v in a.ks.split(",")]:
            if placement == "box" and int(round(math.sqrt(k))) ** 2 != k:
                continue
            for placeholder in a.placeholders.split(","):
                for kl in [int(v) for v in a.klate.split(",")]:
                    if fusion is not None and kl not in fusion.klates and 0 < kl < 12:
                        continue
                    t1 = time.time()
                    po = []
                    for bi, i in enumerate(range(0, n, a.batch)):
                        gen.manual_seed(a.seed * 1000003 + k * 1009 + bi)          # E1's S sets
                        S = E.choose_S(placement, k, sal[i:i + a.batch], gen)
                        with torch.no_grad():
                            po.append(E.restricted(late_inject(vit, px[i:i + a.batch], S, kl, placeholder, a.inject, fusion), classes))
                    lo = torch.cat(po)
                    pred = lo.argmax(1)
                    kl_ = (logp_full.exp() * (logp_full - lo.log_softmax(1))).sum(1).mean().item()
                    row = dict(placement=placement, placeholder=placeholder, inject=a.inject, k=k, k_late=kl, frac_enc_layers=kl / 12,
                               n=n, acc_full=acc_full, acc=(pred == y_r).float().mean().item(),
                               agree=(pred == full_pred).float().mean().item(), kl=kl_, seconds=time.time() - t1)
                    rows.append(row)
                    print(f"[{placement:10s} {placeholder:4s} |S|={k:3d} k_late={kl:2d}] acc {row['acc']:.3f} agree {row['agree']:.3f} "
                          f"KL {row['kl']:.3f} ({row['seconds']:.0f}s)", flush=True)
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    json.dump(dict(config=vars(a), model=a.model, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] {len(rows)} configs in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
