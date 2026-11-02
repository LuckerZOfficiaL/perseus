#!/usr/bin/env python
"""E1: plaintext simulation of one-way attention on BERT-base SST-2 (textattack/bert-base-uncased-SST-2).

Same construction as e1_oneway_vit.py. The two streams are laid side by side in one 2T-token
sequence and separated by a boolean attention mask, so |S| can differ per sentence:
  P (plaintext) : tokens 0..T-1, the sentence with the S tokens replaced by [MASK] ("mask") or
                  removed from attention ("drop"). Runs in the clear on the server.
  E (encrypted) : tokens T..2T-1, of which only CLS, SEP and the S tokens are real; they attend to
                  each other and to P's non-sensitive tokens. Runs under FHE, |S|+2 tokens.
Read-outs: full model, redaction (P's CLS through the pooler), one-way (E's CLS through the pooler).
Sanity: S = all content tokens and S = none both reproduce the full model exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

MODEL = "textattack/bert-base-uncased-SST-2"


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=872, help="validation sentences (872 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.5,0.7,1.0", help="fraction of content tokens in S")
    ap.add_argument("--placements", default="random,span,salient,nonsalient")
    ap.add_argument("--placeholders", default="mask,drop")
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e1/bert_sst2.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=MODEL, help="hub id or a directory saved by e1b_finetune.py")
    return ap.parse_args()


def load_data(n, tok):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    sents, labels = ds["sentence"][:n], ds["label"][:n]
    enc = tok(sents, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"].bool(), torch.as_tensor(labels)


class Bert:
    def __init__(self, model):
        b = model.bert
        self.emb, self.layers, self.pooler, self.head = b.embeddings, b.encoder.layer, b.pooler, model.classifier
        self.H = model.config.num_attention_heads
        self.D = model.config.hidden_size
        self.dh = self.D // self.H

    def qkv(self, L, h):
        s = L.attention.self
        B, T, _ = h.shape

        def split(x):
            return x.view(B, T, self.H, self.dh).transpose(1, 2)

        return split(s.query(h)), split(s.key(h)), split(s.value(h))

    def layer(self, L, x, mask, cls_sal=None):
        q, k, v = self.qkv(L, x)
        if cls_sal is not None:
            s = (q[:, :, :1] @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            s = s.masked_fill(~mask[:, :, :1, :], float("-inf"))
            cls_sal += s.softmax(-1).mean(1)[:, 0]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, H, T, dh = a.shape
        a = a.transpose(1, 2).reshape(B, T, H * dh)
        x = L.attention.output.LayerNorm(L.attention.output.dense(a) + x)
        return L.output.LayerNorm(L.output.dense(L.intermediate.intermediate_act_fn(L.intermediate.dense(x))) + x)

    def readout(self, h_cls):
        return self.head(self.pooler.activation(self.pooler.dense(h_cls)))

    def full(self, ids, valid):
        x = self.emb(input_ids=ids)
        mask = valid[:, None, None, :]
        sal = torch.zeros(ids.shape, device=ids.device)
        for L in self.layers:
            x = self.layer(L, x, mask, sal)
        return self.readout(x[:, 0]), sal / len(self.layers)

    def two_stream(self, ids, valid, S, placeholder, mask_id, sep_id):
        B, T = ids.shape
        dev = ids.device
        idsP = ids.clone()
        if placeholder == "mask":
            idsP[S] = mask_id
        elif placeholder != "drop":
            raise ValueError(placeholder)
        x = torch.cat([self.emb(input_ids=idsP), self.emb(input_ids=ids)], 1)
        is_cls = torch.zeros(B, T, dtype=torch.bool, device=dev)
        is_cls[:, 0] = True
        enc = is_cls | (ids == sep_id) | S                 # the real encrypted tokens
        key_pp = valid & (~S if placeholder == "drop" else torch.ones_like(S))
        mask = torch.zeros(B, 2 * T, 2 * T, dtype=torch.bool, device=dev)
        mask[:, :T, :T] = key_pp[:, None, :]              # P rows read P keys
        mask[:, T:, T:] = (valid & enc)[:, None, :]       # E rows read the real E keys
        mask[:, T:, :T] = (valid & ~enc)[:, None, :]      # ... and P's non-sensitive keys
        mask = mask[:, None]
        for L in self.layers:
            x = self.layer(L, x, mask)
        return self.readout(x[:, 0]), self.readout(x[:, T])


def topk_var(score, content, k):
    """Per-row top-k over `content` positions with a per-row k. Returns a bool (B,T)."""
    s = score.masked_fill(~content, float("-inf"))
    rank = s.argsort(1, descending=True).argsort(1)
    return (rank < k[:, None]) & content


def choose_S(placement, frac, sal, content, gen):
    B, T = content.shape
    dev = content.device
    L = content.sum(1)
    k = torch.round(frac * L.float()).long()
    if placement == "random":
        return topk_var(torch.rand(B, T, generator=gen, device=dev), content, k)
    if placement == "salient":
        return topk_var(sal, content, k)
    if placement == "nonsalient":
        return topk_var(-sal, content, k)
    if placement == "span":  # content tokens sit at positions 1..L
        start = (torch.rand(B, generator=gen, device=dev) * (L - k + 1).float()).floor().long()
        pos = torch.arange(T, device=dev)[None, :]
        return content & (pos >= 1 + start[:, None]) & (pos < 1 + (start + k)[:, None])
    raise ValueError(placement)


def main():
    a = parse()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(a.device).eval()
    bert = Bert(model)
    ids, valid, y = load_data(a.n, tok)
    ids, valid, y = ids.to(a.device), valid.to(a.device), y.to(a.device)
    n, T = ids.shape
    mask_id, sep_id = tok.mask_token_id, tok.sep_token_id
    content = valid & (ids != sep_id)
    content[:, 0] = False
    print(f"[data] {n} sentences, padded length {T}, mean content tokens {content.sum(1).float().mean():.1f}", flush=True)

    full_logits, sal = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch):
            lg, s = bert.full(ids[i:i + a.batch], valid[i:i + a.batch])
            if i == 0:
                ref = model(input_ids=ids[:a.batch], attention_mask=valid[:a.batch].long()).logits
                err = (ref - lg).abs().max().item()
                assert err < 1e-3, f"manual BERT forward differs from HF by {err}"
                print(f"[check] manual forward matches HF (max abs diff {err:.2e})", flush=True)
            full_logits.append(lg)
            sal.append(s)
    full_logits, sal = torch.cat(full_logits), torch.cat(sal)
    full_pred = full_logits.argmax(1)
    acc_full = (full_pred == y).float().mean().item()
    print(f"[full] SST-2 validation accuracy = {acc_full:.4f}", flush=True)
    logp_full = full_logits.log_softmax(1)

    rows = []
    gen = torch.Generator(device=a.device)
    for placement in a.placements.split(","):
        for frac in [float(f) for f in a.fracs.split(",")]:
            for placeholder in a.placeholders.split(","):
                t1 = time.time()
                pr, po, ks = [], [], []
                for bi, i in enumerate(range(0, n, a.batch)):
                    gen.manual_seed(a.seed * 1000003 + int(frac * 1000) * 1009 + bi)
                    S = choose_S(placement, frac, sal[i:i + a.batch], content[i:i + a.batch], gen)
                    with torch.no_grad():
                        lr, lo = bert.two_stream(ids[i:i + a.batch], valid[i:i + a.batch], S, placeholder, mask_id, sep_id)
                    pr.append(lr)
                    po.append(lo)
                    ks.append(S.sum(1))
                lr, lo, ks = torch.cat(pr), torch.cat(po), torch.cat(ks)
                pred_r, pred_o = lr.argmax(1), lo.argmax(1)
                kl = lambda lq: (logp_full.exp() * (logp_full - lq.log_softmax(1))).sum(1).mean().item()
                frac_enc = ((ks + 2).float() / (valid.sum(1).float())).mean().item()
                row = dict(placement=placement, placeholder=placeholder, frac=frac, mean_k=ks.float().mean().item(),
                           frac_enc=frac_enc, n=n, acc_full=acc_full,
                           acc_redact=(pred_r == y).float().mean().item(),
                           acc_oneway=(pred_o == y).float().mean().item(),
                           agree_redact=(pred_r == full_pred).float().mean().item(),
                           agree_oneway=(pred_o == full_pred).float().mean().item(),
                           kl_redact=kl(lr), kl_oneway=kl(lo),
                           max_abs_logit_diff_oneway=(lo - full_logits).abs().max().item(),
                           seconds=time.time() - t1)
                rows.append(row)
                print(f"[{placement:10s} {placeholder:4s} f={frac:.1f} k={row['mean_k']:4.1f}] acc full {acc_full:.3f} | redact "
                      f"{row['acc_redact']:.3f} | one-way {row['acc_oneway']:.3f} | agree redact {row['agree_redact']:.3f} "
                      f"one-way {row['agree_oneway']:.3f} | KL redact {row['kl_redact']:.3f} one-way {row['kl_oneway']:.3f}",
                      flush=True)
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump(dict(config=vars(a), model=a.model, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] {len(rows)} configs in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
