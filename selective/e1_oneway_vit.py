#!/usr/bin/env python
"""E1: plaintext simulation of one-way attention on ViT-B/16 (google/vit-base-patch16-224).

Given a set S of "sensitive" patch tokens per image, two streams are run:
  P (plaintext) : the standard ViT forward on the redacted image, where the S patches are either
                  replaced by a gray placeholder ("gray") or removed from attention ("drop").
                  In the system this stream runs in the clear on the server.
  E (encrypted) : CLS + the S patches with their real content. At every layer these tokens attend
                  to their own keys/values and to P's keys/values of the non-sensitive patches.
                  In the system this stream runs under FHE and never grows beyond |S|+1 tokens.
Read-outs: full model (what full FHE computes), redaction (P's CLS), one-way (E's CLS).
Sanity: S = all patches and S = no patches both reproduce the full model exactly.

Data: Tiny-ImageNet validation images (zh-plus/tiny-imagenet), labels mapped to ImageNet-1k
indices by WNID; predictions restricted to the mapped classes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "google/vit-base-patch16-224"
GRID, PATCH = 14, 16
NP = GRID * GRID
NT = NP + 1


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="number of validation images")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="0,4,9,16,25,36,49,64,100,144,196", help="|S| values (squares for box)")
    ap.add_argument("--placements", default="random,box,salient,nonsalient")
    ap.add_argument("--placeholders", default="gray,drop")
    ap.add_argument("--out", default=os.path.expanduser("~/data/selective-fhe/e1/vit_tinyimagenet.json"))
    ap.add_argument("--class-index", default=os.path.expanduser("~/data/selective-fhe/data/imagenet_class_index.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=MODEL, help="hub id or a directory saved by e1b_finetune.py")
    return ap.parse_args()


def preprocess(images):
    """google/vit-base-patch16-224 preprocessing by hand: bilinear resize to 224, scale to [0,1], (x-0.5)/0.5."""
    from PIL import Image
    out = np.empty((len(images), 3, GRID * PATCH, GRID * PATCH), dtype=np.float32)
    for i, im in enumerate(images):
        im = im.convert("RGB").resize((GRID * PATCH, GRID * PATCH), Image.BILINEAR)
        out[i] = (np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(out)


def load_data(n, seed, class_index):
    from datasets import load_dataset
    ds = load_dataset("zh-plus/tiny-imagenet", split="valid")
    names = ds.features["label"].names
    wnid2idx = {v[0]: int(k) for k, v in json.load(open(class_index)).items()}
    tiny2in = np.array([wnid2idx.get(w, -1) for w in names])
    labels = np.array(ds["label"])
    ok = np.where(tiny2in[labels] >= 0)[0]
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))
    sub = ds.select(sel.tolist())
    px = preprocess(sub["image"])
    y = torch.as_tensor(tiny2in[labels[sel]])
    classes = torch.as_tensor(sorted(set(int(c) for c in tiny2in if c >= 0)))
    return px, y, classes, int((tiny2in < 0).sum())


class Vit:
    """The HF ViT-B/16 pieces (transformers 5.x layout), driven by hand so both streams share one layer's weights."""

    def __init__(self, model):
        v = model.vit
        self.emb, self.layers, self.lnf, self.head = v.embeddings, v.layers, v.layernorm, model.classifier
        self.H = model.config.num_attention_heads
        self.D = model.config.hidden_size
        self.dh = self.D // self.H

    def qkv(self, L, h):
        a = L.attention
        B, T, _ = h.shape

        def split(x):
            return x.view(B, T, self.H, self.dh).transpose(1, 2)

        return split(a.q_proj(h)), split(a.k_proj(h)), split(a.v_proj(h))

    def attn_out(self, L, a):
        B, H, T, dh = a.shape
        return L.attention.o_proj(a.transpose(1, 2).reshape(B, T, H * dh))

    def mlp(self, L, h):
        return L.mlp.fc2(L.mlp.activation_fn(L.mlp.fc1(h)))

    def full(self, px):
        """Full-model logits and a per-patch saliency (CLS attention averaged over heads and layers)."""
        x = self.emb(px)
        sal = torch.zeros(px.shape[0], NP, device=px.device)
        for L in self.layers:
            h = L.layernorm_before(x)
            q, k, v = self.qkv(L, h)
            s = (q[:, :, :1] @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            sal += s.softmax(-1).mean(1)[:, 0, 1:]
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + self.attn_out(L, a)
            x = x + self.mlp(L, L.layernorm_after(x))
        return self.head(self.lnf(x)[:, 0]), sal / len(self.layers)

    @staticmethod
    def take(t, idx):
        """t: (B,H,T,dh), idx: (B,n) -> (B,H,n,dh)."""
        B, H, _, dh = t.shape
        return torch.gather(t, 2, idx[:, None, :, None].expand(B, H, idx.shape[1], dh))

    def two_stream(self, px, S, placeholder):
        """S: (B,k) patch indices in [0,NP). Returns (redaction logits, one-way logits)."""
        B, k = S.shape
        dev = px.device
        x_full = self.emb(px)
        isS = torch.zeros(B, NP, dtype=torch.bool, device=dev)
        if k:
            isS.scatter_(1, S, True)
        tokS = S + 1
        if placeholder == "gray":
            m = isS.view(B, 1, GRID, 1, GRID, 1).expand(B, 3, GRID, PATCH, GRID, PATCH).reshape(B, 3, GRID * PATCH, GRID * PATCH)
            P = self.emb(px.masked_fill(m, 0.0))  # 0 in normalized space is mid-gray
            maskP = None
        elif placeholder == "drop":
            P = x_full.clone()
            allowed = torch.ones(B, NT, dtype=torch.bool, device=dev)
            if k:
                allowed.scatter_(1, tokS, False)
            maskP = allowed[:, None, None, :]
        else:
            raise ValueError(placeholder)
        NS = torch.arange(1, NT, device=dev).expand(B, NP)[~isS].view(B, NP - k)
        E_idx = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), tokS], 1)
        E = torch.gather(x_full, 1, E_idx[..., None].expand(B, k + 1, self.D))
        for L in self.layers:
            qP, kP, vP = self.qkv(L, L.layernorm_before(P))
            qE, kE, vE = self.qkv(L, L.layernorm_before(E))
            K = torch.cat([kE, self.take(kP, NS)], 2)
            V = torch.cat([vE, self.take(vP, NS)], 2)
            aE = F.scaled_dot_product_attention(qE, K, V)
            aP = F.scaled_dot_product_attention(qP, kP, vP, attn_mask=maskP)
            P = P + self.attn_out(L, aP)
            P = P + self.mlp(L, L.layernorm_after(P))
            E = E + self.attn_out(L, aE)
            E = E + self.mlp(L, L.layernorm_after(E))
        return self.head(self.lnf(P)[:, 0]), self.head(self.lnf(E)[:, 0])


def choose_S(placement, k, sal, gen):
    """Per-image sensitive patch sets, shape (B,k)."""
    B = sal.shape[0]
    dev = sal.device
    if k == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=dev)
    if placement == "random":
        return torch.rand(B, NP, generator=gen, device=dev).argsort(1)[:, :k]
    if placement == "salient":
        return sal.topk(k, dim=1).indices
    if placement == "nonsalient":
        return (-sal).topk(k, dim=1).indices
    if placement == "box":
        s = int(round(math.sqrt(k)))
        if s * s != k:
            raise ValueError(f"box needs a square k, got {k}")
        r0 = torch.randint(0, GRID - s + 1, (B, 1), generator=gen, device=dev)
        c0 = torch.randint(0, GRID - s + 1, (B, 1), generator=gen, device=dev)
        i = torch.arange(s, device=dev)
        rows = (r0 + i[None, :])[:, :, None]
        cols = (c0 + i[None, :])[:, None, :]
        return (rows * GRID + cols).reshape(B, k)
    raise ValueError(placement)


def restricted(logits, classes):
    return logits[:, classes]


def main():
    a = parse()
    torch.manual_seed(a.seed)
    from transformers import AutoModelForImageClassification
    t0 = time.time()
    model = AutoModelForImageClassification.from_pretrained(a.model).to(a.device).eval()
    vit = Vit(model)
    px, y, classes, n_unmapped = load_data(a.n, a.seed, a.class_index)
    px, y, classes = px.to(a.device), y.to(a.device), classes.to(a.device)
    n = px.shape[0]
    print(f"[data] {n} images, {len(classes)} mapped classes ({n_unmapped} Tiny-ImageNet classes unmapped), "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    # Full model once: logits (restricted), predictions, saliency; and a check against the HF forward.
    full_logits, sal = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch):
            lg, s = vit.full(px[i:i + a.batch])
            if i == 0:
                ref = model(pixel_values=px[:a.batch]).logits
                err = (ref - lg).abs().max().item()
                assert err < 1e-3, f"manual ViT forward differs from HF by {err}"
                print(f"[check] manual forward matches HF (max abs diff {err:.2e})", flush=True)
            full_logits.append(restricted(lg, classes))
            sal.append(s)
    full_logits, sal = torch.cat(full_logits), torch.cat(sal)
    full_pred = full_logits.argmax(1)
    y_r = (y[:, None] == classes[None, :]).float().argmax(1)
    acc_full = (full_pred == y_r).float().mean().item()
    print(f"[full] top-1 over {len(classes)} classes = {acc_full:.4f}", flush=True)
    logp_full = full_logits.log_softmax(1)

    ks = [int(k) for k in a.ks.split(",")]
    rows = []
    gen = torch.Generator(device=a.device)
    for placement in a.placements.split(","):
        for k in ks:
            if placement == "box" and int(round(math.sqrt(k))) ** 2 != k:
                continue
            for placeholder in a.placeholders.split(","):
                t1 = time.time()
                pr, po = [], []
                for bi, i in enumerate(range(0, n, a.batch)):
                    gen.manual_seed(a.seed * 1000003 + k * 1009 + bi)  # same S for every placeholder
                    S = choose_S(placement, k, sal[i:i + a.batch], gen)
                    with torch.no_grad():
                        lr, lo = vit.two_stream(px[i:i + a.batch], S, placeholder)
                    pr.append(restricted(lr, classes))
                    po.append(restricted(lo, classes))
                lr, lo = torch.cat(pr), torch.cat(po)
                pred_r, pred_o = lr.argmax(1), lo.argmax(1)
                kl = lambda lq: (logp_full.exp() * (logp_full - lq.log_softmax(1))).sum(1).mean().item()
                row = dict(placement=placement, placeholder=placeholder, k=k, frac_enc=(k + 1) / NT, n=n,
                           acc_full=acc_full,
                           acc_redact=(pred_r == y_r).float().mean().item(),
                           acc_oneway=(pred_o == y_r).float().mean().item(),
                           agree_redact=(pred_r == full_pred).float().mean().item(),
                           agree_oneway=(pred_o == full_pred).float().mean().item(),
                           kl_redact=kl(lr), kl_oneway=kl(lo),
                           max_abs_logit_diff_oneway=(lo - full_logits).abs().max().item(),
                           seconds=time.time() - t1)
                rows.append(row)
                print(f"[{placement:10s} {placeholder:4s} k={k:3d}] acc full {acc_full:.3f} | redact {row['acc_redact']:.3f} "
                      f"| one-way {row['acc_oneway']:.3f} | agree redact {row['agree_redact']:.3f} one-way "
                      f"{row['agree_oneway']:.3f} | KL redact {row['kl_redact']:.3f} one-way {row['kl_oneway']:.3f} "
                      f"| {row['seconds']:.0f}s", flush=True)
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump(dict(config=vars(a), model=a.model, rows=rows), open(a.out, "w"), indent=1)
    print(f"[done] {len(rows)} configs in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
