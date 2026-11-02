#!/usr/bin/env python
"""E8 stage 0: exact float64 two-stream BERT (the one-way rule of e1_oneway_bert.py, post-LN), the oracle for the
encrypted implementation. Mirrors perseus.examples.bert.data.exact_blocks.

Given a sentence (ids padded to T, n real tokens) and the sensitive positions S (content positions 1..n-2):
  P stream: the plain model on the sentence with the S tokens replaced by [MASK] ("mask") or removed from P's keys
            ("drop"); its per-layer K / V on NS = the real tokens that are neither CLS, SEP nor sensitive are what the
            server hands to the encrypted stream.
  E stream: [CLS] + S + [SEP] at their true positions (the client's embeddings), each layer attending to its own keys
            and to P's NS keys of the same layer. The answer is E's CLS through the client's pooler + classifier.
S = none reproduces the full model.

  python selective/e8_reference.py --check    # S = none vs exact_blocks, random S vs the torch E1 construction
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

MASK_ID, SEP_ID = 103, 102


def _ln(x, g, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * g + b


def _attn(q, k, v):
    o = np.zeros_like(q)
    for hh in range(12):
        sl = slice(hh * 64, (hh + 1) * 64)
        s = q[:, sl] @ k[:, sl].T / 8.0
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        o[:, sl] = p @ v[:, sl]
    return o


def streams(ids, n, S):
    """(E_idx, NS): the encrypted tokens (CLS, S, SEPs; by position) and the plaintext keys they read."""
    ids = np.asarray(ids)
    S = sorted(int(s) for s in S)
    enc = np.zeros(n, dtype=bool)
    enc[0] = True
    enc[S] = True
    enc |= ids[:n] == SEP_ID
    return np.where(enc)[0], np.where(~enc)[0]


def two_stream(st, cl, ids, n, S, placeholder="mask", n_layers=12, record=None):
    """record(layer, "ln1" | "ln2" | "gelu", stream "P" | "E", rows): optional tap of the LayerNorm / GELU inputs (model
    units); record(layer, "qk", "E", (queries, keys)): the encrypted stream's attention inputs."""
    from perseus.examples.bert import data as D
    from perseus.examples.gpt2 import weights as W
    ids = np.asarray(ids)
    E_idx, NS = streams(ids, n, S)
    idsP = ids.copy()
    if placeholder == "mask":
        idsP[list(S)] = MASK_ID
        pkeys = np.arange(n)
    elif placeholder == "drop":
        pkeys = np.setdiff1d(np.arange(n), np.asarray(list(S), dtype=np.int64))
    else:
        raise ValueError(placeholder)
    P = D.embed(idsP, cl)[:n]
    E = D.embed(ids, cl)[E_idx]
    t = st.tensor
    E_hidden, P_kv = [], []
    for b in range(n_layers):
        nm = W.block_names(b)
        Wq, bq = t(nm["qkv_w"]).T, t(nm["qkv_b"])
        qP, kP, vP = np.split(P @ Wq + bq, 3, axis=1)
        qE, kE, vE = np.split(E @ Wq + bq, 3, axis=1)
        oP = _attn(qP, kP[pkeys], vP[pkeys])
        if record:
            record(b, "qk", "E", (qE, np.vstack([kE, kP[NS]])))     # the encrypted stream's queries and keys
        oE = _attn(qE, np.vstack([kE, kP[NS]]), np.vstack([vE, vP[NS]]))
        P_kv.append((kP[NS].copy(), vP[NS].copy()))
        out = []
        for tag, X, o in (("P", P, oP), ("E", E, oE)):
            y = X + o @ t(nm["out_w"]).T + t(nm["out_b"])
            if record:
                record(b, "ln1", tag, y)
            x = _ln(y, t(nm["ln1_w"]), t(nm["ln1_b"]))
            u = x @ t(nm["up_w"]).T + t(nm["up_b"])
            if record:
                record(b, "gelu", tag, u)                            # the GELU inputs (E14: range checks)
            y = x + D.gelu(u) @ t(nm["down_w"]).T + t(nm["down_b"])
            if record:
                record(b, "ln2", tag, y)
            out.append(_ln(y, t(nm["ln2_w"]), t(nm["ln2_b"])))
        P, E = out
        E_hidden.append(E.copy())
    return dict(E_idx=E_idx, NS=NS, E_hidden=E_hidden, P_kv=P_kv, logits=D.head(E[0], cl), logits_redact=D.head(P[0], cl))


def span(n, frac, rng):
    """A contiguous span of round(frac * content) content tokens (positions 1..n-2) at a random start (E1's "span")."""
    L = n - 2
    k = int(round(frac * L))
    s0 = int(rng.integers(0, L - k + 1))
    return list(range(1 + s0, 1 + s0 + k))


def _check():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e1_oneway_bert import Bert
    from transformers import AutoModelForSequenceClassification
    from perseus.examples.bert import data as D
    st, cl = D.store(), D.client()
    model = AutoModelForSequenceClassification.from_pretrained(os.environ.get("BERT_HF", D.MODEL)).eval().double()
    bert = Bert(model)
    rng = np.random.default_rng(0)
    for i, (s, lab) in enumerate(D.sentences(6)):
        ids, n = D.encode(s, 128)
        full = D.head(D.exact_blocks(st, D.embed(ids, cl), n)[-1][0], cl)
        r0 = two_stream(st, cl, ids, n, [])
        S = span(n, 0.3, rng)
        r = two_stream(st, cl, ids, n, S)
        ti = torch.as_tensor(ids[:n])[None]
        valid = torch.ones_like(ti, dtype=torch.bool)
        Sm = torch.zeros_like(valid)
        Sm[0, S] = True
        with torch.no_grad():
            _lr, lo = bert.two_stream(ti, valid, Sm, "mask", MASK_ID, SEP_ID)
        print(f"[e8-ref] sentence {i} n={n} |S|={len(S)}: S=none vs full {np.abs(r0['logits'] - full).max():.2e}; "
              f"two-stream vs torch E1 {np.abs(r['logits'] - lo[0].numpy()).max():.2e}; logits {np.round(r['logits'], 3)} "
              f"full {np.round(full, 3)} label {lab}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    if a.check:
        _check()
