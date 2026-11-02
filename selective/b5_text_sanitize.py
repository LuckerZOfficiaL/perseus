#!/usr/bin/env python
"""Baseline B5 (accuracy only, plaintext): sanitise the private tokens instead of encrypting them, then run the ordinary
model. Same texts, private positions (detector files) and models as the encrypted text runs. Methods, applied to the
private word pieces only:
  mask      each replaced by [MASK] (our redaction; two-stream's plaintext stream sees the same),
  delete    removed,
  random    each replaced by a uniformly random whole-word vocabulary token,
  dp<eps>   SanText-style metric-DP substitution (Yue et al. 2021): a replacement token sampled with probability
            ~ exp(-eps * d(x, y) / 2), d the Euclidean distance between the model's input embeddings (normalised to
            unit mean distance), over the whole-word vocabulary; eps in --eps.
Reports accuracy, agreement with the full model, and two-stream (plaintext = the encrypted answers, E12-E14) minus each
baseline with a paired bootstrap 95% CI, on the texts that have private tokens.

  python selective/b5_text_sanitize.py --task sst2|hatex|hatexval|enron|drugs|drugsc|mts --model HF_DIR_OR_ID [--eps 1,3,10]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
TASKS = {   # texts (None = SST-2 validation), detector file, tokens
    "sst2": (None, f"{D}/e11/ner_private.json", 128),
    "hatex": (f"{D}/e13/hatexplain.json", f"{D}/e13/hatexplain_private.json", 128),
    "hatexval": (f"{D}/e13/hatexplain_val.json", f"{D}/e13/hatexplain_val_private.json", 128),
    "enron": (f"{D}/e14/enron_t256.json", f"{D}/e14/enron_t256_private.json", 256),
    "drugs": (f"{D}/e15/drugs_t128.json", f"{D}/e15/drugs_t128_private.json", 128),
    "drugsc": (f"{D}/e15/drugs_t128.json", f"{D}/e15/drugs_t128c_private.json", 128),   # E4: condition names too
    "mts": (f"{D}/e15m/mts_t128.json", f"{D}/e15m/mts_t128p_private.json", 128),          # E15m: + procedures
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--eps", default="1,3,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import e1_oneway_bert as E
    from e12_report import ci
    data, privf, T = TASKS[a.task]
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    if data:
        rows = json.load(open(data))["validation"]
    else:
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        rows = [[s, int(l)] for s, l in zip(ds["sentence"], ds["label"])]
    priv = json.load(open(privf))["validation"]
    keep = [i for i in range(len(rows)) if priv.get(str(i))]
    model = AutoModelForSequenceClassification.from_pretrained(os.path.expanduser(a.model)).eval().double()
    bert = E.Bert(model)
    emb = model.bert.embeddings.word_embeddings.weight.detach()
    vocab = tok.get_vocab()
    words = torch.tensor(sorted(i for t, i in vocab.items() if t.isalpha() and not t.startswith("##")))
    W = emb[words]
    scale = torch.cdist(W[torch.randperm(len(W), generator=torch.Generator().manual_seed(0))[:2000]],
                        W[:2000]).mean()
    gen = torch.Generator().manual_seed(a.seed)
    eps_list = [float(e) for e in a.eps.split(",") if e]

    def dp_sub(tid, eps):
        d = torch.cdist(emb[tid][None], W)[0] / scale
        p = torch.softmax(-eps * d / 2, 0)
        return int(words[torch.multinomial(p, 1, generator=gen)])

    methods = ["mask", "delete", "random"] + [f"dp{e:g}" for e in eps_list]
    preds = {m: [] for m in ["full", "two_stream"] + methods}
    y = []
    for b0 in range(0, len(keep), a.batch):
        idx = keep[b0:b0 + a.batch]
        enc = [tok(rows[i][0], truncation=True, max_length=T)["input_ids"] for i in idx]
        S = [[p for p in priv[str(i)] if 0 < p < len(ids) - 1] for i, ids in zip(idx, enc)]
        y += [int(rows[i][1]) for i in idx]
        variants = {"full": enc}
        variants["mask"] = [[tok.mask_token_id if j in set(s) else t for j, t in enumerate(ids)] for ids, s in zip(enc, S)]
        variants["delete"] = [[t for j, t in enumerate(ids) if j not in set(s)] for ids, s in zip(enc, S)]
        variants["random"] = [[int(words[torch.randint(len(words), (1,), generator=gen)]) if j in set(s) else t
                               for j, t in enumerate(ids)] for ids, s in zip(enc, S)]
        for e in eps_list:
            variants[f"dp{e:g}"] = [[dp_sub(t, e) if j in set(s) else t for j, t in enumerate(ids)] for ids, s in zip(enc, S)]
        with torch.no_grad():
            for m, seqs in variants.items():
                L = max(len(s) for s in seqs)
                ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in seqs])
                valid = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs]).bool()
                lg, _ = bert.full(ids, valid)
                preds[m] += lg.argmax(1).tolist()
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
    out = dict(task=a.task, model=a.model, n=len(y), full=float((P["full"] == y).mean()),
               two_stream=float((P["two_stream"] == y).mean()))
    print(f"[b5] {a.task} {os.path.basename(os.path.normpath(a.model))}: {len(y)} texts with private tokens; full "
          f"{out['full']:.3f}, two-stream {out['two_stream']:.3f}", flush=True)
    for m in methods:
        mm, lo, hi = ci(P["two_stream"] == y, P[m] == y)
        out[m] = dict(acc=float((P[m] == y).mean()), agree_full=float((P[m] == P["full"]).mean()), two_minus=[mm, lo, hi])
        print(f"[b5] {m:7s} acc {out[m]['acc']:.3f} agree-with-full {out[m]['agree_full']:.3f} | two-stream minus it "
              f"{mm:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    tag = os.path.basename(os.path.normpath(a.model)).replace("/", "_")
    json.dump(out, open(a.out or f"{D}/e12/b5_{a.task}_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
