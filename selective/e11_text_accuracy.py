#!/usr/bin/env python
"""E11 (realistic text), plaintext: accuracy on SST-2 validation when the private tokens are the detector's entities
(e11_ner_sst2.py): the full model, two-stream (E1's one-way construction, [MASK] placeholder), redaction (the entities
replaced by [MASK], plain model). Sentences without entities run the plain model (no encryption needed).

  python selective/e11_text_accuracy.py --private ~/data/selective-fhe/e11/ner_private.json [--model DIR]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_oneway_bert as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--model", default=E.MODEL)
    ap.add_argument("--n", type=int, default=872)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval().double()
    bert = E.Bert(model)
    ids, valid, y = E.load_data(a.n, tok)
    priv = json.load(open(os.path.expanduser(a.private)))["validation"]
    S = torch.zeros_like(valid)
    for i in range(a.n):
        for p in priv[str(i)]:
            if 0 < p < int(valid[i].sum()) - 1:
                S[i, p] = True
    has = S.any(1)
    with torch.no_grad():
        full, _ = bert.full(ids, valid)
        red, one = bert.two_stream(ids, valid, S, "mask", tok.mask_token_id, tok.sep_token_id)
    pf, pr, po = full.argmax(1), red.argmax(1), one.argmax(1)
    res = {}
    for name, m in (("all", torch.ones_like(has)), ("with_pii", has)):
        k = int(m.sum())
        res[name] = dict(n=k, full=float((pf[m] == y[m]).float().mean()), two_stream=float((po[m] == y[m]).float().mean()),
                         redaction=float((pr[m] == y[m]).float().mean()),
                         agree_two_stream=float((po[m] == pf[m]).float().mean()),
                         agree_redaction=float((pr[m] == pf[m]).float().mean()))
        r = res[name]
        print(f"[e11-text] {os.path.basename(a.model)} {name} ({k} sentences): accuracy full {r['full']:.4f} two-stream "
              f"{r['two_stream']:.4f} redaction {r['redaction']:.4f} | agreement with full: two-stream "
              f"{r['agree_two_stream']:.4f} redaction {r['agree_redaction']:.4f}", flush=True)
    if a.out:
        json.dump(res, open(os.path.expanduser(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
