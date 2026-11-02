#!/usr/bin/env python
"""E11 (realistic text): the client's PII detector on SST-2. A public NER tagger (dslim/bert-base-NER-uncased: person,
organisation, location, misc) marks entities; every BERT word piece of the SST-2 model's tokenisation that overlaps a
detected entity is private. Writes {split: {index: [private token positions]}} (positions in the SST-2 model's input,
CLS = 0). CPU only.

  python selective/e11_ner_sst2.py --val 872 --train 480 --out ~/data/selective-fhe/e11/ner_private.json
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=872)
    ap.add_argument("--train", type=int, default=480)
    ap.add_argument("--ner", default="dslim/bert-base-NER-uncased")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer, pipeline
    tagger = pipeline("token-classification", model=a.ner, aggregation_strategy="simple", device=-1)
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    out, stats = {}, {}
    for split, n in (("validation", a.val), ("train", a.train)):
        ds = load_dataset("nyu-mll/glue", "sst2", split=split).select(range(n))
        sents = list(ds["sentence"])
        ents = tagger(sents, batch_size=64)
        res, kinds = {}, {}
        for i, (s, es) in enumerate(zip(sents, ents)):
            enc = tok(s, truncation=True, max_length=128, return_offsets_mapping=True)
            spans = [(e["start"], e["end"]) for e in es if e["score"] >= a.min_score]
            for e in es:
                if e["score"] >= a.min_score:
                    kinds[e["entity_group"]] = kinds.get(e["entity_group"], 0) + 1
            pos = [j for j, (c0, c1) in enumerate(enc["offset_mapping"])
                   if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1 in spans)]
            res[str(i)] = pos
        out[split] = res
        k = np.array([len(v) for v in res.values()])
        stats[split] = dict(sentences=n, with_pii=int((k > 0).sum()), mean_private_tokens_if_any=float(k[k > 0].mean()),
                            max_private_tokens=int(k.max()), over_8=int((k > 8).sum()), entity_kinds=kinds)
        print(split, json.dumps(stats[split]), flush=True)
    out["stats"] = stats
    json.dump(out, open(os.path.expanduser(a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
