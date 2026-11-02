#!/usr/bin/env python
"""E14 (realistic long text, personal data): Enron e-mails (SetFit/enron_spam: the Enron-Spam corpus of Metsis et al.
2006, ham = real Enron employees' mail, spam = spam), spam vs ham. The private tokens are the personal data the client's
detector finds: people, organisations and places (the public NER tagger dslim/bert-base-NER-uncased, as E11) plus
e-mail addresses and phone numbers (patterns over the corpus's space-separated text, e.g. "liz . bellamy @ enron . com",
"( 713 ) 853 - 1234"); every BERT word piece overlapping one is private.

Writes the texts in the SFHE_TEXT_DATA format ({split: [[text, label], ...]}: "validation" = a sample of the test split,
"train" = a sample of the train split), the private positions in the detector format ({split: {index: [positions]}},
CLS = 0, positions < --tokens), and statistics. CPU only.

  python selective/e14_enron.py --tokens 256 --val 1000 --train 6000 --out ~/data/selective-fhe/e14
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC = "~/data/selective-fhe/data/enron_spam"
# the corpus is lower-cased and space-separated around punctuation
EMAIL = re.compile(r"[a-z0-9_\-]+(?: \. [a-z0-9_\-]+)* @ [a-z0-9_\-]+(?: \. [a-z0-9_\-]+)+")
PHONE = re.compile(r"(?:\( \d{3} \) |\d{3} [-.] )\d{3} [-.] \d{4}|\b\d{3} [-.] \d{4}\b(?: x \d+)?")


def detect(texts, tagger, min_score):
    """Per text: [(start, end, kind)] character spans of personal data."""
    ents = tagger(texts, batch_size=32)
    out = []
    for t, es in zip(texts, ents):
        sp = [(int(e["start"]), int(e["end"]), e["entity_group"]) for e in es if e["score"] >= min_score]
        sp += [(m.start(), m.end(), "EMAIL") for m in EMAIL.finditer(t)]
        sp += [(m.start(), m.end(), "PHONE") for m in PHONE.finditer(t)]
        out.append(sp)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--tokens", type=int, default=256, help="T: texts are truncated to T word pieces (CLS, SEP included)")
    ap.add_argument("--val", type=int, default=1000, help="test e-mails (a seeded sample, both classes)")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--ner", default="dslim/bert-base-NER-uncased")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/data/selective-fhe/e14")
    ap.add_argument("--eval-model", default=None, help="only the plaintext accuracy of a fine-tuned model (full, two-stream, "
                                                       "redaction) on the texts and private sets already written")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    if a.eval_model:
        import e13_hatexplain as H13
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        data = json.load(open(os.path.join(out, f"enron_t{a.tokens}.json")))
        priv = json.load(open(os.path.join(out, f"enron_t{a.tokens}_private.json")))
        tag = os.path.basename(os.path.normpath(os.path.expanduser(a.eval_model)))
        H13.accuracy(os.path.expanduser(a.eval_model), tok, data, priv,
                     os.path.join(out, f"enron_t{a.tokens}_accuracy_{tag}.json"), max_len=a.tokens, focus=("ham", 0))
        return
    from transformers import AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")   # the BERT example's tokenizer
    tagger = pipeline("token-classification", model=a.ner, aggregation_strategy="simple", device=-1)
    rng = np.random.default_rng(a.seed)
    data, priv, stats = {}, {}, {}
    for src_split, name, n in (("test", "validation", a.val), ("train", "train", a.train)):
        rows = [json.loads(line) for line in open(os.path.join(os.path.expanduser(a.src), f"{src_split}.jsonl"))]
        rows = [r for r in rows if r["text"].strip()]
        pick = sorted(rng.choice(len(rows), min(n, len(rows)), replace=False).tolist())
        texts, labels = [], []
        for i in pick:
            enc = tok(rows[i]["text"], truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            end = max(c1 for c0, c1 in enc["offset_mapping"])          # the text the model sees (truncated)
            texts.append(rows[i]["text"][:end])
            labels.append(int(rows[i]["label"]))
        spans = detect(texts, tagger, a.min_score)
        pos_all, kinds, frac = {}, Counter(), []
        for j, (t, sp) in enumerate(zip(texts, spans)):
            enc = tok(t, truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            pos = [k for k, (c0, c1) in enumerate(enc["offset_mapping"])
                   if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1, _ in sp)]
            for _s, _e, kd in sp:
                kinds[kd] += 1
            pos_all[str(j)] = pos
            frac.append(len(pos) / max(1, len(enc["input_ids"]) - 2))
        data[name] = [[t, lab] for t, lab in zip(texts, labels)]
        priv[name] = pos_all
        k = np.array([len(v) for v in pos_all.values()])
        L = np.array([len(tok(t)["input_ids"]) for t in texts])
        f = np.array(frac)
        stats[name] = dict(source_split=src_split, emails=len(texts), ham=int(sum(1 - np.array(labels))),
                           spam=int(sum(labels)), tokens_median=float(np.median(L)), tokens_mean=float(L.mean()),
                           at_cap=int((L >= a.tokens).sum()), with_private=int((k > 0).sum()),
                           private_tokens_median=float(np.median(k)), private_tokens_p90=float(np.percentile(k, 90)),
                           private_fraction_median=float(np.median(f)), private_fraction_p90=float(np.percentile(f, 90)),
                           kinds=dict(kinds))
        print(name, json.dumps(stats[name]), flush=True)
    meta = dict(source="SetFit/enron_spam (Enron-Spam, Metsis et al. 2006)", tokens=a.tokens, stats=stats)
    json.dump(dict(data, meta=meta), open(os.path.join(out, f"enron_t{a.tokens}.json"), "w"))
    json.dump(dict(priv, stats=stats), open(os.path.join(out, f"enron_t{a.tokens}_private.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
