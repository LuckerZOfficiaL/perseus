#!/usr/bin/env python
"""E15 (a task whose answer depends on the private part, text): patient drug reviews (UCI Drug Review Dataset, Grasser
et al. 2018, drugs.com; HF lewtun/drug-reviews); the task is the condition being treated (the 10 most frequent; Weight
Loss and Obesity merged). The private tokens are the health data the client's detector finds: medication names (a drug
lexicon, every drug name in the corpus and its components, as a drug database lists them) plus the medications and
diseases a public biomedical NER model tags (d4data/biomedical-ner-all: Medication, Disease_disorder), each span widened
to whole words; every BERT word piece overlapping one is private. Descriptions of effects stay public.

Writes the texts in the SFHE_TEXT_DATA format ({split: [[text, label], ...]}: "validation" = a class-balanced sample of
the test split, "train" = of the train split), the private positions in the detector format ({split: {index:
[positions]}}, CLS = 0, positions < --tokens), the label names and statistics. CPU only.

  python selective/e15_drugs.py --tokens 128 --val 100 --train 2000 --out ~/data/selective-fhe/e15
  python selective/e15_drugs.py --eval-model MODEL      (plaintext full / two-stream / redaction accuracy)
  python selective/e15_drugs.py --conditions [...]      (E4's fix: condition names private too, drugs_t128c_private.json)
"""
import argparse
import html
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CLASSES = [("Birth Control",), ("Depression",), ("Pain",), ("Anxiety",), ("Acne",), ("Bipolar Disorde",), ("Insomnia",),
           ("Weight Loss", "Obesity"), ("ADHD",), ("Diabetes, Type 2",)]
NAMES = ["birth control", "depression", "pain", "anxiety", "acne", "bipolar disorder", "insomnia", "weight loss",
         "adhd", "type 2 diabetes"]
NER_KINDS = ("Medication", "Disease_disorder")


def clean(review):
    return re.sub(r"\s+", " ", html.unescape(review)).strip().strip('"').strip()


def widen(text, s, e):
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    while e < len(text) and text[e].isalnum():
        e += 1
    return s, e


def detect(texts, lexicon, ner, min_score, conditions=None):
    """Per text: [(start, end, kind)] character spans of health data."""
    out = []
    for t, es in zip(texts, ner(texts, batch_size=64)):
        sp = [(*widen(t, int(e["start"]), int(e["end"])), e["entity_group"]) for e in es
              if e["entity_group"] in NER_KINDS and e["score"] >= min_score]
        sp += [(m.start(), m.end(), "DRUG") for m in lexicon.finditer(t)]
        sp += [(m.start(), m.end(), "CONDITION") for m in conditions.finditer(t)] if conditions else []
        out.append(sp)
    return out


def drug_names(ds):
    """Every drug name in the corpus and its components (>= 3 letters), lower case, longest first."""
    names = {p.strip().lower() for d in list(ds["train"]["drugName"]) + list(ds["test"]["drugName"]) for p in re.split(r"[/,]", d)}
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def condition_names(ds):
    """Every condition name in the corpus (drugs.com's list), lower case, longest first. The corpus cuts some names
    ("Bipolar Disorde", "ibromyalgia"); both spellings are kept."""
    names = set()
    for c in set(ds["train"]["condition"]) | set(ds["test"]["condition"]):
        if not c or "<" in c:
            continue
        for p in re.split(r"[/,]", c.lower()):
            p = p.strip()
            if len(p) >= 3:
                names |= {p, p + "r"} if p.endswith("e") else {p}
                names |= {"f" + p} if p.startswith("ibro") else set()
    return sorted(names, key=len, reverse=True)


def phrases(names):
    return re.compile(r"(?<![\w])(?:" + "|".join(map(re.escape, names)) + r")(?![\w])", re.IGNORECASE)


def lexicon_of(ds):
    """The drug lexicon: every drug name in the corpus and its components (>= 3 letters), whole words, any case."""
    return phrases(drug_names(ds))


def detector(ner_name="d4data/biomedical-ner-all", min_score=0.5, conditions=False):
    """texts -> per text [(start, end, kind)], the detector used to write the private sets (conditions: E4's fix, the
    corpus's condition names too)."""
    from datasets import load_dataset
    from transformers import pipeline
    ds = load_dataset("lewtun/drug-reviews")
    lexicon, cond = lexicon_of(ds), (phrases(condition_names(ds)) if conditions else None)
    ner = pipeline("token-classification", model=ner_name, aggregation_strategy="simple", device=-1)
    return lambda texts: detect(texts, lexicon, ner, min_score, cond)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=128, help="T: texts are truncated to T word pieces (CLS, SEP included)")
    ap.add_argument("--val", type=int, default=100, help="test reviews per class")
    ap.add_argument("--train", type=int, default=2000, help="train reviews per class")
    ap.add_argument("--ner", default="d4data/biomedical-ner-all")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/data/selective-fhe/e15")
    ap.add_argument("--conditions", action="store_true", help="E4's detector fix: the corpus's condition names are private "
                    "too; writes drugs_t{T}c_private.json for the texts already written (--eval-model: scores with it)")
    ap.add_argument("--eval-model", default=None, help="only the plaintext accuracy of a fine-tuned model (full, two-stream, "
                                                       "redaction) on the texts and private sets already written")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    pv = f"drugs_t{a.tokens}{'c' if a.conditions else ''}_private.json"
    if a.eval_model:
        import e13_hatexplain as H13
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        data = json.load(open(os.path.join(out, f"drugs_t{a.tokens}.json")))
        priv = json.load(open(os.path.join(out, pv)))
        tag = os.path.basename(os.path.normpath(os.path.expanduser(a.eval_model)))
        H13.accuracy(os.path.expanduser(a.eval_model), tok, data, priv, os.path.join(
            out, f"drugs_t{a.tokens}{'c' if a.conditions else ''}_accuracy_{tag}.json"), max_len=a.tokens,
            focus=("birth_control", 0))
        return
    from datasets import load_dataset
    from transformers import AutoTokenizer, pipeline
    ds = load_dataset("lewtun/drug-reviews")
    lexicon = lexicon_of(ds)
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")   # the BERT example's tokenizer
    ner = pipeline("token-classification", model=a.ner, aggregation_strategy="simple", device=-1)
    cond = phrases(condition_names(ds)) if a.conditions else None
    cls_of = {c: k for k, cs in enumerate(CLASSES) for c in cs}
    rng = np.random.default_rng(a.seed)
    data, priv, stats = {}, {}, {}
    for src_split, name, n in (("test", "validation", a.val), ("train", "train", a.train)):
        rows = ds[src_split]
        by = {}
        for i, c in enumerate(rows["condition"]):
            if c in cls_of:
                by.setdefault(cls_of[c], []).append(i)
        pick = sorted(j for k in range(len(CLASSES)) for j in rng.choice(by[k], min(n, len(by[k])), replace=False))
        texts, labels = [], []
        for i in pick:
            t = clean(rows[i]["review"])
            enc = tok(t, truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            texts.append(t[:max(c1 for c0, c1 in enc["offset_mapping"])])     # the text the model sees (truncated)
            labels.append(cls_of[rows[i]["condition"]])
        spans = detect(texts, lexicon, ner, a.min_score, cond)
        pos_all, kinds = {}, Counter()
        for j, (t, sp) in enumerate(zip(texts, spans)):
            enc = tok(t, truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            pos_all[str(j)] = [k for k, (c0, c1) in enumerate(enc["offset_mapping"])
                               if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1, _ in sp)]
            kinds.update(kd for _s, _e, kd in sp)
        data[name] = [[t, lab] for t, lab in zip(texts, labels)]
        priv[name] = pos_all
        k = np.array([len(v) for v in pos_all.values()])
        L = np.array([len(tok(t)["input_ids"]) for t in texts])
        stats[name] = dict(source_split=src_split, reviews=len(texts), per_class=dict(Counter(labels)),
                           tokens_median=float(np.median(L)), at_cap=int((L >= a.tokens).sum()),
                           with_private=int((k > 0).sum()), private_tokens_median=float(np.median(k)),
                           private_tokens_p90=float(np.percentile(k, 90)), over_13=int((k > 13).sum()),
                           kinds=dict(kinds))
        print(name, json.dumps(stats[name]), flush=True)
    meta = dict(source="lewtun/drug-reviews (UCI Drug Review Dataset, Grasser et al. 2018)", tokens=a.tokens,
                labels=NAMES, stats=stats)
    f = os.path.join(out, f"drugs_t{a.tokens}.json")
    if a.conditions:                                             # same texts, another private set
        old = json.load(open(f))
        assert all(old[k] == data[k] for k in ("validation", "train")), "texts differ from the ones written"
    else:
        json.dump(dict(data, meta=meta), open(f, "w"))
    json.dump(dict(priv, stats=stats), open(os.path.join(out, pv), "w"), indent=1)


if __name__ == "__main__":
    main()
