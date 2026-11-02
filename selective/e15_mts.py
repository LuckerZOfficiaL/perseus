#!/usr/bin/env python
"""E15, a third task whose answer depends on the private part (text): medical transcriptions (MTSamples, mtsamples.com;
HF harishnair04/mtsamples); the task is the medical specialty, the 10 specialty classes with >= 80 notes that name an
organ system (Surgery, Radiology, Neurosurgery and the document-type classes such as Consult or Discharge Summary are
left out: they cut across specialties). The private tokens are the health data E4's fixed drug detector finds
(medication names, the corpus's condition names, biomedical NER Medication / Disease_disorder) plus, with --procedures,
NER Therapeutic_procedure / Diagnostic_procedure; widened to whole words; every BERT word piece overlapping one is
private. Notes are de-duplicated (a note filed under two of the classes is dropped) and split 70 / 30 by class.

Writes ~/data/selective-fhe/e15m/mts_t{T}.json (SFHE_TEXT_DATA format) and mts_t{T}[p]_private.json (detector format)
and the label names / statistics. CPU only.

  python selective/e15_mts.py [--procedures]
  python selective/e15_mts.py --eval-model MODEL [--procedures]   (plaintext full / two-stream / redaction accuracy)
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e15m")
CLASSES = ["Cardiovascular / Pulmonary", "Orthopedic", "Gastroenterology", "Neurology", "Obstetrics / Gynecology",
           "Urology", "ENT - Otolaryngology", "Hematology - Oncology", "Ophthalmology", "Nephrology"]
PROCEDURES = ("Therapeutic_procedure", "Diagnostic_procedure")


def detector(procedures=True):
    """texts -> per text [(start, end, kind)]: E4's fixed drug detector, plus NER procedures (--procedures)."""
    from datasets import load_dataset
    from transformers import pipeline
    import e15_drugs as E15
    ds = load_dataset("lewtun/drug-reviews")
    lexicon, cond = E15.lexicon_of(ds), E15.phrases(E15.condition_names(ds))
    ner = pipeline("token-classification", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=-1)

    def run(texts):
        spans = E15.detect(texts, lexicon, ner, 0.5, cond)
        if procedures:
            for t, sp, es in zip(texts, spans, ner(texts, batch_size=32)):
                sp += [(*E15.widen(t, int(e["start"]), int(e["end"])), e["entity_group"]) for e in es
                       if e["entity_group"] in PROCEDURES and e["score"] >= 0.5]
        return spans
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--procedures", action="store_true", help="NER procedures are private too")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-model", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"t{a.tokens}{'p' if a.procedures else ''}"
    pv = f"{OUT}/mts_{tag}_private.json"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    if a.eval_model:
        import e13_hatexplain as H13
        data = json.load(open(f"{OUT}/mts_t{a.tokens}.json"))
        name = os.path.basename(os.path.normpath(os.path.expanduser(a.eval_model)))
        H13.accuracy(os.path.expanduser(a.eval_model), tok, data, json.load(open(pv)),
                     f"{OUT}/mts_{tag}_accuracy_{name}.json", max_len=a.tokens, focus=("orthopedic", 1))
        return
    from datasets import load_dataset
    import e15_drugs as E15
    ds = load_dataset("harishnair04/mtsamples")["train"]
    seen = {}
    for spec, tr in zip(ds["medical_specialty"], ds["transcription"]):
        spec = (spec or "").strip()
        if spec in CLASSES and tr and tr.strip():
            seen.setdefault(E15.clean(tr), set()).add(spec)
    items = sorted((t, CLASSES.index(next(iter(s)))) for t, s in seen.items() if len(s) == 1)
    rng = np.random.default_rng(a.seed)
    split = {"validation": [], "train": []}
    for k in range(len(CLASSES)):
        ix = rng.permutation([i for i, (_t, lab) in enumerate(items) if lab == k])
        nv = int(round(a.val_frac * len(ix)))
        split["validation"] += sorted(ix[:nv].tolist())
        split["train"] += sorted(ix[nv:].tolist())
    detect = detector(a.procedures)
    data, priv, stats = {}, {}, {}
    for name, ix in split.items():
        texts, labels = [], []
        for i in sorted(ix):
            t, lab = items[i]
            enc = tok(t, truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            texts.append(t[:max(c1 for _c0, c1 in enc["offset_mapping"])])          # the text the model sees
            labels.append(lab)
        spans = detect(texts)
        pos, kinds = {}, Counter()
        for j, (t, sp) in enumerate(zip(texts, spans)):
            enc = tok(t, truncation=True, max_length=a.tokens, return_offsets_mapping=True)
            pos[str(j)] = [k for k, (c0, c1) in enumerate(enc["offset_mapping"])
                           if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1, _ in sp)]
            kinds.update(kd for _s, _e, kd in sp)
        data[name] = [[t, lab] for t, lab in zip(texts, labels)]
        priv[name] = pos
        k = np.array([len(v) for v in pos.values()])
        stats[name] = dict(notes=len(texts), per_class=dict(Counter(labels)), with_private=int((k > 0).sum()),
                           private_tokens_median=float(np.median(k)), private_tokens_p90=float(np.percentile(k, 90)),
                           over_13=int((k > 13).sum()), over_29=int((k > 29).sum()), kinds=dict(kinds))
        print(name, json.dumps(stats[name]), flush=True)
    f = f"{OUT}/mts_t{a.tokens}.json"
    meta = dict(source="MTSamples (harishnair04/mtsamples)", tokens=a.tokens, labels=CLASSES, stats=stats)
    if os.path.exists(f):                                        # the procedure variant shares the texts
        old = json.load(open(f))
        assert all(old[s] == data[s] for s in ("validation", "train")), "texts differ from the ones written"
    else:
        json.dump(dict(data, meta=meta), open(f, "w"))
    json.dump(dict(priv, stats=stats), open(pv, "w"), indent=1)


if __name__ == "__main__":
    main()
