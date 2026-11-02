#!/usr/bin/env python
"""E13 (realistic text, sensitive attributes): HateXplain posts (Twitter / Gab; hate speech / offensive / normal by
annotator majority) under the public HateXplain BERT-base (Hate-speech-CNERG/bert-base-uncased-hatexplain). The private
tokens are the post's mentions of protected identity groups (religion, ethnicity / race / nationality, sexual orientation
and gender identity, gender, disability), found by the client with a fixed identity-term list (the terms of Dixon et al.
2018, "Measuring and Mitigating Unintended Bias in Text Classification", without the age terms, plus plurals and the
common group nouns of HateXplain's target communities): every BERT word piece overlapping a matched word is private.

Writes the texts in the SFHE_TEXT_DATA format ({split: [[text, label], ...]}: "validation" = HateXplain's test split,
"train" = its train split, for range fits and calibration pages; --eval-split val: its val split as a second held-out
set, files with a _val suffix), the private positions in the detector format of
e11_ner_sst2.py ({split: {index: [positions]}}, CLS = 0), and the plaintext accuracy on the test posts of the full model,
two-stream (E1's one-way construction, [MASK] placeholder) and redaction (the mentions replaced by [MASK]). CPU only.

  python selective/e13_hatexplain.py --src ~/data/selective-fhe/data/hatexplain --out ~/data/selective-fhe/e13
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_oneway_bert as E

MODEL = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
LABELS = {"hatespeech": 0, "normal": 1, "offensive": 2}            # the model's id2label: hate speech, normal, offensive
TERMS = {
    "religion": "muslim muslims islam islamic islamist islamists jew jews jewish judaism christian christians christianity "
                "catholic catholics protestant protestants hindu hindus sikh sikhs buddhist buddhists taoist atheist atheists",
    "ethnicity": "black blacks white whites african africans asian asians arab arabs hispanic hispanics latino latinos "
                 "latina latinas latinx mexican mexicans indian indians chinese japanese european europeans caucasian "
                 "caucasians canadian canadians american americans middle eastern immigrant immigrants migrant migrants "
                 "refugee refugees",
    "sexuality": "lesbian lesbians gay gays bisexual bisexuals transgender transgenders trans queer queers lgbt lgbtq "
                 "homosexual homosexuals heterosexual heterosexuals straight nonbinary",
    "gender": "male males female females",
    "disability": "blind deaf paralyzed disabled",
}
WORD = {w: c for c, ws in TERMS.items() for w in ws.split()}


def majority(post):
    c = Counter(a["label"] for a in post["annotators"])
    lab, k = c.most_common(1)[0]
    return LABELS[lab] if k >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/data/selective-fhe/data/hatexplain")
    ap.add_argument("--out", default="~/data/selective-fhe/e13")
    ap.add_argument("--n-e-cap", type=int, default=15, help="report posts with more private tokens than a page lane holds")
    ap.add_argument("--eval-split", default="test", choices=("test", "val"),
                    help="the HateXplain split written as 'validation' (val: a second held-out set; files get a _val suffix)")
    ap.add_argument("--eval-model", default=None, help="only the plaintext accuracy of another model (e.g. the one-way "
                                                       "fine-tune) on the texts and private sets already written")
    a = ap.parse_args()
    src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")   # the BERT example's tokenizer (same vocab)
    sfx = "" if a.eval_split == "test" else "_" + a.eval_split
    if a.eval_model:
        data = json.load(open(os.path.join(out, f"hatexplain{sfx}.json")))
        priv = json.load(open(os.path.join(out, f"hatexplain{sfx}_private.json")))
        tag = os.path.basename(os.path.normpath(os.path.expanduser(a.eval_model)))
        accuracy(os.path.expanduser(a.eval_model), tok, data, priv, os.path.join(out, f"hatexplain{sfx}_accuracy_{tag}.json"))
        return
    posts = json.load(open(os.path.join(src, "dataset.json")))
    div = json.load(open(os.path.join(src, "post_id_divisions.json")))
    data, priv, stats = {}, {}, {}
    for split, name in ((a.eval_split, "validation"), ("train", "train")):
        rows, pos_all, kinds, dropped = [], {}, Counter(), 0
        for pid in div[split]:
            lab = majority(posts[pid])
            if lab is None:
                dropped += 1
                continue
            text = " ".join(posts[pid]["post_tokens"])
            enc = tok(text, truncation=True, max_length=128, return_offsets_mapping=True)
            spans = []
            for m in re.finditer(r"[a-z]+", text.lower()):
                c = WORD.get(m.group())
                if c:
                    spans.append((m.start(), m.end()))
                    kinds[c] += 1
            pos = [j for j, (c0, c1) in enumerate(enc["offset_mapping"])
                   if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1 in spans)]
            pos_all[str(len(rows))] = pos
            rows.append([text, lab])
        data[name], priv[name] = rows, pos_all
        k = np.array([len(v) for v in pos_all.values()])
        stats[name] = dict(source_split=split, posts=len(rows), no_majority_dropped=dropped,
                           labels=dict(Counter(LABELS_INV[r[1]] for r in rows)), with_private=int((k > 0).sum()),
                           mean_private_tokens_if_any=float(k[k > 0].mean()), max_private_tokens=int(k.max()),
                           over_cap=int((k > a.n_e_cap).sum()), mentions=dict(kinds))
        print(name, json.dumps(stats[name]), flush=True)
    json.dump(dict(data, meta=dict(model=MODEL, source="HateXplain (Mathew et al. 2021)", stats=stats)),
              open(os.path.join(out, f"hatexplain{sfx}.json"), "w"))
    json.dump(dict(priv, stats=stats, terms=TERMS), open(os.path.join(out, f"hatexplain{sfx}_private.json"), "w"), indent=1)

    accuracy(MODEL, tok, data, priv, os.path.join(out, f"hatexplain{sfx}_accuracy.json"))


def accuracy(model_name, tok, data, priv, path, max_len=128, focus=("hate", 0)):
    """Plaintext accuracy on the test texts: full model, two-stream, redaction (all texts, those with private tokens,
    those of class focus = (name, label) among them)."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().double()
    bert = E.Bert(model)
    rows = data["validation"]
    enc = tok([t for t, _ in rows], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ids, valid, y = enc["input_ids"], enc["attention_mask"].bool(), torch.as_tensor([lab for _, lab in rows])
    S = torch.zeros_like(valid)
    for i in range(len(rows)):
        for p in priv["validation"][str(i)]:
            if 0 < p < int(valid[i].sum()) - 1:
                S[i, p] = True
    has = S.any(1)
    preds = {"full": [], "two_stream": [], "redaction": []}
    with torch.no_grad():
        for b0 in range(0, len(rows), 64):
            sl = slice(b0, b0 + 64)
            full, _ = bert.full(ids[sl], valid[sl])
            red, one = bert.two_stream(ids[sl], valid[sl], S[sl], "mask", tok.mask_token_id, tok.sep_token_id)
            preds["full"].append(full.argmax(1)); preds["two_stream"].append(one.argmax(1)); preds["redaction"].append(red.argmax(1))
    pf, po, pr = (torch.cat(preds[k]) for k in ("full", "two_stream", "redaction"))
    res = {}
    for name, m in (("all", torch.ones_like(has)), ("with_private", has),
                    (f"with_private_{focus[0]}", has & (y == focus[1]))):
        k = int(m.sum())
        res[name] = dict(n=k, full=float((pf[m] == y[m]).float().mean()), two_stream=float((po[m] == y[m]).float().mean()),
                         redaction=float((pr[m] == y[m]).float().mean()),
                         agree_two_stream=float((po[m] == pf[m]).float().mean()),
                         agree_redaction=float((pr[m] == pf[m]).float().mean()))
        r = res[name]
        print(f"[e13] {os.path.basename(str(model_name))} {name} ({k} posts): accuracy full {r['full']:.4f} two-stream {r['two_stream']:.4f} redaction "
              f"{r['redaction']:.4f} | agreement with full: two-stream {r['agree_two_stream']:.4f} redaction "
              f"{r['agree_redaction']:.4f}", flush=True)
    json.dump(res, open(path, "w"), indent=1)


LABELS_INV = {v: k for k, v in LABELS.items()}

if __name__ == "__main__":
    main()
