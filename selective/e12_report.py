#!/usr/bin/env python
"""E12-E13 paper tables from the result files (~/data/selective-fhe/e12): two-stream on face photos (final config: faces5,
else faces4), late injection on face photos (late4), two-stream on text with real private parts (SST-2 + detected
names: text3 / text24; HateXplain + identity-group mentions: hatex / hatex24). Redaction accuracy for the photos is
recomputed in plaintext on exactly the scored photos (the drop placeholder, as two-stream's plaintext stream).
Speed-ups are against the full model as Perseus runs it (one input per page) and, for text, against an ideally packed
full model (short inputs sharing a page at no extra cost: 14.40 s x tokens / 128). The two-stream minus redaction
accuracy carries a paired bootstrap 95% interval over the inputs.

  python selective/e12_report.py [--tag orig|ft] > table.md
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = os.path.expanduser("~/data/selective-fhe/e12")
FULL_VIT = 36.57
FULL_BERT = 14.40


def load(name):
    f = f"{E}/{name}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def face_redaction(res, weights):
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    import e11_faces_common as F
    det = json.load(open(F.FACES))["detections"]
    vit = EV.Vit(ViTForImageClassification.from_pretrained(os.path.expanduser(weights)).eval())
    for r in res:
        S = torch.tensor([det[r["file"]]["patches"]], dtype=torch.long)
        with torch.no_grad():
            lr, _ = vit.two_stream(EV.preprocess([Image.open(os.path.join(F.ROOT, r["file"]))]), S, "drop")
        r["redact"] = int(lr.argmax())


def ci(a, b, n_boot=10000, seed=0):
    """Paired bootstrap 95% interval of mean(a) - mean(b) (per-input 0/1 arrays)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def faces_table(tag, redaction):
    wv = "google/vit-base-patch16-224" if tag == "orig" else "~/data/selective-fhe/e1b/vit_oneway"
    print(f"### Faces in photos (Imagenette validation; private = detected faces), {tag} weights; full model {FULL_VIT} s/photo\n")
    print("| Faces | Photos | Per page | s/photo | Speed-up | Enc = plaintext | Acc full | Acc two-stream (enc) | Acc redaction | Two-stream - redaction (95% CI) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    allr, tot = [], 0.0
    for b, lab, per in (("s", "1-20 patches", 8), ("m", "21-31", 4), ("l", "32-63", 2)):
        d = load(f"faces5_{b}_{tag}") or load(f"faces4_{b}_{tag}")
        if d is None:
            print(f"| {lab} | (pending) |"); continue
        res = d["results"]
        if redaction:
            face_redaction(res, wv)
        walls = {r["page"]: r["page_wall"] for r in res}
        n, t = len(res), sum(walls.values())
        tot += t
        allr += res
        print(row(lab, res, t, FULL_VIT, per))
    if allr:
        print(row("all", allr, tot, FULL_VIT, ""))


def row(lab, res, t, full, per, extra=""):
    n = len(res)
    acc = lambda k: sum(r.get(k) == r["label"] for r in res) / n
    gap = ""
    if all("redact" in r for r in res):
        m, lo, hi = ci([r["argmax"] == r["label"] for r in res], [r["redact"] == r["label"] for r in res])
        gap = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return (f"| {lab} | {n} | {per} | {t / n:.2f} | {full / (t / n):.1f}x{extra} | {sum(r['argmax'] == r['oracle'] for r in res)}/{n} | "
            f"{acc('full'):.3f} | {acc('argmax'):.3f} | {acc('redact') if all('redact' in r for r in res) else float('nan'):.3f} | {gap} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="orig")
    ap.add_argument("--no-redaction", action="store_true")
    a = ap.parse_args()
    faces_table(a.tag, not a.no_redaction)
    print(f"\n### Late injection on the small-face photos (E5b weights)\n")
    print("| Encrypted layers | Photos | s/photo | Speed-up | Enc = plaintext | Acc full (E5b) | Acc late (enc) |")
    print("|---|---|---|---|---|---|---|")
    for k in (1, 2):
        d = load(f"late4{k}_faces")
        if d is None:
            print(f"| {k} | (pending) |"); continue
        res = d["results"]
        n = len(res)
        w = float(np.median([r["wall"] for r in res]))
        acc = lambda key: sum(r[key] == r["label"] for r in res) / n
        print(f"| {k} | {n} | {w:.2f} | {FULL_VIT / w:.1f}x | {sum(r['argmax'] == r['oracle'] for r in res)}/{n} | "
              f"{acc('full'):.3f} | {acc('argmax'):.3f} |")
    print(f"\n### Text with real private parts, {a.tag} weights; full model {FULL_BERT} s per input (one per page); "
          f"speed-up vs that / vs an ideally packed full model\n")
    print("| Task (private part) | Inputs | Per page | s/input | Speed-up | Enc = plaintext | Acc full | Acc two-stream (enc) | Acc redaction | Two-stream - redaction (95% CI) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    # final config: 24 per page on the multi-page plans (*m); 12 per page kept for reference; HateXplain val = 2nd set
    for task, names in (("SST-2 (detected names)", (("text3_ner", 12), ("text24m_ner", 24))),
                        ("HateXplain test (identity mentions)", (("hatex", 12), ("hatex24m", 24))),
                        ("HateXplain val (identity mentions)", (("hatexval24m", 24),))):
        for nm, G in names:
            d = load(f"{nm}_{a.tag}")
            if d is None:
                print(f"| {task} | (pending) | {G} |"); continue
            res = d["results"]
            walls = {r["page"]: r["page_wall"] for r in res}
            t = sum(walls.values())
            packed = sum(FULL_BERT * r["n"] / 128 for r in res) / len(res)
            print(row(task, res, t, FULL_BERT, G, extra=f" / {packed / (t / len(res)):.1f}x"))

    enron_table(a.tag)


def full_wall(log):
    """Median wall of upstream run_bert's planned SUMMARY lines in a log (s), or None."""
    path = f"{E}/{log}.log"
    if not os.path.exists(path):
        return None
    w = [float(m.group(1)) for m in re.finditer(r"\[bert\] SUMMARY .*? wall=([0-9.]+)s", open(path).read())]
    return float(np.median(w)) if w else None


def enron_table(tag):
    """E14: Enron e-mails at 256 tokens (private = personal data), two-stream 8 per page against upstream's full BERT
    at 256 tokens measured on the same card (e-mails with 256 real tokens)."""
    full = full_wall("enron_full_t256")
    fs = f"{full:.2f} s" if full else "(pending)"
    print(f"\n### Long e-mails (Enron, 256 tokens; private = names, organisations, places, e-mail addresses, phone "
          f"numbers), {tag} weights; full model at 256 tokens {fs} per e-mail\n")
    print("| Task | Inputs | Per page | s/input | Speed-up | Enc = plaintext | Acc full | Acc two-stream (enc) | Acc redaction | Two-stream - redaction (95% CI) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    d = load(f"enron_{tag}")
    if d is None or full is None:
        print(f"| Enron | (pending) | 3 |")
        return
    res = d["results"]
    t = sum({r["page"]: r["page_wall"] for r in res}.values())
    print(row("Enron spam/ham", res, t, full, 3))


if __name__ == "__main__":
    main()
