#!/usr/bin/env python
"""E21: the model x task grid from the result files, one table per task, the same columns for every model:
full model, two-stream without one-way training, two-stream after it (fine-tune / LoRA), the best hiding method (over
every baseline and both weight sets, the most generous choice for hiding), two-stream (one-way) minus best hiding and
minus full with paired 95% CIs (the CI of the comparison made on the same texts / photos), and the encrypted check
where Perseus runs the model. Missing cells print as "-" (not run yet or not applicable).

  python selective/e21_table.py [--md OUT.md]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
E = f"{D}/e21"
TEXT = ("drugsc", "mts", "sst2", "hatex", "enron")
IMAGE = ("ages", "faces")
NAMES = {"drugsc": "Drug reviews (condition; health data)", "mts": "Medical transcriptions (specialty; health data)",
         "sst2": "SST-2 (sentiment; names)", "hatex": "HateXplain (hate speech; identity mentions)",
         "enron": "Enron e-mails (spam; personal data, 256 tokens)", "ages": "IMDB-WIKI (age group; face)",
         "faces": "Imagenette (object; faces)"}
LLMS = ("Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
        "Qwen2.5-32B-Instruct")                          # Gemma-3-12B left out (user, 2026-10-31); its files stay in e21/llm
ENCODERS = ("bert-base", "bert-large")
VITS = ("vit-base", "vit-large", "deit-base", "deit-small")


def load(f):
    try:
        return json.load(open(f))
    except (OSError, ValueError):
        return None


def ci(a, b, n_boot=10000, seed=0):
    from e12_report import ci as c
    return c(a, b, n_boot, seed)


def cell(v):
    return "-" if v is None else f"{v:.3f}"


def diff(c):
    if c is None:
        return "-"
    return f"{100 * c[0]:+.1f}" + ("" if c[1] is None else f" [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")


def block(full, ours, hide, hname, d_hide, d_full):
    return dict(full=full, ours=ours, hide=hide, hide_name=hname, d_hide=d_hide, d_full=d_full)


def encoder_row(task, m):
    """Per weight set (ordinary, one-way): e21_text.py accuracies and B5 / B5b baselines on the same weights (each with
    the paired CI of that weights' two-stream minus it)."""
    out = {}
    for key, wt in (("plain", m), ("ow", f"{m}_oneway")):
        r = load(f"{E}/text/{task}_{wt}_acc.json")
        if r is None:
            out[key] = None
            continue
        w = r["with_private"]
        best = None
        for f in (f"{E}/text/{task}_{wt}_b5.json", f"{E}/text/{task}_{wt}_b5b.json"):
            for k, v in (load(f) or {}).items():
                if isinstance(v, dict) and "acc" in v and (best is None or v["acc"] > best[1]):
                    best = (k, v["acc"], v.get("two_minus"))
        out[key] = block(w["full"], w["two_stream"], best and best[1], best and best[0], best and best[2],
                         (w["two_stream"] - w["full"], None, None))
    return out if any(out.values()) else None


def llm_row(task, m):
    """Zero-shot block: ours = the private tokens + the question encrypted (two_stream_q); one-way block: under the
    LoRA, ours = the private tokens + the last token. Hiding: delete / [REDACTED] / typed under the same weights."""
    r = load(f"{E}/llm/{task}_{m}.json")
    if r is None:
        return None
    a, y = r["acc"], np.array(r["labels"])
    P = {k: np.array(v) for k, v in r["preds"].items()}
    out = {}
    for key, pre, ours in (("plain", "", "two_stream_q"), ("ow", "ow_", "ow_two_stream")):
        if ours not in a:
            out[key] = None
            continue
        hn = max((f"{pre}{h}" for h in ("delete", "mask", "typed") if f"{pre}{h}" in a), key=lambda k: a[k])
        out[key] = block(a[f"{pre}full"], a[ours], a[hn], hn.replace("ow_", ""), ci(P[ours] == y, P[hn] == y),
                         ci(P[ours] == y, P[f"{pre}full"] == y))
    out["ts_r"] = a["two_stream"]
    return out


def vit_row(task, m):
    out = {}
    for key, wt in (("plain", m), ("ow", f"{m}_oneway")):
        r = load(f"{E}/vision/{task}_{wt}.json")
        if r is None:
            out[key] = None
            continue
        a = r["acc"]
        hn = max(("redaction", "black", "gray", "blur", "pixelate", "inpaint", "genface"), key=lambda k: a[k])
        out[key] = block(a["full"], a["two_stream"], a[hn], hn, r["two_minus"][hn], r["two_minus"]["full"])
    return out if any(out.values()) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=f"{E}/grid.md")
    a = ap.parse_args()
    lines = ["Two blocks per model: without one-way training (encoders / ViTs: the ordinary fine-tune; LLMs: zero-shot, "
             "the question encrypted with the private words) and with it (one-way fine-tune; LLMs: one-way LoRA, only the "
             "private words + last token encrypted). Best hiding = the best baseline on the same weights. Differences in "
             "points with paired 95% CIs.", ""]
    for task in TEXT + IMAGE:
        rows = ([(m, encoder_row(task, m)) for m in ENCODERS] + [(m, llm_row(task, m)) for m in LLMS]) if task in TEXT \
            else [(m, vit_row(task, m)) for m in VITS]
        lines += [f"### {NAMES[task]}", "",
                  "| Model | Full | Ours | Best hiding | Ours - hiding | Ours - full | One-way: full | Ours | Best hiding | "
                  "Ours - hiding | Ours - full |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in rows:
            cells = []
            for key in ("plain", "ow"):
                b = r and r.get(key)
                cells += ["-"] * 5 if not b else [cell(b["full"]), cell(b["ours"]),
                                                  f"{cell(b['hide'])} ({b['hide_name']})", diff(b["d_hide"]),
                                                  diff(b["d_full"])]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    open(a.md, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
