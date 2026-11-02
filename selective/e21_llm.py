#!/usr/bin/env python
"""E21 (breadth), decoder LLMs on every text task, one result file per (model, task) with the same fields as the
encoder results. Two-stream as E18 (e18_oneway_llm: one [P; E] sequence, true position ids, 4D mask; exact at S = {}
and S = all). Zero-shot prompts (the answer letter's logits at the last prompt token); private spans from each task's
client detector (b5b_typed_sanitize.detect: the same detectors as the encrypted BERT runs), mapped to LLM tokens.

Conditions, on the validation texts with private content:
  full            the whole prompt (no protection),
  two_stream      E = the private tokens + the last prompt token (zero-shot),
  two_stream_q    E = the private tokens + every token after the text (the question; zero-shot),
  delete          the private tokens removed (redaction),
  mask            each private span replaced by [REDACTED],
  typed           each private span replaced by its kind (<PERSON>, <MEDICATION>, <RELIGION>, ...),
  ow_full / ow_two_stream / ow_delete / ow_mask / ow_typed   the same under a one-way LoRA (--lora, trained by --train-lora:
                  rank 16 on q / k / v / o, the answer letter's cross-entropy at r under the two-stream layout, train
                  split's detected private sets).
Tasks: sst2 (sentiment; names), hatex (hate / normal / offensive; identity mentions), enron (ham / spam at 256 BERT
tokens; personal data), drugsc (condition, 10; health data, E4's fixed detector), mts (specialty, 10; health data +
procedures).

  python selective/e21_llm.py --task drugsc --model meta-llama/Llama-3.2-3B-Instruct [--dtype bfloat16]
  python selective/e21_llm.py --task drugsc --model M --train-lora OUT     (then --lora OUT adds the ow_* conditions)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e21/llm")
LETTERS = "ABCDEFGHIJ"


def prompt_of(task, text):
    """(prompt, number of options): the question after the text."""
    if task == "sst2":
        opts, q, what = ["negative", "positive"], "Is the sentiment of this review negative or positive?", "a movie review"
    elif task == "hatex":
        opts, q, what = ["hate speech", "normal", "offensive"], "Is this post hate speech, normal, or offensive?", \
            "a social media post"
    elif task == "enron":
        opts, q, what = ["ham (a legitimate e-mail)", "spam"], "Is this e-mail ham or spam?", "an e-mail"
    elif task == "drugsc":
        import e15_drugs as E15
        opts, q, what = E15.NAMES, "Which condition is the patient taking the medication for?", \
            "a patient's review of a medication"
    else:
        import e15_mts as M
        opts, q, what = M.CLASSES, "Which medical specialty is this transcription from?", "a medical transcription"
    o = "\n".join(f"{LETTERS[k]}. {n}" for k, n in enumerate(opts))
    return f"Here is {what}.\n\nText: {text}\n\n{q} Options:\n{o}\n\nAnswer with the letter only.", len(opts)


def rows(task, split, n, seed):
    """[(text, label, [(start, end, kind)])] of up to n texts with private spans (n = 0: all)."""
    from b5b_typed_sanitize import detect, rows_of
    rs = rows_of(task, split)
    if n and len(rs) > 3 * n:
        idx = np.sort(np.random.default_rng(seed).choice(len(rs), 3 * n, replace=False))
        rs = [rs[i] for i in idx]
    texts = [t for t, _l in rs]
    out = [(t, int(l), sp) for (t, l), sp in zip(rs, detect(task, texts)) if sp]
    return out[:n] if n else out


def encode(tok, task, text, spans):
    """Prompt ids, private token positions S, the positions after the text (the question), the answer letters' ids."""
    p, k = prompt_of(task, text)
    prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    o = prompt.index(text)
    off = enc["offset_mapping"]
    S = [j for j, (c0, c1) in enumerate(off) if c1 > c0 and any(c0 < o + e and c1 > o + s for s, e, _k in spans)]
    Q = [j for j, (c0, _c1) in enumerate(off) if c0 >= o + len(text)]
    letters = [tok.encode(LETTERS[j], add_special_tokens=False)[0] for j in range(k)]
    return enc["input_ids"], S, Q, letters


def substitute(text, spans, how):
    from b5b_typed_sanitize import PLACEHOLDER, replace
    return replace(text, spans, lambda _s, k: "[REDACTED]" if how == "mask" else PLACEHOLDER.get(k, f"<{k.upper()}>"))


@torch.no_grad()
def scores(model, tok, task, text, spans, which):
    import e18_oneway_llm as L
    ids, S, Q, letters = encode(tok, task, text, spans)
    LT = torch.tensor(letters, device=model.device)
    out = {}
    for c in which:
        if c == "full":
            lo = L.run_plain(model, ids)
        elif c == "two_stream":
            lo = L.run_layout(model, ids, S)[0]
        elif c == "two_stream_q":
            lo = L.run_layout(model, ids, sorted(set(S) | set(Q)))[0]
        elif c == "delete":
            lo = L.run_plain(model, [t for i, t in enumerate(ids) if i not in set(S)])
        else:
            ids2, _S, _Q, _l = encode(tok, task, substitute(text, spans, c), [])
            lo = L.run_plain(model, ids2)
        out[c] = int(lo[LT].argmax())
    return out, len(S), len(ids), len(set(S) | {len(ids) - 1})


def evaluate(a):
    import e18_oneway_llm as L
    from e12_report import ci
    base = ("full", "two_stream", "two_stream_q", "delete", "mask", "typed")
    tok, model = L.load(a.model, dtype=getattr(torch, a.dtype))
    data = rows(a.task, "validation", a.limit, a.seed)
    P = {c: [] for c in base}
    y, nS, nT, nE, t0 = [], [], [], [], time.time()
    for j, (text, lab, spans) in enumerate(data):
        s, ns, nt, ne = scores(model, tok, a.task, text, spans, base)
        for c in base:
            P[c].append(s[c])
        y.append(lab); nS.append(ns); nT.append(nt); nE.append(ne)
        if j % 200 == 0:
            print(f"[e21] {a.task} {j}/{len(data)} {time.time() - t0:.0f}s", flush=True)
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora).eval()
        ow = ("full", "two_stream", "delete", "mask", "typed")
        for c in ow:
            P[f"ow_{c}"] = []
        for text, _lab, spans in data:
            s, *_ = scores(model, tok, a.task, text, spans, ow)
            for c in ow:
                P[f"ow_{c}"].append(s[c])
    y = np.array(y)
    acc = {c: float((np.array(v) == y).mean()) for c, v in P.items()}
    agree = {c: float((np.array(v) == np.array(P["full"])).mean()) for c, v in P.items()}
    hide = max(("delete", "mask", "typed"), key=lambda c: acc[c])
    res = dict(task=a.task, model=a.model, dtype=a.dtype, lora=a.lora, n=len(y), acc=acc, agree=agree,
               best_hiding=hide, n_private_median=float(np.median(nS)), n_tokens_median=float(np.median(nT)),
               n_encrypted_median=float(np.median(nE)), labels=y.tolist(), preds=P)
    for c in ("two_stream", "two_stream_q") + (("ow_two_stream",) if a.lora else ()):
        h = f"ow_{hide}" if c.startswith("ow_") and f"ow_{hide}" in P else hide
        f = "ow_full" if c.startswith("ow_") else "full"
        res[f"{c}_minus_hiding"] = ci(np.array(P[c]) == y, np.array(P[h]) == y)
        res[f"{c}_minus_full"] = ci(np.array(P[c]) == y, np.array(P[f]) == y)
    print(f"[e21] {a.task} {a.model.split('/')[-1]}: {len(y)} texts with private content (median {np.median(nS):.0f} "
          f"private of {np.median(nT):.0f} tokens); accuracy " + ", ".join(f"{c} {v:.3f}" for c, v in acc.items()),
          flush=True)
    for k, v in res.items():
        if k.endswith(("_minus_hiding", "_minus_full")):
            print(f"[e21]   {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{a.task}_{a.model.split('/')[-1]}.json", "w"))


def train_lora(a):
    import e18_oneway_llm as L
    from peft import LoraConfig, get_peft_model
    tok, model = L.load(a.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    data = rows(a.task, "train", a.train_n, a.seed)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    t0 = time.time()
    for step in range(a.steps):
        tot = 0.0
        for _ in range(a.batch):
            text, lab, spans = data[int(rng.integers(len(data)))]
            ids, S, _Q, letters = encode(tok, a.task, text, spans)
            lo = L.run_layout(model, ids, S)[0]
            loss = torch.nn.functional.cross_entropy(lo[torch.tensor(letters, device=model.device)][None].float(),
                                                     torch.tensor([lab], device=model.device)) / a.batch
            loss.backward()
            tot += float(loss)
        opt.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(f"[e21] lora {a.task} step {step}/{a.steps} loss {tot:.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(os.path.expanduser(a.train_lora))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("sst2", "hatex", "enron", "drugsc", "mts"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--train-lora", default=None)
    ap.add_argument("--train-n", type=int, default=3000, help="--train-lora: train texts with private content")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="validation texts with private content (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.train_lora:
        train_lora(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
