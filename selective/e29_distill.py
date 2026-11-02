#!/usr/bin/env python
"""E29: a distilled one-way LoRA for the decoder LLMs (drugsc, mts), to shrink the gap to the full-attention reference.
The encoder recipe's idea on E21's one-way LoRA: loss = cross-entropy of the answer letter at r under the two-stream
layout + KL(teacher || student) over the answer letters, teacher = E24's full-attention LoRA on the complete prompt
(precomputed on the train texts). Same data (E21's 3000 train texts), rank 16 on q / k / v / o, alpha 32, dropout 0.05,
lr 1e-4, batch 8 as E21; 1000 steps instead of 300, KL weight 1, temperature 1. Evaluated like E21's grid (two-stream
layout, E21's validation texts, fp32; 32B bf16).

  python selective/e29_distill.py --train --task drugsc --model M [--seed 0]
  python selective/e29_distill.py --eval --task drugsc --model M --seed 0
  python selective/e29_distill.py --summary
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
OUT = os.environ.get("E29_OUT", f"{D}/e29")                   # E30: the CE-only control writes elsewhere
MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]


def teacher_probs(a, tok, base):
    """E24 full-attention LoRA's answer-letter distribution on the complete prompt of every train text (cached)."""
    import e18_oneway_llm as L
    import e21_llm as E
    from e24_llm import cached_rows
    from peft import PeftModel
    n = a.model.split("/")[-1]
    f = f"{OUT}/teacher/{a.task}_{n}.json"
    if os.path.exists(f):
        return json.load(open(f)), base
    t = PeftModel.from_pretrained(base, f"{D}/e24/lora/{a.task}_{n}_full_s0").eval()
    out = []
    with torch.no_grad():
        for text, _l, spans in cached_rows(a.task, "train", 3000, 0):
            ids, _S, _Q, letters = E.encode(tok, a.task, text, spans)
            out.append(torch.softmax(L.run_plain(t, ids)[torch.tensor(letters, device="cuda")].float(), -1).cpu().tolist())
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(out, open(f, "w"))
    return out, t.unload()


def train(a):
    import e18_oneway_llm as L
    import e21_llm as E
    from e24_llm import cached_rows
    from peft import LoraConfig, get_peft_model
    n = a.model.split("/")[-1]
    t0 = time.time()
    tok, base = L.load(a.model, dtype=torch.bfloat16)
    T, base = teacher_probs(a, tok, base)
    t_teacher = time.time() - t0
    torch.manual_seed(a.seed)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    data = cached_rows(a.task, "train", 3000, 0)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    t1 = time.time()
    for step in range(a.steps):
        tot = 0.0
        for _ in range(8):
            i = int(rng.integers(len(data)))
            text, lab, spans = data[i]
            ids, S, _Q, letters = E.encode(tok, a.task, text, spans)
            lo = L.run_layout(model, ids, S)[0][torch.tensor(letters, device=model.device)].float()
            ls = torch.log_softmax(lo, -1)
            pt = torch.tensor(T[i], device=model.device)
            loss = (torch.nn.functional.nll_loss(ls[None], torch.tensor([lab], device=model.device))
                    + a.kl * (pt * (torch.log(pt.clamp_min(1e-12)) - ls)).sum()) / 8
            loss.backward()
            tot += float(loss.detach())
        opt.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(f"[e29] {a.task} {n} s{a.seed} step {step}/{a.steps} loss {tot:.3f} {time.time() - t1:.0f}s", flush=True)
    d = f"{OUT}/lora/{a.task}_{n}_s{a.seed}"
    model.save_pretrained(d)
    cost = dict(steps=a.steps, batch=8, examples=8 * a.steps, train_s=time.time() - t1, teacher_s=t_teacher,
                gpu=torch.cuda.get_device_name(0), peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    json.dump(cost, open(f"{d}/cost.json", "w"))
    print(f"[e29] SUMMARY trained {a.task} {n} s{a.seed}: {cost}", flush=True)


@torch.no_grad()
def evaluate(a):
    import e18_oneway_llm as L
    import e21_llm as E
    from e24_llm import cached_rows
    from peft import PeftModel
    n = a.model.split("/")[-1]
    d = f"{OUT}/lora/{a.task}_{n}_s{a.seed}"
    tok, model = L.load(a.model, dtype=torch.bfloat16 if "32B" in n else torch.float32)
    model = PeftModel.from_pretrained(model, d).eval()
    val = cached_rows(a.task, "validation", 0, 0)
    y = np.array([l for _t, l, _s in val])
    pred = []
    for text, _l, spans in val:
        ids, S, _Q, letters = E.encode(tok, a.task, text, spans)
        pred.append(int(L.run_layout(model, ids, S)[0][torch.tensor(letters, device="cuda")].argmax()))
    acc = float((np.array(pred) == y).mean())
    json.dump(dict(task=a.task, model=a.model, seed=a.seed, acc=acc, preds=pred, labels=y.tolist(),
                   cost=json.load(open(f"{d}/cost.json"))), open(f"{OUT}/{a.task}_{n}_s{a.seed}.json", "w"))
    print(f"[e29] SUMMARY {a.task} {n} s{a.seed}: two-stream accuracy {acc:.3f}", flush=True)


def summary():
    from e12_report import ci
    out = {}
    for task in ("drugsc", "mts"):
        for m in MODELS:
            n = m.split("/")[-1]
            ref = json.load(open(f"{D}/e21/llm/{task}_{n}.json"))
            y = np.array(ref["labels"])
            ok = lambda p: np.array(p) == y
            runs = sorted(glob.glob(f"{OUT}/{task}_{n}_s*.json"))
            if not runs:
                continue
            R = {json.load(open(f))["seed"]: json.load(open(f)) for f in runs}
            s0 = ok(R[min(R)]["preds"])
            full = json.load(open(f"{D}/e24/{task}_{n}_full_s0.json"))
            hide_t = max(("ow_delete", "ow_mask", "ow_typed"), key=lambda c: ok(ref["preds"][c]).mean())
            tr = {h: json.load(open(f"{D}/e24/{task}_{n}_{h}_s0.json")) for h in ("delete", "mask", "typed")}
            hide_tr = max(tr, key=lambda h: tr[h]["acc"])
            accs = [R[s]["acc"] for s in sorted(R)]
            out[f"{task}/{n}"] = dict(
                seeds=sorted(R), acc=accs, mean=float(np.mean(accs)),
                std=float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                owte_e21=float(ok(ref["preds"]["ow_two_stream"]).mean()),
                full_lora=full["acc"], minus_full_lora=ci(s0, ok(full["preds"])),
                hiding_test=dict(name=hide_t[3:], acc=float(ok(ref["preds"][hide_t]).mean()),
                                 minus=ci(s0, ok(ref["preds"][hide_t]))),
                hiding_trained=dict(name=hide_tr, acc=tr[hide_tr]["acc"], minus=ci(s0, ok(tr[hide_tr]["preds"]))),
                minus_owte_e21=ci(s0, ok(ref["preds"]["ow_two_stream"])),
                cost={s: R[s]["cost"] for s in sorted(R)})
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
    for k, r in out.items():
        f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
        print(f"{k:32s} distilled {r['acc']} (E21 OWTE {r['owte_e21']:.3f}) | -full LoRA {r['full_lora']:.3f}: "
              f"{f(r['minus_full_lora'])} | -test hiding {f(r['hiding_test']['minus'])} | -trained hiding "
              f"{f(r['hiding_trained']['minus'])} | train {r['cost'][r['seeds'][0]]['train_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("drugsc", "mts"))
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.train:
        train(a)
    elif a.eval:
        evaluate(a)
    else:
        summary()


if __name__ == "__main__":
    main()
