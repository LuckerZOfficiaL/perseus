#!/usr/bin/env python
"""E24 (blind-review follow-ups), decoder LLMs on the tasks where the private part decides the answer (drugsc, mts).
Same prompts, detectors, validation texts and LoRA recipe as E21 (e21_llm.py: rank 16 on q / k / v / o, alpha 32,
dropout 0.05, lr 1e-4, batch 8, 300 steps, the answer letter's cross-entropy, train split's 3000 texts with private
content); only what the LoRA is trained and evaluated on changes. Regimes (LoRA trained AND evaluated on):
  full      the whole prompt, ordinary attention            (fair reference for what the one-way mask costs)
  delete    the private tokens removed                      (hiding-aware baselines; delete = the public-only model)
  mask      each private span replaced by [REDACTED]
  typed     each private span replaced by its kind
  private   only the private words (joined by spaces) + the question    (necessity ablation)
  oneway    the two-stream layout (E = private tokens + last token)     (seeds 1, 2; seed 0 is E21's adapter)
Late fusion (client side): w * log p(public-only = delete LoRA) + (1 - w) * log p(private-only LoRA) over the answer
letters, w tuned on train texts (drugsc: train texts not used for the LoRAs; mts: the train split is the LoRAs' own
data, all 1134 notes) and also reported at w = 0.5.

  python selective/e24_llm.py --cache --task drugsc                              (CPU: detector rows, once per task)
  python selective/e24_llm.py --train full --task drugsc --model M [--seed 0]
  python selective/e24_llm.py --eval --task drugsc --model M                     (every finished adapter not yet scored)
  python selective/e24_llm.py --summary
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
OUT = os.environ.get("E24_OUT", f"{D}/e24")             # E31: the 1000-step adapters write elsewhere
E21 = f"{D}/e21/llm"
MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
          "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
EVAL_COND = {"full": "full", "delete": "delete", "mask": "mask", "typed": "typed", "private": "private",
             "oneway": "two_stream"}


def cached_rows(task, split, n, seed):
    """e21_llm.rows, computed once (the CPU detectors are slow) and reused by every run."""
    f = f"{D}/e24/cache/{task}_{split}_n{n}_s{seed}.json"
    if not os.path.exists(f):
        import e21_llm as E
        os.makedirs(os.path.dirname(f), exist_ok=True)
        json.dump(E.rows(task, split, n, seed), open(f, "w"))
    return [(t, int(l), [tuple(s) for s in sp]) for t, l, sp in json.load(open(f))]


def fusion_rows(task):
    """Train texts to tune the fusion weight: drugsc, ones not among the LoRAs' 3000; mts, the whole (small) train split."""
    used = {t for t, _l, _s in cached_rows(task, "train", 3000, 0)}
    if task == "mts":
        return cached_rows(task, "train", 3000, 0)
    return [r for r in cached_rows(task, "train", 600, 1) if r[0] not in used][:500]


def letter_logits(model, tok, task, text, spans, cond):
    """The answer letters' logits at the readout under one condition (a graph is kept if grad is enabled)."""
    import e18_oneway_llm as L
    import e21_llm as E
    ids, S, _Q, letters = E.encode(tok, task, text, spans)
    if cond == "full":
        lo = L.run_plain(model, ids)
    elif cond == "two_stream":
        lo = L.run_layout(model, ids, S)[0]
    elif cond == "delete":
        lo = L.run_plain(model, [t for i, t in enumerate(ids) if i not in set(S)])
    elif cond in ("mask", "typed"):
        lo = L.run_plain(model, E.encode(tok, task, E.substitute(text, spans, cond), [])[0])
    elif cond == "private":
        lo = L.run_plain(model, E.encode(tok, task, " ".join(text[s:e] for s, e, _k in spans), [])[0])
    else:
        raise ValueError(cond)
    return lo[torch.tensor(letters, device=model.device)]


def adapter_dir(task, model, regime, seed):
    return f"{OUT}/lora/{task}_{model.split('/')[-1]}_{regime}_s{seed}"


def train(a):
    import e18_oneway_llm as L
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    tok, model = L.load(a.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    data = cached_rows(a.task, "train", 3000, 0)                   # E21's training texts (seed 0 subset)
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    cond, t0 = EVAL_COND[a.train], time.time()
    for step in range(a.steps):
        tot = 0.0
        for _ in range(8):
            text, lab, spans = data[int(rng.integers(len(data)))]
            lo = letter_logits(model, tok, a.task, text, spans, cond)
            loss = torch.nn.functional.cross_entropy(lo[None].float(), torch.tensor([lab], device=model.device)) / 8
            loss.backward()
            tot += float(loss)
        opt.step()
        opt.zero_grad()
        if step % 50 == 0:
            print(f"[e24] lora {a.train} {a.task} step {step}/{a.steps} loss {tot:.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(adapter_dir(a.task, a.model, a.train, a.seed))
    print(f"[e24] SUMMARY trained {a.train} s{a.seed} {a.task} {a.model} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def evaluate(a):
    import e18_oneway_llm as L
    from peft import PeftModel
    n = a.model.split("/")[-1]
    dtype = torch.bfloat16 if "32B" in n else torch.float32                 # as E21's grid evaluations
    todo = []
    for d in sorted(glob.glob(f"{OUT}/lora/{a.task}_{n}_*_s*")):
        if not os.path.exists(f"{d}/adapter_config.json"):
            continue
        tag = d.split(f"{a.task}_{n}_")[1]
        if not os.path.exists(f"{OUT}/{a.task}_{n}_{tag}.json"):
            todo.append((tag, d))
    fus_needed = all(os.path.exists(f"{OUT}/lora/{a.task}_{n}_{r}_s0/adapter_config.json") for r in ("delete", "private")) \
        and not os.path.exists(f"{OUT}/{a.task}_{n}_fusiontrain.json")
    if not todo and not fus_needed:
        print(f"[e24] SUMMARY nothing to evaluate for {a.task} {n}")
        return
    tok, base = L.load(a.model, dtype=dtype)
    val = cached_rows(a.task, "validation", 0, 0)
    ref = json.load(open(f"{E21}/{a.task}_{n}.json"))
    y = np.array([l for _t, l, _s in val])
    assert len(y) == ref["n"] and (y == np.array(ref["labels"])).all(), "validation rows differ from E21's"
    model = None
    for k, (tag, d) in enumerate(todo + ([("fusion", None)] if fus_needed else [])):
        if tag == "fusion":
            regimes = [("delete_s0", f"{OUT}/lora/{a.task}_{n}_delete_s0"), ("private_s0", f"{OUT}/lora/{a.task}_{n}_private_s0")]
        else:
            regimes = [(tag, d)]
        logits = {}
        for rt, rd in regimes:
            if model is None:
                model = PeftModel.from_pretrained(base, rd, adapter_name=rt).eval()
            elif rt not in model.peft_config:
                model.load_adapter(rd, adapter_name=rt)
            model.set_adapter(rt)
            if tag == "fusion":
                rows = fusion_rows(a.task)
                logits[rt] = [letter_logits(model, tok, a.task, t, s, EVAL_COND[rt.split("_s")[0]]).float().cpu().tolist()
                              for t, _l, s in rows]
            else:
                t0 = time.time()
                lg = [letter_logits(model, tok, a.task, t, s, EVAL_COND[tag.split("_s")[0]]).float().cpu().tolist()
                      for t, _l, s in val]
                pred = np.array([int(np.argmax(v)) for v in lg])
                acc = float((pred == y).mean())
                json.dump(dict(task=a.task, model=a.model, regime=tag, cond=EVAL_COND[tag.split("_s")[0]],
                               dtype=str(dtype), n=len(y), acc=acc, preds=pred.tolist(), logits=lg, labels=y.tolist()),
                          open(f"{OUT}/{a.task}_{n}_{tag}.json", "w"))
                print(f"[e24] SUMMARY {a.task} {n} {tag}: accuracy {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if tag == "fusion":
            rows = fusion_rows(a.task)
            json.dump(dict(task=a.task, model=a.model, labels=[l for _t, l, _s in rows], logits=logits),
                      open(f"{OUT}/{a.task}_{n}_fusiontrain.json", "w"))
            print(f"[e24] SUMMARY {a.task} {n} fusion train logits on {len(rows)} texts", flush=True)


def summary():
    from e12_report import ci
    def ls(v):                                                     # log-softmax over the answer letters
        v = np.asarray(v, float)
        m = v.max(1, keepdims=True)
        return v - m - np.log(np.exp(v - m).sum(1, keepdims=True))
    out = {}
    for task in ("drugsc", "mts"):
        for m in MODELS:
            n = m.split("/")[-1]
            ref = json.load(open(f"{E21}/{task}_{n}.json"))
            y = np.array(ref["labels"])
            ok = lambda p: np.array(p) == y
            P = {k: ref["preds"][k] for k in ("ow_two_stream", "ow_full", "ow_delete", "ow_mask", "ow_typed")}
            r = {"n": len(y), "owte": float(ok(P["ow_two_stream"]).mean()), "ow_full": float(ok(P["ow_full"]).mean())}
            hide = max(("ow_delete", "ow_mask", "ow_typed"), key=lambda c: ok(P[c]).mean())
            r["hiding_test"] = dict(name=hide[3:], acc=float(ok(P[hide]).mean()), owte_minus=ci(ok(P["ow_two_stream"]), ok(P[hide])))
            E = {}
            for f in glob.glob(f"{OUT}/{task}_{n}_*_s*.json"):
                e = json.load(open(f))
                E[e["regime"]] = e
            owte = ok(P["ow_two_stream"])
            if "full_s0" in E:
                r["full_lora"] = dict(acc=E["full_s0"]["acc"], owte_minus=ci(owte, ok(E["full_s0"]["preds"])))
            tr = [h for h in ("delete_s0", "mask_s0", "typed_s0") if h in E]
            if tr:
                b = max(tr, key=lambda h: E[h]["acc"])
                r["hiding_trained"] = dict(name=b.split("_")[0], acc=E[b]["acc"], owte_minus=ci(owte, ok(E[b]["preds"])),
                                           all={h.split("_")[0]: E[h]["acc"] for h in tr})
            if "private_s0" in E:
                r["private_only"] = dict(acc=E["private_s0"]["acc"], owte_minus=ci(owte, ok(E["private_s0"]["preds"])))
            ft = f"{OUT}/{task}_{n}_fusiontrain.json"
            if os.path.exists(ft) and "delete_s0" in E and "private_s0" in E:
                F = json.load(open(ft))
                yt = np.array(F["labels"])
                a_t, b_t = ls(F["logits"]["delete_s0"]), ls(F["logits"]["private_s0"])
                ws = np.round(np.arange(0, 1.0001, 0.05), 2)
                w = float(max(ws, key=lambda w: ((w * a_t + (1 - w) * b_t).argmax(1) == yt).mean()))
                a_v, b_v = ls(E["delete_s0"]["logits"]), ls(E["private_s0"]["logits"])
                fz = {}
                for tag, ww in (("tuned", w), ("half", 0.5)):
                    pr = (ww * a_v + (1 - ww) * b_v).argmax(1)
                    fz[tag] = dict(w=ww, acc=float((pr == y).mean()), owte_minus=ci(owte, pr == y))
                r["late_fusion"] = fz
            seeds = [r["owte"]] + [E[f"oneway_s{s}"]["acc"] for s in (1, 2) if f"oneway_s{s}" in E]
            r["owte_seeds"] = dict(accs=seeds, mean=float(np.mean(seeds)), std=float(np.std(seeds, ddof=1)) if len(seeds) > 1 else None)
            out[f"{task}/{n}"] = r
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
    for k, r in out.items():
        f = lambda d: f"{d['acc']:.3f} ({d['owte_minus'][0]:+.3f} [{d['owte_minus'][1]:+.3f}, {d['owte_minus'][2]:+.3f}])" if d else "-"
        print(f"{k:34s} owte {r['owte']:.3f} | full LoRA {f(r.get('full_lora'))} | ow_full {r['ow_full']:.3f} | "
              f"hide test {f(r['hiding_test'])} | hide trained {f(r.get('hiding_trained'))} | private {f(r.get('private_only'))} | "
              f"fusion {f((r.get('late_fusion') or {}).get('tuned'))} | seeds {r['owte_seeds']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("drugsc", "mts"))
    ap.add_argument("--model")
    ap.add_argument("--train", choices=tuple(EVAL_COND))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.cache:
        cached_rows(a.task, "validation", 0, 0)
        cached_rows(a.task, "train", 3000, 0)
        fusion_rows(a.task)
        print(f"[e24] SUMMARY cached rows for {a.task}")
    elif a.train:
        train(a)
    elif a.eval:
        evaluate(a)
    elif a.summary:
        summary()


if __name__ == "__main__":
    main()
