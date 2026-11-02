#!/usr/bin/env python
"""E18 (two-stream on a decoder LLM, plaintext): the two-stream construction on a causal HF model (Llama-3.x). Prompt
tokens split into the plaintext stream P (public tokens, causal attention over public tokens only: they never see a
private token) and the encrypted stream E (the private tokens S, the last prompt token r that the answer is read from,
and any answer tokens: causal attention over everything, the public tokens' keys / values taken from P). Laid out as
one sequence [P; E] with the original position ids (RoPE) and a 4D mask; exact by construction at S = {} and S = all.

Tasks (private spans found by the client's detector, mapped to the LLM's tokens by character offsets):
  drugs   E15 drug reviews, which of the 10 conditions (validation, 1000), private = drug / disease / condition names
          (e15_drugs.detector(conditions=True), E4's fixed detector); zero-shot, the answer letter's logits at r.
  hatex   E13 HateXplain test posts, hate speech / offensive / normal, private = identity-group mentions.
  mts     E15m MTSamples transcriptions, which of the 10 specialties (validation, 485), private = medications, diseases,
          condition names and procedures (e15_mts.detector(procedures=True)).

  python selective/e18_oneway_llm.py --check --model meta-llama/Llama-3.2-1B-Instruct
  python selective/e18_oneway_llm.py --task drugs --model meta-llama/Llama-3.1-8B-Instruct [--lora DIR]
  python selective/e18_oneway_llm.py --task drugs --train-lora OUT --model ...       (one-way LoRA fine-tune)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
OUT = os.environ.get("E18_OUT", f"{D}/e18")
LETTERS = "ABCDEFGHIJ"


def layout(n, S, extra=0):
    """Order of the n prompt tokens (+ `extra` answer tokens at positions n..) in the [P; E] layout, their position ids,
    the layout index of every original position's E copy (or P copy if public), and the allowed-attention matrix."""
    S = set(int(s) for s in S)
    r = n - 1
    P = [i for i in range(n) if i not in S and i != r]
    E = sorted(S | {r}) + list(range(n, n + extra))
    order = P + E
    pos = np.array(order)
    stream = np.array([0] * len(P) + [1] * len(E))
    allowed = (pos[None, :] <= pos[:, None]) & ((stream[:, None] == 1) | (stream[None, :] == 0))
    where = {p: k for k, p in enumerate(order)}            # E copies come later and win for S and r
    return order, pos, where, allowed


def run_layout(model, ids, S, extra_ids=()):
    """Logits of the two-stream model at r and at each answer token (rows: r, then extra_ids[:-1])."""
    n = len(ids)
    full = list(ids) + list(extra_ids)
    order, pos, where, allowed = layout(n, S, len(extra_ids))
    dev = model.device
    x = torch.tensor([[full[i] for i in order]], device=dev)
    mask = torch.zeros(allowed.shape, dtype=model.dtype, device=dev)
    mask[~torch.from_numpy(allowed).to(dev)] = torch.finfo(model.dtype).min
    out = model(input_ids=x, position_ids=torch.tensor([pos], device=dev), attention_mask=mask[None, None])
    rows = [where[n - 1]] + [where[n + j] for j in range(len(extra_ids) - 1)] if extra_ids else [where[n - 1]]
    return out.logits[0, rows]


def run_plain(model, ids):
    return model(input_ids=torch.tensor([list(ids)], device=model.device)).logits[0, -1]


def load(name, dtype=torch.float32, lora=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora).eval()
    return tok, model


# ---------------------------------------------------------------- tasks

ORDER = os.environ.get("E18_ORDER", "text_first")     # text_first: the question after the text; question_first: before


def drug_prompt(text):
    import e15_drugs as E15
    opts = "\n".join(f"{LETTERS[k]}. {n}" for k, n in enumerate(E15.NAMES))
    q = f"Which condition is the patient taking the medication for? Options:\n{opts}\n\nAnswer with the letter only."
    if ORDER == "question_first":
        return f"{q}\n\nHere is the patient's review of the medication.\n\nReview: {text}", len(E15.NAMES)
    return f"Here is a patient's review of a medication.\n\nReview: {text}\n\n{q}", len(E15.NAMES)


def hatex_prompt(text):
    q = ("Is this post hate speech, normal, or offensive? Options:\nA. hate speech\nB. normal\nC. offensive\n\n"
         "Answer with the letter only.")                                                  # e13's label order
    if ORDER == "question_first":
        return f"{q}\n\nHere is the social media post.\n\nPost: {text}", 3
    return f"Here is a social media post.\n\nPost: {text}\n\n{q}", 3


def mts_prompt(text):
    import e15_mts as M
    opts = "\n".join(f"{LETTERS[k]}. {n}" for k, n in enumerate(M.CLASSES))
    q = f"Which medical specialty is this transcription from? Options:\n{opts}\n\nAnswer with the letter only."
    if ORDER == "question_first":
        return f"{q}\n\nHere is the medical transcription.\n\nTranscription: {text}", len(M.CLASSES)
    return f"Here is a medical transcription.\n\nTranscription: {text}\n\n{q}", len(M.CLASSES)


PROMPTS = {"drugs": drug_prompt, "hatex": hatex_prompt, "mts": mts_prompt}


def task_rows(task, split="validation"):
    """[(text, label, [char spans])]."""
    if task == "mts":
        import e15_mts as M
        rows = json.load(open(f"{D}/e15m/mts_t128.json"))[split]
        texts = [t for t, _l in rows]
        spans = M.detector(procedures=True)(texts)
        return [(t, l, [(s, e) for s, e, _k in sp]) for (t, l), sp in zip(rows, spans)]
    if task == "drugs":
        import e15_drugs as E15
        rows = json.load(open(f"{D}/e15/drugs_t128.json"))[split]
        texts = [t for t, _l in rows]
        spans = E15.detector(conditions=True)(texts)
        return [(t, l, [(s, e) for s, e, _k in sp]) for (t, l), sp in zip(rows, spans)]
    import re
    from e13_hatexplain import WORD
    rows = json.load(open(f"{D}/e13/hatexplain.json"))[split]
    return [(t, l, [(m.start(), m.end()) for m in re.finditer(r"[a-z]+", t.lower()) if m.group() in WORD]) for t, l in rows]


def encode(tok, task, text, spans, tail="r"):
    """Prompt token ids (chat template, generation prompt), the encrypted stream's prompt positions (the private tokens
    plus, by `tail`: "r" the last token only, "turn" the chat template's turn end and assistant header, "after" every
    token after the text), the answer letters' ids."""
    q, k = PROMPTS[task](text)
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    o = prompt.index(text)
    S = [j for j, (c0, c1) in enumerate(enc["offset_mapping"])
         if c1 > c0 and any(c0 < o + e and c1 > o + s for s, e in spans)]
    letters = [tok.encode(LETTERS[j], add_special_tokens=False)[0] for j in range(k)]
    n = len(enc["input_ids"])
    if tail == "after":
        S = sorted(set(S) | {j for j, (c0, _c1) in enumerate(enc["offset_mapping"]) if c0 >= o + len(text)})
    elif tail == "turn":
        eot = prompt.rindex("<|eot_id|>")
        S = sorted(set(S) | {j for j, (c0, _c1) in enumerate(enc["offset_mapping"]) if c0 >= eot})
    return enc["input_ids"], S, letters


@torch.no_grad()
def evaluate(a):
    from e12_report import ci
    tok, model = load(a.model, dtype=getattr(torch, a.dtype), lora=a.lora)
    rows = task_rows(a.task)
    if a.limit:
        rows = rows[:a.limit]
    P = {k: [] for k in ("full", "two_stream", "redaction")}
    y, nS, nT, nE, t0 = [], [], [], [], time.time()
    for j, (text, lab, spans) in enumerate(rows):
        ids, S, letters = encode(tok, a.task, text, spans)
        if not S:
            continue
        _ids, SE, _l = encode(tok, a.task, text, spans, a.tail)
        L = torch.tensor(letters, device=model.device)
        P["full"].append(int(run_plain(model, ids)[L].argmax()))
        P["two_stream"].append(int(run_layout(model, ids, SE)[0][L].argmax()))
        keep = [t for i, t in enumerate(ids) if i not in set(S)]
        P["redaction"].append(int(run_plain(model, keep)[L].argmax()))
        y.append(int(lab)); nS.append(len(S)); nT.append(len(ids)); nE.append(len(SE) + (len(ids) - 1 not in SE))
        if j % 100 == 0:
            print(f"[e18] {j}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    y = np.array(y)
    acc = {k: float((np.array(v) == y).mean()) for k, v in P.items()}
    agree = {k: float((np.array(v) == np.array(P["full"])).mean()) for k, v in P.items()}
    c_red = ci(np.array(P["two_stream"]) == y, np.array(P["redaction"]) == y)
    c_full = ci(np.array(P["two_stream"]) == y, np.array(P["full"]) == y)
    tag = (os.path.basename(a.lora.rstrip("/")) if a.lora else a.model.split("/")[-1]) + f"_{ORDER}_tail-{a.tail}" \
        + ("" if a.dtype == "float32" else f"_{a.dtype}")
    print(f"[e18] {a.task} {tag}: {len(y)} prompts with private tokens (median {np.median(nS):.0f} of {np.median(nT):.0f} "
          f"tokens, encrypted stream median {np.median(nE):.0f}); accuracy " + ", ".join(f"{k} {v:.3f}" for k, v in acc.items()) + " | agreement with full: "
          + ", ".join(f"{k} {v:.3f}" for k, v in agree.items()))
    print(f"[e18] {a.task} {tag}: two-stream minus redaction {c_red[0]:+.3f} [{c_red[1]:+.3f}, {c_red[2]:+.3f}]; "
          f"two-stream minus full {c_full[0]:+.3f} [{c_full[1]:+.3f}, {c_full[2]:+.3f}]")
    json.dump(dict(task=a.task, model=a.model, lora=a.lora, n=len(y), acc=acc, agree=agree, ci_redaction=c_red,
                   ci_full=c_full, preds=P, labels=y.tolist(), n_private=nS, n_tokens=nT, n_encrypted=nE, order=ORDER,
                   tail=a.tail),
              open(f"{OUT}/{a.task}_{tag}.json", "w"))


@torch.no_grad()
def check(a):
    """Exactness: two-stream at S = {} and S = all equals the full model (fp32 and bf16), plus a generic S."""
    for dtype in (getattr(torch, a.dtype),):
        tok, model = load(a.model, dtype=dtype)
        rows = task_rows("drugs")[:4]
        for text, _lab, spans in rows:
            ids, S, letters = encode(tok, "drugs", text, spans)
            ref = run_plain(model, ids).float()
            e0 = run_layout(model, ids, []).float()[0]
            e1 = run_layout(model, ids, range(len(ids))).float()[0]
            es = run_layout(model, ids, S).float()[0]
            ans = [letters[3], letters[5]]                           # two answer tokens appended to E
            ref_ans = model(input_ids=torch.tensor([list(ids) + ans], device=model.device)).logits[0, -3:-1].float()
            e_ans = run_layout(model, ids, range(len(ids)), ans).float()
            print(f"[e18] {str(dtype):14s} n={len(ids)} |S|={len(S)}: max |logit diff| S={{}} {(e0 - ref).abs().max():.2e}, "
                  f"S=all {(e1 - ref).abs().max():.2e}, answer rows S=all {(e_ans - ref_ans).abs().max():.2e}; "
                  f"real S changes the logits by {(es - ref).abs().max():.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


def train_lora(a):
    """One-way LoRA fine-tune on the train split: the answer letter's cross-entropy at r under the two-stream layout
    (private sets as found by the detector), rank 16 on q/k/v/o, bf16 base weights."""
    from peft import LoraConfig, get_peft_model
    tok, model = load(a.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.train()
    rows = task_rows(a.task, "train")
    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    t0 = time.time()
    for step in range(a.steps):
        loss_sum = 0.0
        for _ in range(a.batch):
            text, lab, spans = rows[int(rng.integers(len(rows)))]
            ids, S, letters = encode(tok, a.task, text, spans, a.tail)
            lo = run_layout(model, ids, S)[0] if S else model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1]
            loss = torch.nn.functional.cross_entropy(lo[torch.tensor(letters, device=model.device)][None].float(),
                                                     torch.tensor([int(lab)], device=model.device)) / a.batch
            loss.backward()
            loss_sum += float(loss)
        opt.step()
        opt.zero_grad()
        if step % 20 == 0:
            print(f"[e18] lora step {step}/{a.steps} loss {loss_sum:.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(os.path.expanduser(a.train_lora))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--task", default="drugs", choices=("drugs", "hatex", "mts"))
    ap.add_argument("--dtype", default="float32", help="float32 (exact) or bfloat16 (models too big for fp32)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--train-lora", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tail", default="r", choices=("r", "turn", "after"),
                    help="encrypted stream beyond the private tokens: r (last token), turn (turn end + assistant header), "
                         "after (every token after the text); E18_ORDER=question_first puts the question before the text")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.check:
        check(a)
    elif a.train_lora:
        train_lora(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
