#!/usr/bin/env python
"""E27: PrivacyRestore (Zeng et al., ACL 2025) reimplemented from the paper as a baseline, decoder LLMs on drugsc / mts.

Server model: E24's full-attention LoRA (same recipe as E21's one-way LoRA, ordinary attention), merged and frozen: the
server's best task model, which PrivacyRestore leaves unchanged. Private spans: our detectors' (cached E24 rows).
  1. Edited heads (paper Sec. 4.1, App. G): per span value s among the 50 most frequent, a logistic probe per attention
     head on the head's output at the last token of the complete prompt (label: the input contains s; balanced; 70/30);
     each span's top-K heads by probe accuracy, Top-K Heads Selector (average rank score 1..K), common set Hc.
     K = 175 per 1024 heads (the paper's Llama-2-7B choice), scaled to the model's head count.
  2. Restoration vectors (Sec. 4.1): one r_s in R^{K x d} per span value seen >= 3 times in the train texts, plus one per
     kind (medication / condition / procedure / other) for the rest (departure: our detectors' spans are open-ended);
     trained with ORPO (Eq. 2-3, lambda 0.1) to make the frozen model on the prompt with the private tokens deleted
     answer what it answers on the complete prompt (a) rather than what it answers without restoration (a-hat).
  3. Client (Sec. 4.2): span weights w_s = average attention the span receives in BERT-base (all layers and heads),
     Z = sum w_s r_s / ||sum w_s r_s|| over the concatenation of all edited heads, R = Z + N with p(N) ~ exp(-eps ||N||)
     (Feyisetan et al.: uniform direction, ||N|| ~ Gamma(d, 1/eps), d = K * head_dim).
  4. Server (Eq. 7): u_h <- u_h + ||u_h|| R_h at the last token for every h in Hc, on the prompt with the private tokens
     deleted (E21's delete condition).
Evaluation on E21's validation texts: eps = inf (no noise), 75 (paper default), 25 (stronger), and a per-coordinate
Laplace(1/eps) variant at 75 and 25; plus the frozen model on the complete and the deleted prompt. Paired 95% CIs of OWTE (E21
one-way LoRA) minus each.

  python selective/e27_privacyrestore.py --task drugsc --model M [--epochs 2]
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/data/selective-fhe")
OUT = f"{D}/e27"
KIND = {"DRUG": "MEDICATION", "Medication": "MEDICATION", "CONDITION": "CONDITION", "Disease_disorder": "CONDITION",
        "Therapeutic_procedure": "PROCEDURE", "Diagnostic_procedure": "PROCEDURE"}
EPS = [("inf", None, "l2"), ("75", 75.0, "l2"), ("25", 25.0, "l2"), ("75_coord", 75.0, "coord"), ("25_coord", 25.0, "coord")]


def key(text, s, e, k):
    return text[s:e].lower().strip(), KIND.get(k, "OTHER")


class Steer:
    """Forward pre-hooks on every o_proj: adds ||u_h|| R_h to the last token's head outputs u_h of the edited heads."""

    def __init__(self, model, heads, hd):
        self.layers = model.model.layers
        self.heads, self.hd, self.R, self.cap = heads, hd, None, None
        self.by_layer = collections.defaultdict(list)
        for j, (l, h) in enumerate(heads):
            self.by_layer[l].append((j, h))
        for l, lay in enumerate(self.layers):
            lay.self_attn.o_proj.register_forward_pre_hook(self.hook(l))

    def hook(self, l):
        def f(_mod, args):
            x = args[0]
            if self.cap is not None:
                self.cap[l] = x[0, -1].detach().float().cpu()
            if self.R is None or l not in self.by_layer:
                return None
            js = torch.tensor([j for j, _h in self.by_layer[l]], device=x.device)
            hs = torch.tensor([h for _j, h in self.by_layer[l]], device=x.device)
            u = x[0, -1].view(-1, self.hd)[hs].float()                          # the edited heads' outputs
            add = u.norm(dim=-1, keepdim=True) * self.R[js].float()             # ||u_h|| R_h (Eq. 7), out of place
            d = torch.zeros(x.shape[-1] // self.hd, self.hd, device=x.device, dtype=torch.float32).index_add(0, hs, add)
            pad = torch.zeros_like(x)
            pad[0, -1] = d.view(-1).to(x.dtype)
            return (x + pad,) + tuple(args[1:])
        return f


def awa_weights(bert, btok, text, spans):
    """Per span: the average attention it receives in BERT-base (all layers, heads and query tokens), Eq. 4."""
    enc = btok(text, return_offsets_mapping=True, truncation=True, max_length=512, return_tensors="pt")
    off = enc.pop("offset_mapping")[0].tolist()
    att = torch.stack(bert(**{k: v.cuda() for k, v in enc.items()}, output_attentions=True).attentions)  # L,1,H,T,T
    a = att[:, 0].mean((0, 1)).sum(0).cpu()                                             # attention received per token
    n = att.shape[-1]
    out = []
    for s, e, _k in spans:
        idx = [j for j, (c0, c1) in enumerate(off) if c1 > c0 and c0 < e and c1 > s]
        out.append(float(a[idx].sum()) / n if idx else 1.0 / n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("drugsc", "mts"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--probe-n", type=int, default=1500)
    a = ap.parse_args()
    import e18_oneway_llm as L
    import e21_llm as E
    from e24_llm import cached_rows
    from e12_report import ci
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    n = a.model.split("/")[-1]
    os.makedirs(OUT, exist_ok=True)
    dtype = torch.bfloat16 if "32B" in n else torch.float32
    tok, model = L.load(a.model, dtype=dtype)
    model = PeftModel.from_pretrained(model, f"{D}/e24/lora/{a.task}_{n}_full_s0").merge_and_unload().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    H = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    NL = cfg.num_hidden_layers
    K = int(round(175 / 1024 * NL * H))
    tr, va = cached_rows(a.task, "train", 3000, 0), cached_rows(a.task, "validation", 0, 0)
    t0 = time.time()

    def prompt_ids(text, spans, deleted):
        ids, S, _Q, letters = E.encode(tok, a.task, text, spans)
        return ([t for i, t in enumerate(ids) if i not in set(S)] if deleted else ids), letters

    steer = Steer(model, [], hd)

    def last_logits(ids):
        return model(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1]

    # ---- frozen model's answers on the complete (a) and deleted (a-hat) prompts, and the probe features
    rng = np.random.default_rng(0)
    probe_idx = set(rng.choice(len(tr), min(a.probe_n, len(tr)), replace=False).tolist())
    feats, A, Ah = {}, [], []
    with torch.no_grad():
        for i, (text, _lab, spans) in enumerate(tr):
            ids, letters = prompt_ids(text, spans, False)
            steer.cap = {} if i in probe_idx else None
            A.append(int(last_logits(ids)[letters].argmax()))
            if steer.cap is not None:
                feats[i] = torch.stack([steer.cap[l] for l in range(NL)]).view(NL * H, hd).half()
            steer.cap = None
            ids, letters = prompt_ids(text, spans, True)
            Ah.append(int(last_logits(ids)[letters].argmax()))
    print(f"[e27] answers + probe features {time.time() - t0:.0f}s", flush=True)

    # ---- edited heads: probes per frequent span value, Top-K Heads Selector
    cnt = collections.Counter(key(t, s, e, k)[0] for t, _l, sp in tr for s, e, k in sp)
    top = [v for v, c in cnt.most_common(50) if c >= 20]
    pi = sorted(feats)
    X = torch.stack([feats[i] for i in pi]).cuda().float()                      # m, NL*H, hd
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    has = [{key(tr[i][0], s, e, k)[0] for s, e, k in tr[i][2]} for i in pi]
    scores = collections.defaultdict(list)
    for v in top:
        pos = [j for j, h in enumerate(has) if v in h]
        neg = [j for j, h in enumerate(has) if v not in h]
        if len(pos) < 10:
            continue
        neg = list(rng.choice(neg, min(len(neg), len(pos)), replace=False))
        idx = np.array(pos + neg)
        y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device="cuda")
        perm = rng.permutation(len(idx))
        cut = int(0.7 * len(idx))
        trn, tst = perm[:cut], perm[cut:]
        W = torch.zeros(NL * H, hd, device="cuda", requires_grad=True)
        b = torch.zeros(NL * H, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-3)
        Xt = X[idx[trn]]
        for _ in range(150):
            lo = torch.einsum("mhd,hd->mh", Xt, W) + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, y[trn][:, None].expand_as(lo))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = ((torch.einsum("mhd,hd->mh", X[idx[tst]], W) + b > 0).float() == y[tst][:, None]).float().mean(0)
        best = torch.argsort(acc, descending=True)[:K].tolist()
        for r, h in enumerate(reversed(best)):                                  # rank 1 = lowest accuracy of the top K
            scores[h].append(r + 1)
    avg = {h: float(np.mean(v)) for h, v in scores.items()}
    Hc = sorted(avg, key=lambda h: -avg[h])[:K]
    heads = sorted((h // H, h % H) for h in Hc)
    del X, feats
    torch.cuda.empty_cache()
    print(f"[e27] {len(top)} probed spans, K = {K} edited heads over {len({l for l, _ in heads})} layers "
          f"{time.time() - t0:.0f}s", flush=True)
    steer = None
    for lay in model.model.layers:
        lay.self_attn.o_proj._forward_pre_hooks.clear()
    steer = Steer(model, heads, hd)

    # ---- restoration vectors and the client's attention-aware weights
    vocab = [v for v, c in cnt.items() if c >= 3] + ["<KIND:MEDICATION>", "<KIND:CONDITION>", "<KIND:PROCEDURE>",
                                                     "<KIND:OTHER>"]
    vid = {v: j for j, v in enumerate(vocab)}

    def span_ids(text, spans):
        out = []
        for s, e, k in spans:
            v, kd = key(text, s, e, k)
            out.append(vid.get(v, vid[f"<KIND:{kd}>"]))
        return out

    btok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").cuda().eval()
    with torch.no_grad():
        Wtr = [awa_weights(bert, btok, t, sp) for t, _l, sp in tr]
        Wva = [awa_weights(bert, btok, t, sp) for t, _l, sp in va]
    del bert
    Rv = torch.nn.Parameter(0.01 * torch.randn(len(vocab), K, hd, device="cuda"))

    def meta(text, spans, w):
        ids = torch.tensor(span_ids(text, spans), device="cuda")
        z = (torch.tensor(w, device="cuda")[:, None, None] * Rv[ids]).sum(0)
        return z / z.norm()

    opt = torch.optim.Adam([Rv], lr=a.lr)
    log = []
    for ep in range(a.epochs):
        tot, nb = 0.0, 0
        for i in np.random.default_rng(ep).permutation(len(tr)):
            text, _lab, spans = tr[i]
            ids, letters = prompt_ids(text, spans, True)
            steer.R = meta(text, spans, Wtr[i])
            lp = torch.log_softmax(last_logits(ids).float(), -1)[torch.tensor(letters, device="cuda")]
            la, lh = lp[A[i]], lp[Ah[i]]
            loss = -la
            if A[i] != Ah[i]:
                lodds = lambda x: x - torch.log1p(-torch.exp(x).clamp(max=1 - 1e-6))
                loss = loss - a.lam * torch.nn.functional.logsigmoid(lodds(la) - lodds(lh))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            if nb % 500 == 0:
                print(f"[e27] epoch {ep} {nb}/{len(tr)} loss {tot / nb:.3f} {time.time() - t0:.0f}s", flush=True)
        log.append(tot / nb)
    steer.R = None

    # ---- evaluation
    ref = json.load(open(f"{D}/e21/llm/{a.task}_{n}.json"))
    y = np.array([l for _t, l, _s in va])
    assert (y == np.array(ref["labels"])).all()
    owte = np.array(ref["preds"]["ow_two_stream"]) == y
    P = collections.defaultdict(list)
    d = K * hd
    nrng = np.random.default_rng(123)
    with torch.no_grad():
        for i, (text, _lab, spans) in enumerate(va):
            for cond, deleted in (("full", False), ("delete", True)):
                ids, letters = prompt_ids(text, spans, deleted)
                P[cond].append(int(last_logits(ids)[letters].argmax()))
            ids, letters = prompt_ids(text, spans, True)
            Z = meta(text, spans, Wva[i])
            for tag, eps, kind in EPS:
                if eps is None:
                    N = 0.0
                elif kind == "l2":
                    u = nrng.standard_normal(d)
                    N = torch.tensor(u / np.linalg.norm(u) * nrng.gamma(d, 1.0 / eps), device="cuda",
                                     dtype=torch.float32).view(K, hd)
                else:
                    N = torch.tensor(nrng.laplace(0, 1.0 / eps, d), device="cuda", dtype=torch.float32).view(K, hd)
                steer.R = Z + N
                P[f"pr_eps{tag}"].append(int(last_logits(ids)[letters].argmax()))
                steer.R = None
    res = dict(task=a.task, model=a.model, n=len(y), K=K, head_dim=hd, edited_layers=sorted({l for l, _ in heads}),
               vocab=len(vocab), probed_spans=len(top), epochs=a.epochs, lr=a.lr, lam=a.lam, train_loss=log,
               server_model="E24 full-attention LoRA (merged, frozen)", owte=float(owte.mean()),
               acc={c: float((np.array(v) == y).mean()) for c, v in P.items()},
               owte_minus={c: ci(owte, np.array(v) == y) for c, v in P.items()},
               noise_norm_l2={t: (K * hd / e if e else 0.0) for t, e, kd in EPS if kd == "l2"},
               preds={c: list(map(int, v)) for c, v in P.items()}, labels=y.tolist(), seconds=time.time() - t0)
    json.dump(res, open(f"{OUT}/{a.task}_{n}.json", "w"))
    print(f"[e27] SUMMARY {a.task} {n}: OWTE {res['owte']:.3f} | " + ", ".join(
        f"{c} {v:.3f} ({res['owte_minus'][c][0]:+.3f})" for c, v in res["acc"].items()) + f" | {time.time() - t0:.0f}s",
        flush=True)


if __name__ == "__main__":
    main()
