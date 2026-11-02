#!/usr/bin/env python
"""E18 phase 2, milestone 1: the two-stream prefill of Llama-3.1 on Perseus's CPU twin (fakecap) for one E15 drug
prompt, against the float64 two-stream model. The public tokens (the prompt minus the private tokens and the readout,
BOS as upstream's public prefix) are ordinary pages of the paged blocked GQA prefill (their own causal attention: the
plaintext stream; on the device the server computes them in plaintext and stores their K / V); the encrypted stream
(private tokens + the last prompt token r) is one more page at its true positions, attending to every stored page
(sfhe.llama_ts). Reports per block the twin's error against the float64 two-stream model on the public rows and on
the encrypted rows, the readout logits' KL / argmax when every layer runs, and the twin's bootstrap counts per page
next to the full encrypted prefill of the same prompt (--full: upstream's paged prefill on the whole prompt).

Run under the shipped prefill plan's runtime (as upstream's capture):
  RT=$(python -c "from perseus import config; print(' '.join(k+'='+v for k,v in \\
      config.runtime_env('bootstrap_placements/llama31_8b_prefill128').items()))")
  env $RT PF_LAST_QUERY=0 PF_PCMM_CALIB_IN=bootstrap_placements/llama31_8b_prefill128/pcmm_calib.json \\
      python selective/e18_llama_ts.py --idx 0 --layers 2 [--full] [--capture GRAPH_DIR]
  (PF_PCMM_CALIB=FILE also writes the PCMM linears' measured input bands, as upstream's calibration runs)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
PLAN = "bootstrap_placements/llama31_8b_prefill128"
CONFIGS = "configs/model/approximation/llama31_8b_n32_cheb1e-5_kc256/configs.json"
D = os.path.expanduser("~/data/selective-fhe")


# SFHE_TS_PAD=copy: a stored page's padding keys / values are copies of its last real token's (as upstream pads a page
# with copies of its last token; never attended either way). Zeros (the default) leave an all-zero key ciphertext on a
# mostly-padding page, which the planner cannot size ("no captured magnitude").
PAD_COPY = os.environ.get("SFHE_TS_PAD", "") == "copy"
# SFHE_TS_PADQ=r (E26 fix): the encrypted page's padding query lanes take the readout's position r (RoPE and the
# geometry's query position), so each is an exact copy of the readout row and attends only real keys. Before, their
# query position was BIG: they attended every key including the stored pages' padding keys, drifted (|x| 11 by block 1
# on prompts whose second stored page is mostly padding) and pushed later blocks' polynomial inputs out of range.
PADQ_R = os.environ.get("SFHE_TS_PADQ", "") == "r"


def query_positions(kpos_e, r):
    """The geometry's query positions of the encrypted page (padding lanes at r under SFHE_TS_PADQ=r)."""
    return [x if (x < (1 << 30) or not PADQ_R) else r for x in kpos_e]


def n_real(kp):
    return sum(x < (1 << 30) for x in kp)


def enron_prompts(L, n, tokenizer, cat=1, max_enc=100):
    """n E14 Enron validation e-mails (cat > 1: `cat` consecutive e-mails joined) as text-first spam / ham prompts of
    exactly L Llama tokens (the e-mail text cut to fit), private spans from E14's detector (people, organisations,
    places, e-mail addresses, phones) on the final text; only prompts whose encrypted stream (private + r) has at most
    max_enc tokens. Cached in p2/prompts_enron_L<L>.json: [{ids, S, label}]."""
    f = f"{D}/e18/p2/prompts_enron_L{L}.json"
    if os.path.exists(f):
        return json.load(open(f))
    from transformers import AutoTokenizer, pipeline
    import e14_enron as E14
    tok = AutoTokenizer.from_pretrained(tokenizer)
    tagger = pipeline("token-classification", model="dslim/bert-base-NER-uncased", aggregation_strategy="simple", device=-1)
    rows = json.load(open(f"{D}/e14/enron_t512.json"))["validation"]
    q = "Is this e-mail spam or a normal e-mail? Options:\nA. normal (ham)\nB. spam\n\nAnswer with the letter only."
    wrap = lambda t: tok.apply_chat_template([{"role": "user", "content": f"Here is an e-mail.\n\nE-mail: {t}\n\n{q}"}],
                                            tokenize=False, add_generation_prompt=True)
    over = len(tok(wrap(""), add_special_tokens=False)["input_ids"])
    out, i = [], 0
    while len(out) < n and i + cat <= len(rows):
        grp, i = rows[i:i + cat], i + cat
        text = "\n\n".join(t for t, _l in grp)
        tt = tok(text, add_special_tokens=False)["input_ids"]
        k = L - over
        if len(tt) < k:
            continue
        for _ in range(8):                       # cut the text to k tokens; re-tokenised, the prompt may move by a few
            t = tok.decode(tt[:k])
            prompt = wrap(t)
            n_ = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if n_ == L:
                break
            k += L - n_
        if n_ != L:
            continue
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        o = prompt.index(t)
        spans = [(s_, e_) for s_, e_, _k in E14.detect([t], tagger, 0.5)[0]]
        S = sorted({j for j, (c0, c1) in enumerate(enc["offset_mapping"]) if c1 > c0
                    and any(c0 < o + e_ and c1 > o + s_ for s_, e_ in spans)})
        if not S or len(S) + 1 > max_enc:
            continue
        out.append(dict(ids=list(enc["input_ids"]), S=S, label=int(grp[-1][1]), first=i - cat))
    json.dump(out, open(f, "w"))
    return out


def owt_ids(i, pages):
    """Openwebtext prompt i of `pages` x 128 tokens: `pages` cached 128-token windows joined (BOS once; the hub is
    offline here); the cost does not depend on the text."""
    from perseus.examples.llama import hf
    w = hf.text_windows(16, 128)
    k = i * pages
    if k + pages > len(w):
        raise ValueError(f"openwebtext prompt {i} of {pages} pages: only {len(w)} cached windows")
    ids = [int(t) for t in w[k]]
    for j in range(1, pages):
        ids += [int(t) for t in w[k + j][1:]]
    return ids + [int(w[k + pages - 1][-1])] * (pages - 1)     # pad to pages x 128 (the dropped BOS slots)


def prompt(a):
    """(token ids with BOS first, private token indices S) of validation review a.idx, as e18_oneway_llm builds it.
    a.owt = L: the full encrypted prefill's second page instead: an openwebtext window of L tokens (upstream's prompt
    source; its shipped ranges hold there), page 0 = positions 1..127 as the store, every later token "encrypted"
    (the full model's page 1: plain causal attention over page 0 and itself)."""
    if getattr(a, "enron", 0):                               # E14 e-mails at a.enron tokens (a.cat joined)
        from transformers import AutoTokenizer
        P = enron_prompts(a.enron, 40, a.tokenizer, cat=getattr(a, "cat", 1))[a.idx]
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        return P["ids"], P["S"], [tok.encode(c, add_special_tokens=False)[0] for c in "AB"], P["label"]
    if getattr(a, "owt_page", 1) >= 2:                      # the full prefill's page p >= 2: pages 0..p-1 stored
        p = a.owt_page
        ids = owt_ids(a.idx, p + 1)
        return ids, list(range(128 * p, 128 * (p + 1) - 1)), [], -1
    if getattr(a, "owt", 0):
        from perseus.examples.llama import hf
        if a.idx < 2:                                       # the cached 512-token windows
            ids = [int(t) for t in hf.text_windows(a.idx + 1, 512)[a.idx][:a.owt]]
        else:                                               # two cached 128-token windows joined (BOS once; offline)
            w = hf.text_windows(16, 128)
            k = 2 * (a.idx - 2)
            ids = ([int(t) for t in w[k]] + [int(t) for t in w[k + 1][1:]])[:a.owt]
        if getattr(a, "owt_page", 1) == 0:                   # the full prefill's page 0: BOS + 127 tokens, no store
            return ids[:128], list(range(1, 127)), [], -1
        return ids, list(range(128, a.owt - 1)), [], -1
    from transformers import AutoTokenizer
    import e15_drugs as E15
    import e18_oneway_llm as E18
    text, label = json.load(open(f"{D}/e15/drugs_t128.json"))[a.split][a.idx]
    spans = [(s, e) for s, e, _k in E15.detector(conditions=True)([text])[0]]
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids, S, letters = E18.encode(tok, "drugs", text, spans, tail="r")
    return list(ids), sorted(set(S)), letters, label


def layout(ids, S):
    """Public token indices (BOS excluded: the public prefix), encrypted-stream indices E = S + r."""
    n, r = len(ids), len(ids) - 1
    pub = [i for i in range(1, n) if i not in set(S) and i != r]
    return pub, sorted(set(S) | {r})


def exact_hidden_ts(model, ids, pub, E, n_layers):
    """Float64 Llama on the two-stream layout [BOS; public; E] (true positions; public rows see BOS + public keys,
    E rows every key at or before their position): per block the hidden rows (rotated basis) of public + E rows."""
    from perseus.examples.llama.weights import rotation_of
    st = model.w.store
    c = st.config
    H, H_kv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
    Dm = int(c["hidden_size"]); dh = Dm // H
    eps = float(c["rms_norm_eps"])
    inv = np.asarray(model.cfgs.model["rope_inv_freq"], dtype=np.float64)
    t = lambda nm: st.tensor(nm).astype(np.float64)
    order = [0] + pub + E
    pos = np.array(order)
    stream = np.array([0] * (1 + len(pub)) + [1] * len(E))
    allowed = (pos[None, :] <= pos[:, None]) & ((stream[:, None] == 1) | (stream[None, :] == 0))
    mask = np.where(allowed, 0.0, -np.inf)
    x = t("model.embed_tokens.weight")[np.asarray([ids[i] for i in order])]
    T = x.shape[0]
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(np.concatenate([ang, ang], 1)), np.sin(np.concatenate([ang, ang], 1))

    def rope(v):
        h = dh // 2
        return v * cos[:, None, :] + np.concatenate([-v[..., h:], v[..., :h]], -1) * sin[:, None, :]
    rms = lambda v, g: v / np.sqrt((v * v).mean(1, keepdims=True) + eps) * g
    R = rotation_of(model.cfgs, Dm)
    out, kvs = [], []
    for b in range(n_layers):
        p = f"model.layers.{b}."
        nrm = rms(x, t(p + "input_layernorm.weight"))
        q = rope((nrm @ t(p + "self_attn.q_proj.weight").T).reshape(T, H, dh))
        k = rope((nrm @ t(p + "self_attn.k_proj.weight").T).reshape(T, H_kv, dh))
        v = (nrm @ t(p + "self_attn.v_proj.weight").T).reshape(T, H_kv, dh)
        kvs.append((k.reshape(T, -1).copy(), v.reshape(T, -1).copy()))
        o = np.zeros((T, H, dh))
        for hh in range(H):
            kh = hh // (H // H_kv)
            s = q[:, hh] @ k[:, kh].T / np.sqrt(dh) + mask
            pr = np.exp(s - s.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            o[:, hh] = pr @ v[:, kh]
        x = x + o.reshape(T, Dm) @ t(p + "self_attn.o_proj.weight").T
        nrm = rms(x, t(p + "post_attention_layernorm.weight"))
        g = nrm @ t(p + "mlp.gate_proj.weight").T
        x = x + (g / (1 + np.exp(-g)) * (nrm @ t(p + "mlp.up_proj.weight").T)) @ t(p + "mlp.down_proj.weight").T
        out.append((x if R is None else x @ R)[1:].copy())
    return out, kvs


def store_patterns(model, kvs, rows_per_page):
    """Per layer, per public page: (K, V) SinC slot patterns from the float64 public stream: every row's (post-RoPE K,
    V) on its key row of the SinC operands, as upstream places the BOS (page 0's key 127). rows_per_page[j] = the
    layout rows of page j's keys (index into kvs rows; None = padding)."""
    from perseus.examples.llama import prefix as PX
    from perseus.impl import sinc as sc
    from perseus.impl.ccmm_attn import _bos
    B, d = model.blocks, model.rt.dims
    H, H_kv = model.H, model.H_kv
    dh = d.dim // H

    def kv_dims(k, v):          # one token's raw (k, v) -> ks[kv][dim], vs[kv][dim] (ccmm_attn._bos of its prefix vectors)
        kf, vf = PX.expanded_heads(k, H, H_kv, dh), PX.expanded_heads(v, H, H_kv, dh)
        kg, vg = B.kv_vec(kf), B.kv_vec(vf)
        pre = [(np.repeat(kg[j * B.nch:(j + 1) * B.nch], B.t), np.repeat(vg[j * B.nch:(j + 1) * B.nch], B.t))
               for j in range(B.Kkv)]
        return _bos(pre)
    out = []
    for K, V in kvs:
        layer = []
        for rows in rows_per_page:
            kp, vp = np.zeros((sc.NCT, sc.SLOTS)), np.zeros((sc.NCT, sc.SLOTS))
            for tok, row in enumerate(rows):
                if row is not None:
                    ks, vs = kv_dims(K[row], V[row])
                    kp += sc.kv_row_pattern(ks, tok)
                    vp += sc.kv_row_pattern(vs, tok)
            layer.append((kp, vp))
        out.append(layer)
    return out


def raw_store(model, pats):
    """One layer's stored pages as encryptions of their public patterns (the stage's input ciphertexts)."""
    ops = model.rt.ops
    zero = lambda: ops.core.encode_token_input(ops.inf, np.zeros(model.rt.dims.dim))
    return [([ops.add_pt(zero(), kp[o]) for o in range(len(kp))], [ops.add_pt(zero(), vp[o]) for o in range(len(vp))])
            for kp, vp in pats]


def prep_store(rt, eng, raw, gscale, merge, kmeta=None, vmeta=None):
    """A raw stored page -> the CCMM operands attention_paged stores (inside the stage: recorded with the block). The
    raw ciphertexts take the (level, degree) of the page's own K / V (kmeta, vmeta): on the device the server encrypts
    the public K / V there."""
    from perseus.impl import ccmm_paged as CP, sinc as sc
    from perseus.impl.ccmm_attn import KMAX, QMAX, SHIFT_PV, SHIFT_QK, VMAX
    Ks, Vs = raw
    for cts, meta in ((Ks, kmeta), (Vs, vmeta)):
        if meta is not None:              # the own K / V's level; degree 2, the degree the planner enters block inputs at
            for c in cts:
                c.level, c.noise_deg = int(meta[0]), 2
    if merge:
        Ks, Vs = sc.dup(rt, Ks), sc.dup(rt, Vs)
    kw = dict(keep=True, export=True) if CP._DEVKV else {}
    return (eng.prep_b(Ks, amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=0.5 * gscale, merge=merge, **kw),
            eng.prep_b(Vs, amax=1.5, bmax=VMAX, shift=SHIFT_PV, merge=merge, **kw))


def pages_of(ids, pub, E, BIG, t):
    """[(tokens, PosChunks, key positions for the geometry)]: public pages (127, then 128 tokens; padding = copies of
    the page's last token after the prompt), then the encrypted page (E + padding)."""
    from sfhe.llama_ts import PosChunks
    r = len(ids) - 1
    out, i, p, extra = [], 0, 0, 0
    groups = []
    while i < len(pub):
        n = 127 if p == 0 else 128
        groups.append(pub[i:i + n])
        i += n
        p += 1
    groups.append(E)
    for p, g in enumerate(groups):
        n = 127 if p == 0 else 128
        if len(g) > n:
            raise ValueError(f"page {p}: {len(g)} tokens > {n}")
        pad = n - len(g)
        toks = [ids[j] for j in g] + [ids[g[-1]]] * pad
        rope_pos = list(g) + ([r] * pad if (PADQ_R and p == len(groups) - 1)
                              else [r + 1 + extra + k for k in range(pad)])
        extra += pad
        kpos = list(g) + [BIG] * pad
        out.append((toks, PosChunks(n, t, base=int(g[0]), pos=tuple(rope_pos)), kpos))
    return out


def run_pages(model, pages, geom_page=None, geom=None, ref_rows=None, log=print, stores=None, first=0, store_pats=None):
    """The paged prefill loop of LlamaPrefill.prefill_paged on explicit pages (page indices first, first + 1, ...;
    `stores`: the earlier pages' K / V operands per layer); returns per page (Xs, ch, bootstraps)."""
    from perseus.examples.llama.prefill import llama_prefill_block_blocked
    from perseus.impl import blocked as BL
    from perseus.impl.prefill import evict_block_masks
    from sfhe.llama_ts import set_geometry
    B = model.blocks
    stores = stores if stores is not None else [[] for _ in range(model.n_layers)]
    fhe = model.inf.fhe
    model.token_begin(0)
    out = []
    for p, (toks, ch, _kp) in enumerate(pages, start=first):
        set_geometry(p, geom if p == geom_page else None)
        b0 = int(fhe.total_bootstraps)
        t0 = time.time()
        Xs = BL.encode_blocked(model.rt, np.stack([model.w.embedding(t) for t in toks]), ch, B)
        for b in range(model.n_layers):
            model.rt._scope_visits.clear()
            if store_pats is not None:                      # this layer's stored pages: the stage's extra inputs
                stores[b] = raw_store(model, store_pats[b])
            cap = model.enter_stage(b, 1)
            BL.name_blocked(model.inf, Xs)
            with model.rt.step(f"blk{b}"):
                Xs = llama_prefill_block_blocked(model.rt, Xs, model.pcmm_weights(b), model.cfgs.block(b), ch,
                                                 model.rope, model.prefix_kv(b), B, None, layer=b,
                                                 paged=(p, stores[b], p == first + len(pages) - 1), last=None)
            model.exit_stage(b, cap)
            evict_block_masks(model.rt)
            if ref_rows is not None:
                h = model.hidden_rows(Xs, ch)
                ref = ref_rows(p, b)
                n = len(ref)
                err = np.abs(h[:n] - ref).max(1)
                log(f"[e18-ts] page {p} block {b}: maxerr {err.max():.3g} (row {int(err.argmax())}) |x| "
                    f"{np.abs(ref).max():.3g} rel {err.max() / np.abs(ref).max():.3g}")
        bts = int(fhe.total_bootstraps) - b0
        log(f"[e18-ts] page {p}: {len(toks)} lanes, {bts} bootstraps, {time.time() - t0:.0f}s")
        out.append((Xs, ch, bts))
        set_geometry(p, None)
    model.token_end(0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idx", type=int, default=0, help="E15 drug review (validation index)")
    ap.add_argument("--split", default="validation", choices=("validation", "train"))
    ap.add_argument("--owt", type=int, default=0, help="L: the full prefill's page 1 on an openwebtext window of L tokens")
    ap.add_argument("--owt-page", type=int, default=1, help="--owt: page 0 (BOS + 127 tokens), page 1 (241-token "
                    "window), or page p >= 2 (p + 1 joined 128-token windows, pages 0..p-1 stored)")
    ap.add_argument("--enron", type=int, default=0, help="L: E14 Enron e-mails as spam / ham prompts of L tokens")
    ap.add_argument("--cat", type=int, default=1, help="--enron: e-mails joined per prompt (long prompts)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct", help="chat template / tokens")
    ap.add_argument("--configs", default=CONFIGS)
    ap.add_argument("--store", choices=("exact", "twin"), default="exact", help="the public pages' K / V: from the "
                    "float64 public stream (as the server computes them in plaintext), or run encrypted on the twin")
    ap.add_argument("--capture", default=None, help="graph dir: capture the encrypted page's blocks (magnitudes) "
                    "for the planner (FHE_GRAPH_DIR)")
    ap.add_argument("--full", action="store_true", help="also upstream's paged prefill on the whole prompt (bootstraps)")
    ap.add_argument("--out", default=f"{D}/e18/twin_ts.json")
    a = ap.parse_args()
    os.chdir(REPO)
    ids, S, letters, label = prompt(a)
    pub, E = layout(ids, S)
    print(f"[e18-ts] review {a.idx}: {len(ids)} tokens, {len(S)} private, encrypted stream {len(E)} at {E}, "
          f"{len(pub)} public; model {os.environ.get('LLAMA_MODEL', 'meta-llama/Llama-3.1-8B')}", flush=True)
    from perseus import config as pcfg
    RT = pcfg.runtime_env(PLAN)
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": "1", "FIDESLIB_ENCODE_NOSYNC": "0", **RT,
            "LLAMA_PACKING": "cachemir_complex", "PF_LAST_QUERY": "0"}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ[k] = str(v)
    from perseus.examples.llama import env, hf, weights as W
    from perseus.examples.llama.config import load_configs
    from perseus.examples.llama.prefill import LlamaPrefill, plain_logits
    from perseus.impl import fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.ccmm import engine as ccmm_engine
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import engine as pcmm_engine, rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import prefill_rot_steps
    from perseus.impl.sinc import rot_steps as sinc_rot_steps
    from sfhe.llama_ts import BIG, Geometry, install
    install()
    t0 = time.time()
    cfgs = load_configs(a.configs)
    store = W.SafetensorsStore(hf.NAME)
    dims = W.llama_dims(store.config)
    rot = (prefill_rot_steps(dims) | pcmm_rot_steps() | ladder_radix_steps(dims, ladder_radix_bits())
           | sinc_rot_steps() | {16, -16} | blocked_rot_steps(Blocks(dims, int(os.environ.get("PF_NCH", "512"))),
                                                            ladder_radix_bits()))
    if a.capture:
        os.environ["FHE_GRAPH_DIR"] = a.capture
        os.environ.setdefault("FHE_GRAPH_CAPTURE_TOKEN", "1")
    env.export_env(None, chain="n32", **{"CKKS_COMPLEX": "1", **over})
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | rot)
    sess = fakecap.FakeSession(inf)
    lw = W.LlamaWeights(store, cfgs, a.configs, dims, "cachemir_complex", pinned=None)
    model = LlamaPrefill(inf, lw, cfgs, core=fakecap.core, n_layers=a.layers, packing="cachemir_complex")
    if a.capture:
        from perseus.impl.magtape import save_rot_steps
        model.set_capture(a.capture)
        save_rot_steps(a.capture, inf.fhe.loaded_rot_steps)
    for b in range(model.n_layers):
        model.pcmm_weights(b)
    pcmm_engine(model.rt).prepare(dims.dim)
    ccmm_engine(model.rt).prepare(dims.dim)
    print(f"[e18-ts] twin ready {time.time() - t0:.0f}s (layers {model.n_layers})", flush=True)
    t1 = time.time()
    ref, kvs = exact_hidden_ts(model, ids, pub, E, model.n_layers)
    print(f"[e18-ts] float64 two-stream reference {time.time() - t1:.0f}s", flush=True)
    pages = pages_of(ids, pub, E, BIG, model.blocks.t)
    npub = len(pages) - 1
    kpos = [list(kp) for _t, _c, kp in pages]
    kpos[0] = kpos[0] + [0]                                       # page 0's key 127 = BOS (sinc.BOS_TOK)
    geom = Geometry(kpos, query_positions(pages[-1][2], len(ids) - 1))
    starts = np.cumsum([0] + [len([x for x in kp if x < BIG]) for _t, _c, kp in pages])

    def ref_rows(p, b):
        n = starts[p + 1] - starts[p]
        return ref[b][starts[p]:starts[p] + n]
    t2 = time.time()
    if a.store == "exact":          # the public pages from the float64 public stream, only the encrypted page runs
        rows, k = [], 1             # layout row 0 = BOS, then the public tokens in order
        for j, (_t, _c, kp) in enumerate(pages[:-1]):
            rp = [k + i if x < BIG else (k + n_real(kp) - 1 if PAD_COPY else None) for i, x in enumerate(kp)]
            k += sum(x < BIG for x in kp)
            rows.append(rp + ([0] if j == 0 else []))              # page 0's key 127 = BOS (layout row 0)
        pats = store_patterns(model, kvs, rows)
        geom.prep = prep_store
        print(f"[e18-ts] public store patterns from the float64 stream {time.time() - t2:.0f}s ({npub} pages x "
              f"{model.n_layers} layers)", flush=True)
        t2 = time.time()
        res = run_pages(model, pages[-1:], geom_page=npub, geom=geom, log=lambda m: print(m, flush=True),
                        ref_rows=lambda p, b: ref[b][starts[npub]:], stores=[[] for _ in range(model.n_layers)],
                        first=npub, store_pats=pats)
    else:
        res = run_pages(model, pages, geom_page=npub, geom=geom, ref_rows=ref_rows, log=lambda m: print(m, flush=True))
    print(f"[e18-ts] two-stream twin {time.time() - t2:.0f}s", flush=True)
    out = dict(idx=a.idx, n_tokens=len(ids), n_private=len(S), E=E, n_public=len(pub), layers=model.n_layers,
               store=a.store, bts_pages=[r[2] for r in res])
    hE = model.hidden_rows(res[-1][0], res[-1][1])[:len(E)]
    out["E_maxerr"] = float(np.abs(hE - ref[-1][starts[npub]:]).max())
    if model.n_layers == int(store.config["num_hidden_layers"]):
        lg = plain_logits(model, hE[-1:])[0]
        lr = plain_logits(model, ref[-1][-1:])[0]
        pz = np.exp(lr - lr.max()); pz /= pz.sum()
        qz = np.exp(lg - lg.max()); qz /= qz.sum()
        kl = float((pz * (np.log(pz + 1e-300) - np.log(qz + 1e-300))).sum())
        let = np.array(letters if letters else [0])
        out.update(kl=kl, argmax_twin=int(lg.argmax()), argmax_ref=int(lr.argmax()),
                   letter_twin=int(np.argmax(lg[let])), letter_ref=int(np.argmax(lr[let])), label=int(label))
        print(f"[e18-ts] readout: KL {kl:.3g}, argmax twin {lg.argmax()} ref {lr.argmax()}, answer letter twin "
              f"{out['letter_twin']} ref {out['letter_ref']} (label {label})", flush=True)
    if a.full:
        from perseus.impl.prefill import Chunks
        allp = list(range(1, len(ids)))
        fp = []
        i, p = 0, 0
        while i < len(allp):
            n = 127 if p == 0 else 128
            g = allp[i:i + n]
            pad = n - len(g)
            fp.append(([ids[j] for j in g] + [ids[g[-1]]] * pad, Chunks(n, model.blocks.t, base=g[0]), None))
            i += n
            p += 1
        t3 = time.time()
        rf = run_pages(model, fp, log=lambda m: print(m, flush=True))
        out["bts_full_pages"] = [r[2] for r in rf]
        print(f"[e18-ts] full prefill twin {time.time() - t3:.0f}s: bootstraps per page {out['bts_full_pages']}",
              flush=True)
    print(f"[e18-ts] SUMMARY {json.dumps(out)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    model.close()
    sess.close()


if __name__ == "__main__":
    main()
