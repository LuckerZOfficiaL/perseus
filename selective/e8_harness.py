#!/usr/bin/env python
"""E8 harness: the BERT fake session and model of `perseus.examples.bert.run_bert --fake`, as a function, for the
selective (two-stream) BERT. Same environment as the shipped plan (bert_sst2_128's runtime line, its PCMM calibration),
so the twin, the capture and the planned device run agree. No GPU is touched (the fake session is numpy).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))                   # this checkout's perseus, before any installed one
import sfhe                                                        # noqa: E402,F401  (runtime patches)
from sfhe import runtime as RT                                     # noqa: E402

RT.apply_pins()
SHIPPED = "bootstrap_placements/bert_sst2_128"
# BERT_CONFIGS: another approximation config (e.g. sfhe/configs/bert_base_n32_t128_sst2: LayerNorm bands refitted on SST-2 by
# e8_ln_bands.py from upstream's config; re-derive it after a pull)
CONFIGS = os.environ.get("BERT_CONFIGS", "configs/model/approximation/bert_base_n32_t128/configs.json")


def setup(layers=12, capture=None, plan=None, weights=None, n_e_cap=10, structure_only=False):
    """plan: the plan whose runtime line to use (default: the shipped BERT plan); weights: a perseus export dir of the
    same architecture (BERT_EXPORT, e.g. the E1b fine-tuned model)."""
    os.chdir(REPO)
    os.environ.setdefault("XPT_MB", "1")                     # SST-2: <= 53 plaintext keys, one key block of 64
    os.environ.setdefault("KV_ARENA_GB", "0")
    if weights:
        os.environ["BERT_EXPORT"] = os.path.expanduser(weights)
    from perseus._paths import resolve
    from perseus.examples.bert.run_bert import ARM, FAST
    from perseus.examples.gpt2 import env
    plan = resolve(plan or SHIPPED)
    over = {**{k: os.environ.get(k, v) for k, v in FAST.items()}, **ARM, **env.plan_runtime(plan, "n32"),
            "GPT2_PACKING": "cachemir", **RT.PINS}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    cal = os.path.join(plan, "pcmm_calib.json")
    if os.path.exists(cal):
        os.environ.setdefault("PF_PCMM_CALIB_IN", cal)
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    if capture:
        os.environ["FHE_GRAPH_DIR"] = capture
    from perseus.impl import config, fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from perseus.examples.bert import data as D
    from sfhe.bert_selective import BertSelectivePrefill
    d = Dims.gpt2()
    extra = prefill_rot_steps(d) | diag_rot_steps(d) | pcmm_rot_steps() | \
        blocked_rot_steps(Blocks(d, int(os.environ.get("PF_NCH", "512"))), ladder_radix_bits())
    ccmm_on = os.environ.get("PF_ATTN", "") == "ccmm"
    if ccmm_on:
        from perseus.impl.sinc_mha import rot_steps as mha_rot_steps
        extra |= mha_rot_steps() | {16, d.N - 16}
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d, ladder_radix_bits()) if ladder_radix_bits() >= 2 else set())
    _, inf = fakecap.make_session(d, rot)
    sess = fakecap.FakeSession(inf)
    st, cl = D.store(), D.client()
    cfgs = config.load_configs(CONFIGS)
    model = BertSelectivePrefill(inf, st, cfgs, core=fakecap.core, n_layers=layers)
    model.n_e_cap = int(n_e_cap)
    if capture:
        model.set_capture(capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(capture, inf.fhe.loaded_rot_steps)
        if structure_only:
            inf.fhe.magnitude_suppressed = True
    model.preload()
    from perseus.impl.pcmm import engine
    engine(model.rt).prepare(d.dim)
    if ccmm_on:
        from perseus.impl.ccmm import engine as ccmm_engine
        ccmm_engine(model.rt).prepare(d.dim)
    return SimpleNamespace(model=model, store=st, cl=cl, cfgs=cfgs, inf=inf, sess=sess, B=model.blocks, D=D, d=d,
                           rt=model.rt, ccmm=ccmm_on, plan=plan)


def load_private(path, split="validation"):
    """{index: [private token positions]} of a detector file (e11_ner_sst2.py)."""
    import json
    return {int(k): v for k, v in json.load(open(os.path.expanduser(path)))[split].items()}


def sentence_inputs(D, st, cl, idx, frac=0.3, seed=0, k_max=8, n_layers=12, placeholder="mask", split="validation",
                    private=None):
    """private: {index: positions} from a detector (E11) instead of the random span."""
    """[dict] per SST-2 index of `split` (calibration pages: "train"): ids, n, label, S (a span of round(frac * content) tokens, 1..k_max, at a
    seeded random start), the oracle (e8_reference.two_stream) and the full model's logits."""
    import numpy as np
    import e8_reference as R
    sents = D.sentences(max(idx) + 1, split)
    out = []
    for i in idx:
        s, lab = sents[i]
        ids, n = D.encode(s, int(os.environ.get("SFHE_BERT_T", "128")))   # SFHE_BERT_T: longer texts (E14)
        if private is not None:
            S = [int(p) for p in private[i] if 0 < int(p) < n - 1]
        else:
            rng = np.random.default_rng(seed * 1000003 + i)
            L = n - 2
            k = int(min(k_max, max(1, round(frac * L))))
            s0 = int(rng.integers(0, L - k + 1))
            S = list(range(1 + s0, 1 + s0 + k))
        ref = R.two_stream(st, cl, ids, n, S, placeholder, n_layers)
        rows = D.embed(ids, cl)
        full = D.head(D.exact_blocks(st, rows, n, n_layers)[-1][0], cl) if n_layers == 12 else None
        out.append(dict(i=i, ids=ids, n=n, label=lab, S=S, ref=ref, rows_e=rows[ref["E_idx"]], pos=ref["E_idx"],
                        full=full))
    return out


def pack_pages(ins, G, chunk0, t=64):
    """Group inputs into pages of G (chunk0 on chunk 0, the rest on chunk 1) whose encrypted tokens fit each chunk's t
    lanes: first fit in order, keeping one lane free for every slot still empty (so a filler always fits); an input
    that fits no open page starts a new one. Empty slots get fillers (not scored): repeats of the smallest inputs, cut
    to the free lanes (at least the CLS row: a valid input of fewer private tokens). Returns [(inputs, real count)]."""
    c1 = G - chunk0
    caps = (chunk0, c1)
    lanes = lambda xs: sum(len(y["pos"]) for y in xs)
    pages = []                                                   # [chunk-0 inputs, chunk-1 inputs]
    for x in ins:
        n_e = len(x["pos"])
        placed = False
        for pg in pages:
            for c in (0, 1):
                left = caps[c] - len(pg[c]) - 1                  # slots still empty after this one
                if left >= 0 and lanes(pg[c]) + n_e + left <= t:
                    pg[c].append(x)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            pages.append([[x], []])
    small = sorted(ins, key=lambda y: len(y["pos"]))
    out = []
    for pg in pages:
        real = len(pg[0]) + len(pg[1])
        for c in (0, 1):
            j = 0
            while len(pg[c]) < caps[c]:
                y = small[j % len(small)]
                j += 1
                room = t - lanes(pg[c]) - (caps[c] - len(pg[c]) - 1)
                m = max(1, min(len(y["pos"]), room))
                pg[c].append(dict(y, rows_e=y["rows_e"][:m], pos=y["pos"][:m], filler=True))
        out.append((pg[0] + pg[1], real))
    return out
