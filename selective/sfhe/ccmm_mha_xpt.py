"""Selective (one-way) attention on the paged CCMM: ONE encrypted query page (CLS + the sensitive tokens, n_e real
tokens) against its own encrypted keys / values and the plaintext keys / values of the n_p non-sensitive tokens.

The plaintext pages arrive already in the CCMM's SinC operand layout (examples/vit/selective.plain_kv_sinc computes
them on the CPU twin, outside the recorded stream) and are injected as plaintexts onto zero ciphertexts, so the
encrypted stream spends nothing converting them; the Q.K^T and P.V products against them are still ciphertext
products (a PCMM route for those is the later speed-up).

Same softmax (softmax_pages: row sums over every key page, kc = n_e + n_p on real query rows) and P.V accumulation
as ccmm_mha_paged.attention; only the pairing changes: the query page meets len(pages) + 1 key pages.
"""
from __future__ import annotations

import math

import numpy as np

from perseus.impl import ccmm as cm
from perseus.impl import sinc as sc
from perseus.impl import sinc_mha as smh
from perseus.impl import sinc_softmax as ss
from perseus.impl.ccmm_mha import KMAX, QMAX, SHIFT_PV, SHIFT_QK, UNPACK_FOLD, VMAX, _halves
from perseus.impl.ccmm_mha_paged import PAGE, _acc, _key_rows, softmax_pages


def attention_xpt(rt, Qe, Ke, Ve, pages, cfg, n_e, n_p, layer=0, h_real=12, img=0):
    """Qe, Ke, Ve: the encrypted page's blocked complex ciphertexts (one per 512-channel block, scale 1/2), n_e real
    tokens on its first lanes; pages[j] = (K SinC vectors, V SinC vectors, keys in the page), the plaintext pages
    (n_p keys in all); img tags the per-image plaintexts. Returns [O] (one blocked page, scale 1)."""
    ops = rt.ops
    eng = cm.engine(rt)
    eng.prepare(rt.dims.dim)
    nb = len(Qe)
    gscale = 1.0 / math.sqrt(smh.DH)
    merge = cm.MERGE
    al = ss.score_affine(cfg)[0]
    n = n_e + n_p
    with rt.step("qkt"):
        Kr = [list(_halves(ops, x)) for x in Ke]
        Vr = [list(_halves(ops, x)) for x in Ve]
        kj, vj = smh.to_sinc_kv(rt, Kr), smh.to_sinc_kv(rt, Vr)
        if n_e < PAGE:                  # padded keys: their value rows zeroed (P's stand-ins meet them)
            vj = [rt.mult_mask(x, ("smhp.vrows", n_e), lambda: _key_rows(n_e)) for x in vj]
        if merge:
            kj, vj = sc.dup(rt, kj), sc.dup(rt, vj)
        Ks, Vs, kks = [kj], [vj], [n_e]
        for j, (kv, vv, kk) in enumerate(pages, start=1):
            Ks.append([rt.add_mask(ops.sub(x, x), ("xpt.k", img, layer, j, o), lambda v=v: v)
                       for o, (x, v) in enumerate(zip(kj, kv))])
            Vs.append([rt.add_mask(ops.sub(x, x), ("xpt.v", img, layer, j, o), lambda v=v: v)
                       for o, (x, v) in enumerate(zip(vj, vv))])
            kks.append(kk)
        q = smh.to_sinc_q(rt, Qe)
        Qs = sc.dup(rt, q) if merge else q
        S = []
        for j in range(len(Ks)):
            s = eng.apply([ops.copy(x) for x in Qs], [ops.copy(x) for x in Ks[j]], b_transposed=True,
                          amax=QMAX, bmax=KMAX, shift=SHIFT_QK, out_scale=gscale, site=f"xqk{layer}.0.{j}",
                          merge=merge)
            if merge:
                s = cm.unmerge(rt, s, eng.w, scale=al)
            S.append(ss.pack(rt, s))
    tq, tk = smh.packed_meta(n_e, h_real)
    qv = tq >= 0
    valids = [[qv & (tk[o] < kk) for o in range(sc.NCT)] for kk in kks]
    kc = np.where(qv, n, 1)
    tag = ("mhax", n_e, n, h_real, len(Ks))
    with rt.step("softmax"):
        P = softmax_pages(rt, S, cfg, valids, kc, tag, layer=layer, prescaled=merge)
        if not UNPACK_FOLD:
            P = [ss.unpack(rt, p) for p in P]
    with rt.step("softmax_v"):
        O = None
        for j in range(len(Ks)):
            pj = sc.dup(rt, P[j]) if merge else P[j]
            O = _acc(rt, O, eng.apply(pj, [ops.copy(x) for x in Vs[j]], b_transposed=False, amax=1.5, bmax=VMAX,
                                      shift=SHIFT_PV, site=f"xpv{layer}.0.{j}", merge=merge))
        if merge:
            O = cm.unmerge(rt, O, eng.w)
    return [smh.from_sinc_q(rt, O, nb)]
