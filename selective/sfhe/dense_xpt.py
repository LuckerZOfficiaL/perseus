"""Selective (one-way) attention, route B: the plaintext half as PCMMs. One encrypted page (n_e real tokens on
chunk 0 of the blocked layout) attends its own n_e encrypted keys with the dense lane-offset products of
prefill_attention (per head group, unpaired) and n_p plaintext keys through two plaintext-ciphertext matmuls whose
weights are the plaintext stream's K and V of the layer:

  scores   S_x = Q_page . W_qk,  W_qk = gscale * blockdiag_h(K_h^T): output block 3k + m holds, at channel
           blk * 8 + h_l (head 8k + h_l) and lane = query, the score of key 64 m + blk, i.e. the score layout of the
           dense path (slot blk * tH + h * t + lane), so the own and plaintext scores share one softmax
           (softmax_x: prefill_softmax with explicit validity and n_e + n_p keys per lane);
  values   O_x = P_blocks . W_pv,  W_pv = blockdiag_h(V_h): back to the page layout, added to the own P.V.

A PCMM output is a fresh refresh (one StC-first finish per output block); nothing else of the plaintext half is
bootstrapped. The two linears are per image (their weights are the image's plaintext K/V); on the device they are
re-uploaded per image under fixed names (b<l>.xqk, b<l>.xpv) with a fixed weight scale (ws) and band shift.
"""
from __future__ import annotations

import math
import os

import numpy as np

from perseus.impl import smcheb
from perseus.impl.attention import _recip_rounds
from perseus.impl.blocked import sub_rt
from perseus.impl.pcmm import PcmmLinear, Shape, digit_planes, run as pcmm_run, weight_bits
from perseus.impl.poly import eval_chebyshev, im_cleanse, rotsum
from perseus.impl.prefill import _KcExtrapolated, join, lane_rotate, recip_lanes, split

MB = int(os.environ.get("XPT_MB", "3"))   # key blocks of 64 per head group for the plaintext keys (ViT 3: 192 >= 180; BERT 2)


def gscale(B):
    return 1.0 / math.sqrt(B.d.d_head_real)


def page_columns(B):
    """model column (h * 64 + j, or -1 for a padded head) of every page column."""
    hp = B.head_perm()
    inv = np.empty(B.d.hid, dtype=np.int64)
    inv[hp] = np.arange(B.d.hid)
    j, h = inv // B.d.H, inv % B.d.H
    return np.where(h < B.d.H_real, h * B.d.d_head + j, -1)


def encode_fixed(W, t, ws, shift, name, wbits=None):
    """PcmmLinear.encode with a given weight scale `ws` (None: the scale encode would pick), no bias."""
    sh = Shape(t)
    wbits = wbits or weight_bits()
    W = np.asarray(W, dtype=np.float64)
    n_in, n_out = W.shape[0] // sh.nch, W.shape[1] // sh.nch
    assert W.shape == (n_in * sh.nch, n_out * sh.nch), "W must be whole nch blocks"
    nb = (wbits + 7) // 8
    if ws is None:
        top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
        ws = 2.0 ** math.floor(math.log2(top / max(np.abs(W).max(), 1e-30)))
    Wt = np.rint(W.T * ws).astype(np.int64)
    perm = sh.ch_of_row()
    Wi = np.empty_like(Wt)
    for o in range(n_out):
        for c in range(n_in):
            blk = Wt[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch]
            Wi[o * sh.nch:(o + 1) * sh.nch, c * sh.nch:(c + 1) * sh.nch] = blk[perm[:, None], perm[None, :]]
    return PcmmLinear(n_in, n_out, digit_planes(Wi, nb), float(ws), int(shift), None, name, int(t))


def xqk_weights(K_rows, B):
    """W (hid, MB * hid) of the plaintext scores PCMM: page column (block k, channel j * 8 + h_l) -> output block
    MB k + m, channel blk * 8 + h_l = gscale * K[64 m + blk, head 8k + h_l, dim j]."""
    hg, dh, nch = B.hg, B.d.d_head, B.nch
    n_p = K_rows.shape[0]
    W = np.zeros((B.d.hid, MB * B.d.hid))
    g = gscale(B)
    for k in range(B.K):
        for hl in range(hg):
            h = k * hg + hl
            if h >= B.d.H_real:
                continue
            Kh = K_rows[:, h * dh:(h + 1) * dh]                     # (n_p, 64)
            rows = k * nch + np.arange(dh) * hg + hl                 # page columns of head h, dims j
            for m in range(MB):
                u0, u1 = 64 * m, min(64 * (m + 1), n_p)
                if u0 >= n_p:
                    break
                cols = (MB * k + m) * nch + np.arange(u1 - u0) * hg + hl
                W[np.ix_(rows, cols)] = g * Kh[u0:u1].T
    return W


def xpv_weights(V_rows, B):
    """W (MB * hid, hid) of the plaintext values PCMM: score block MB k + m, channel blk * 8 + h_l -> page column
    (block k, channel j * 8 + h_l) = V[64 m + blk, head 8k + h_l, dim j]."""
    hg, dh, nch = B.hg, B.d.d_head, B.nch
    n_p = V_rows.shape[0]
    W = np.zeros((MB * B.d.hid, B.d.hid))
    for k in range(B.K):
        for hl in range(hg):
            h = k * hg + hl
            if h >= B.d.H_real:
                continue
            Vh = V_rows[:, h * dh:(h + 1) * dh]
            cols = k * nch + np.arange(dh) * hg + hl
            for m in range(MB):
                u0, u1 = 64 * m, min(64 * (m + 1), n_p)
                if u0 >= n_p:
                    break
                rows = (MB * k + m) * nch + np.arange(u1 - u0) * hg + hl
                W[np.ix_(rows, cols)] = Vh[u0:u1]
    return W


# ── sparse fast path: the xqk / xpv weights are block-sparse with a fixed pattern; write the PCMM digit planes
#    straight at the nonzeros (bit-identical to encode_fixed of the dense matrices, checked in selective/e7_check_sparse.py)

_COORDS = {}


def _coords_qk(n_p, B):
    """(rows, cols, key u, model feature f) of xqk_weights' nonzeros for one input: W[rows, cols] = gscale * K[u, f]."""
    key = ("qk", n_p, B.K, B.hg, B.d.d_head, B.nch, B.d.H_real, B.d.hid)
    if key not in _COORDS:
        hg, dh, nch = B.hg, B.d.d_head, B.nch
        R, C, U, F = [], [], [], []
        for k in range(B.K):
            for hl in range(hg):
                h = k * hg + hl
                if h >= B.d.H_real:
                    continue
                for m in range(MB):
                    u0, u1 = 64 * m, min(64 * (m + 1), n_p)
                    if u0 >= n_p:
                        break
                    j, u = np.meshgrid(np.arange(dh), np.arange(u0, u1), indexing="ij")
                    R.append(k * nch + j * hg + hl)
                    C.append((MB * k + m) * nch + (u - u0) * hg + hl)
                    U.append(u)
                    F.append(h * dh + j)
        _COORDS[key] = tuple(np.concatenate([x.ravel() for x in A]) for A in (R, C, U, F))
    return _COORDS[key]


def _coords_pv(n_p, B):
    """(rows, cols, key u, model feature f) of xpv_weights' nonzeros for one input: W[rows, cols] = V[u, f]."""
    key = ("pv", n_p, B.K, B.hg, B.d.d_head, B.nch, B.d.H_real, B.d.hid)
    if key not in _COORDS:
        hg, dh, nch = B.hg, B.d.d_head, B.nch
        R, C, U, F = [], [], [], []
        for k in range(B.K):
            for hl in range(hg):
                h = k * hg + hl
                if h >= B.d.H_real:
                    continue
                for m in range(MB):
                    u0, u1 = 64 * m, min(64 * (m + 1), n_p)
                    if u0 >= n_p:
                        break
                    u, j = np.meshgrid(np.arange(u0, u1), np.arange(dh), indexing="ij")
                    R.append((MB * k + m) * nch + (u - u0) * hg + hl)
                    C.append(k * nch + j * hg + hl)
                    U.append(u)
                    F.append(h * dh + j)
        _COORDS[key] = tuple(np.concatenate([x.ravel() for x in A]) for A in (R, C, U, F))
    return _COORDS[key]


def encode_sparse(n_rows, n_cols, rows, cols, vals, t, ws, shift, name, wbits=None):
    """encode_fixed of the (n_rows, n_cols) matrix that is `vals` at (rows, cols) and zero elsewhere, without building
    it: the digit planes are written at the permuted, transposed positions of the nonzeros only."""
    sh = Shape(t)
    wbits = wbits or weight_bits()
    nch = sh.nch
    n_in, n_out = n_rows // nch, n_cols // nch
    assert n_rows == n_in * nch and n_cols == n_out * nch, "W must be whole nch blocks"
    nb = (wbits + 7) // 8
    vals = np.asarray(vals, dtype=np.float64)
    if ws is None:
        top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
        ws = 2.0 ** math.floor(math.log2(top / max(np.abs(vals).max() if vals.size else 0.0, 1e-30)))
    iv = np.rint(vals * ws).astype(np.int64)
    inv = np.argsort(sh.ch_of_row())
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    Ri = (cols // nch) * nch + inv[cols % nch]          # Wi row  <- W column (output channel)
    Ci = (rows // nch) * nch + inv[rows % nch]          # Wi col  <- W row (input channel)
    planes = np.zeros((nb, n_out * nch, n_in * nch), dtype=np.int8)
    v = iv.copy()
    for u in range(nb):
        r = ((v + 128) % 256) - 128
        planes[u, Ri, Ci] = r
        v = (v - r) // 256
    if np.any(v):
        raise ValueError("encode_sparse: the matrix needs more planes")
    return PcmmLinear(n_in, n_out, planes, float(ws), int(shift), None, name, int(t))


def plain_linears_sparse(K_list, V_list, B, layer, ws_qk=None, ws_pv=None, shift_qk=0, shift_pv=0):
    """plain_linears_batch (one input or several) through encode_sparse: same PcmmLinears, far less work."""
    n_p = K_list[0].shape[0]
    nb_in = len(K_list)
    g = gscale(B)
    rq, cq, uq, fq = _coords_qk(n_p, B)
    rp, cp, up, fp = _coords_pv(n_p, B)
    hid = B.d.hid
    qk = encode_sparse(nb_in * hid, MB * hid, np.concatenate([rq + a * hid for a in range(nb_in)]),
                       np.tile(cq, nb_in), np.concatenate([g * np.asarray(K)[uq, fq] for K in K_list]),
                       B.t, ws_qk, shift_qk, f"b{layer}.xqk")
    pv = encode_sparse(nb_in * MB * hid, hid, np.concatenate([rp + a * MB * hid for a in range(nb_in)]),
                       np.tile(cp, nb_in), np.concatenate([np.asarray(Vr)[up, fp] for Vr in V_list]),
                       B.t, ws_pv, shift_pv, f"b{layer}.xpv")
    return qk, pv


def xqk_weights_batch(K_list, B):
    """E7: several inputs on one page (lane group a = input a). The scores PCMM reads the group-masked page copies
    stacked input-major ([Q_0 blocks, Q_1 blocks, ...]), so its weight stacks each input's xqk_weights vertically;
    each lane only meets its own input's keys and the outputs stay MB * K blocks."""
    return np.vstack([xqk_weights(K, B) for K in K_list])


def xpv_weights_batch(V_list, B):
    """E7: the values PCMM reads the group-masked score blocks stacked input-major; each input's xpv_weights stacked."""
    return np.vstack([xpv_weights(Vr, B) for Vr in V_list])


def plain_linears_batch(K_list, V_list, B, layer, ws_qk=None, ws_pv=None, shift_qk=0, shift_pv=0):
    return (encode_fixed(xqk_weights_batch(K_list, B), B.t, ws_qk, shift_qk, f"b{layer}.xqk"),
            encode_fixed(xpv_weights_batch(V_list, B), B.t, ws_pv, shift_pv, f"b{layer}.xpv"))


def plain_linears(K_rows, V_rows, B, layer, ws_qk=None, ws_pv=None, shift_qk=0, shift_pv=0):
    return (encode_fixed(xqk_weights(K_rows, B), B.t, ws_qk, shift_qk, f"b{layer}.xqk"),
            encode_fixed(xpv_weights(V_rows, B), B.t, ws_pv, shift_pv, f"b{layer}.xpv"))


# ── validity in the score layout (slot = blk * tH + h * t + lane) ──

def _coords(d):
    i = np.arange(d.N)
    return i // d.tH, (i % d.tH) // d.t, i % d.t


def pad_on():
    """XPT_PAD=1: the free (padding) lanes of the page are queries of the plaintext keys of the last input on their chunk
    (never keys of the real inputs), so their values stay near those of an ordinary token. Off, they attend to nothing and
    drift out of range over the layers (a padding overflow corrupted a whole page in the night of 10-29).
    XPT_PAD=2: in addition the padding lanes are one group of their own for the own-key products (each padding lane a
    query of the other padding lanes, within the own-key offsets), i.e. an ordinary extra input made of copied rows:
    with plaintext keys only (=1) they still drifted out of a LayerNorm band on one face page (E12)."""
    return os.environ.get("XPT_PAD", "0") not in ("", "0")


def pad_self():
    return os.environ.get("XPT_PAD", "0") == "2"


def own_valid(d, n_e, e=None, groups=1):
    """Own scores: offset blk holds key (lane - blk) mod t; `e` restricts to one offset. groups > 1 (E7): lanes
    [a n_e, (a + 1) n_e) are input a's tokens, and a query only sees keys of its own input."""
    blk, h, lane = _coords(d)
    end = groups * n_e
    if groups == 1:
        v = (h < d.H_real) & (lane < n_e) & (((lane - blk) % d.t) < n_e)
    else:
        key = (lane - blk) % d.t
        v = (h < d.H_real) & (lane < end) & (key < end) & (key // n_e == lane // n_e)
    if pad_self():                                   # the padding lanes: one group of their own
        key = (lane - blk) % d.t
        v = v | ((h < d.H_real) & (lane >= end) & (key >= end) & (((lane - key) % d.t) == blk))
    return v & (blk == e) if e is not None else v


def plain_valid(d, n_e, n_p, m, groups=1):
    blk, h, lane = _coords(d)
    q = np.ones_like(lane, dtype=bool) if pad_on() else (lane < groups * n_e)
    return (h < d.H_real) & q & (64 * m + blk < n_p)


def group_mask(N, t, n_e, a, last=False):
    """1 on the slots whose lane (slot mod t, in the page and the score layouts alike) belongs to input a (the last input
    also takes the padding lanes when XPT_PAD is on)."""
    lane = np.arange(N) % t
    hi = t if (last and pad_on()) else (a + 1) * n_e
    return ((lane >= a * n_e) & (lane < hi)).astype(np.float64)


def softmax_x(rt, S, cfg, valids, kcs, tag):
    """prefill_softmax over score ciphertexts S[m] with explicit validity valids[m] and keys per lane kcs."""
    ops, d = rt.ops, rt.dims
    mean = (cfg.clip_hi + cfg.clip_lo) / 2.0
    sf = 2.0 ** (-cfg.log2delta1 - cfg.log2delta2)
    lane_scale = np.array([0.5 / kcs[i] if i < len(kcs) else 0.0 for i in range(d.t)])
    Z = []
    for m, (s, valid) in enumerate(zip(S, valids)):
        s = rt.add_mask(s, ("xd.sm.score", m, cfg.clip_lo, mean) + tag,
                        lambda valid=valid: np.where(valid, -mean, cfg.clip_lo - mean))
        z = eval_chebyshev(ops, s, cfg.cheb_coeffs, cfg.cheb_a / sf, cfg.cheb_b / sf)
        for _ in range(cfg.log2delta1):
            z = ops.square(z)
        z = im_cleanse(ops, z)
        Z.append(rt.mult_mask(z, ("xd.sm.active", m) + tag,
                              lambda valid=valid: np.where(valid, lane_scale[np.arange(d.N) % d.t], 0.0)))
    lean = ops.sm_gs_first
    cheb = getattr(ops, "sm_cheb_gs", False)
    per_lane = [smcheb.rounds(cfg, kc) if cheb else _recip_rounds(_KcExtrapolated(cfg), kc, lean) for kc in kcs]
    Y = Z
    for k in range(len(per_lane[0])):
        if k:
            Z = [ops.square(im_cleanse(ops, y)) for y in Y]
        D = Z[0] if len(Z) == 1 else ops.add(Z[0], Z[1])
        for z in Z[2:]:
            ops.inplace_add(D, z)
        D = rotsum(ops, D, d.tH, d.N)
        D = ops.tag_reduce(D, d.tH)
        rnd = [pl[k] for pl in per_lane]
        iters = rnd[0][2]
        a_vals = [r[0] * r[3] for r in rnd]
        b_vals = [r[1] * r[3] * r[3] for r in rnd]
        cv = None if not cheb else ([[r[4][0][j] for r in rnd] for j in range(iters - 1)], [r[4][1] for r in rnd])
        R = recip_lanes(rt, D, a_vals, b_vals, iters,
                        (tuple(round(v, 12) for v in a_vals), tuple(round(v, 15) for v in b_vals)) + tag, cv)
        Y = [ops.mult(z, R) for z in Z]
    return Y


def attention_dense_xpt(rt, Qe, Ke, Ve, lin_qk, lin_pv, cfg, n_e, n_p, B, layer=0, groups=1):
    """Qe, Ke, Ve: the encrypted page's blocked complex ciphertexts (one per block, scale 1/2, n_e real tokens on
    chunk 0); lin_qk, lin_pv: this image's plaintext-key linears (plain_linears). Returns [O] (one blocked page,
    scale 1; chunk 1 a copy of chunk 0).
    groups > 1 (E7): `groups` inputs of n_e tokens each on chunk 0 (lanes a n_e ..), lin_qk / lin_pv the batched
    linears (plain_linears_batch); the PCMM inputs are masked per input so each lane meets its own input's plaintext
    keys, with the same PCMM outputs (refreshes) as one input."""
    ops = rt.ops
    nb = len(Qe)
    g = gscale(B)
    N, t = rt.dims.N, B.t
    with rt.step("xqk"):
        if groups == 1:
            q_in = Qe
        else:
            q_in = [rt.mult_mask(Qe[k], ("xd.qgrp", n_e, a, pad_on()),
                                 lambda a=a: group_mask(N, t, n_e, a, a == groups - 1))
                    for a in range(groups) for k in range(nb)]
        Sx = [im_cleanse(ops, s) for s in pcmm_run(rt, q_in, lin_qk)]    # 2 Re: chunk-0 queries, true scores
    es = sorted({(i - j) % B.t for i in range(n_e) for j in range(n_e)})  # the offsets reaching a real key
    # (groups > 1: a query and its own-input keys are < n_e lanes apart, so the same offsets; own_valid masks the rest)
    own, Pbs = [], []
    for k in range(nb):
        sub = sub_rt(rt, B, k)
        d = sub.dims
        if B.group_dims(k).H_real == 0:
            own.append(None)
            Pbs += [None] * MB
            continue
        with rt.step("qkt"):
            for X in (Qe[k], Ke[k], Ve[k]):
                ops.bootstrap_hint(X, ops.headroom(2), True)
            Q0, _ = split(ops, Qe[k])
            K0, _ = split(ops, Ke[k])
            V0, _ = split(ops, Ve[k])
            S0 = None
            for e in es:
                kc = lane_rotate(sub, K0, e, "xd.krot")
                s = rotsum(ops, ops.mult(Q0, kc), d.tH, d.N)
                v = sub.mult_mask(s, ("xd.qkt", n_e, e) + ((groups,) if groups > 1 else ()),
                                  lambda e=e: np.where(own_valid(d, n_e, e, groups), g, 0.0))
                S0 = v if S0 is None else ops.add(S0, v)
        S = [S0] + [Sx[MB * k + m] for m in range(MB)]
        valids = [own_valid(d, n_e, groups=groups)] + [plain_valid(d, n_e, n_p, m, groups) for m in range(MB)]
        n_pad = t - groups * n_e
        kcs = [n_e + n_p] * (groups * n_e) + ([n_p + (min(n_pad, n_e) if pad_self() else 0)] * n_pad if pad_on() else [])
        with rt.step("softmax"):
            P = softmax_x(sub, S, cfg, valids, kcs, ("xd", n_e, n_p, layer, k) + ((groups,) if groups > 1 else ())
                          + (("pad",) if pad_on() else ()))
        with rt.step("softmax_v"):
            acc = None
            for e in es:
                pb = sub.mult_mask(P[0], ("xd.bsel", e), lambda e=e: np.where(np.arange(d.N) // d.tH == e, 0.5, 0.0))
                pb = rotsum(ops, pb, d.tH, d.N)
                prod = ops.mult(pb, lane_rotate(sub, V0, e, "xd.vrot"))
                acc = prod if acc is None else ops.add(acc, prod)
            own.append(im_cleanse(ops, acc))
        Pbs += P[1:]
    with rt.step("xpv"):
        Pin = [p if p is not None else ops.sub(Qe[0], Qe[0]) for p in Pbs]
        if groups > 1:
            Pin = [rt.mult_mask(p, ("xd.pgrp", n_e, a, pad_on()), lambda a=a: group_mask(N, t, n_e, a, a == groups - 1))
                   for a in range(groups) for p in Pin]
        Ox = pcmm_run(rt, Pin, lin_pv)
    O = [ops.add(own[k], Ox[k]) if own[k] is not None else Ox[k] for k in range(nb)]
    return [[join(ops, o, o) for o in O]]


# ── E8: inputs of different sizes on one page. layout = [(first lane, n_e, n_p)] per input, packed on chunk 0; input a
#    reads its own n_e_a encrypted keys and n_p_a plaintext keys. The op stream depends on (number of inputs, n_e_cap)
#    only, so pages of different layouts share one recorded program; the layout enters through plaintext values. ──

def _lane_group(layout, t):
    """lane -> input (own-key products); XPT_PAD=2: the padding lanes past the last input -> a group of their own."""
    g = np.full(t, -1, dtype=np.int64)
    for a, (s, n_e, _n_p) in enumerate(layout):
        g[s:s + n_e] = a
    if pad_self():
        g[max(s + n_e for s, n_e, _p in layout):] = len(layout)
    return g


def layout_own_valid(d, layout, e=None):
    """own_valid for a layout: a query sees the keys of its own input (offset blk holds key (lane - blk) mod t)."""
    blk, h, lane = _coords(d)
    g = _lane_group(layout, d.t)
    gl, gk = g[lane], g[(lane - blk) % d.t]
    v = (h < d.H_real) & (gl >= 0) & (gl == gk)
    return v & (blk == e) if e is not None else v


def _plain_group(layout, t):
    """lane -> the input whose plaintext keys it queries (XPT_PAD: the padding lanes past the last input -> the last)."""
    g = _lane_group(layout, t)
    if pad_on():
        end = max(s + n_e for s, n_e, _p in layout)
        g[end:] = len(layout) - 1
    return g


def layout_plain_valid(d, layout, m):
    blk, h, lane = _coords(d)
    g = _plain_group(layout, d.t)
    n_p = np.array([x[2] for x in layout] + [0])[g[lane]]       # lanes of no input (g = -1): no keys
    return (h < d.H_real) & (g[lane] >= 0) & (64 * m + blk < n_p)


def layout_mask(N, t, s, n_e, last=False):
    lane = np.arange(N) % t
    hi = t if (last and pad_on()) else s + n_e
    return ((lane >= s) & (lane < hi)).astype(np.float64)


def layout_kcs(layout, n_e_cap):
    """keys per lane over the used lanes (lanes of no input: the capacity, any positive count does)."""
    used = max(s + n_e for s, n_e, _p in layout)
    kcs = [n_e_cap + MB * 64] * used
    for s, n_e, n_p in layout:
        kcs[s:s + n_e] = [n_e + n_p] * n_e
    if pad_on():                                     # padding lanes: the last input's plaintext keys (+ each other)
        n_pad = 64 - used
        kcs += [layout[-1][2] + (min(n_pad, n_e_cap) if pad_self() else 0)] * n_pad
    return kcs


def plain_linears_layout(K_list, V_list, B, layer, ws_qk=None, ws_pv=None, shift_qk=0, shift_pv=0):
    """plain_linears_sparse with each input's own number of plaintext keys (E8); one input = plain_linears_sparse."""
    hid, g = B.d.hid, gscale(B)
    q, p = ([], [], []), ([], [], [])
    for a, (K, Vr) in enumerate(zip(K_list, V_list)):
        K, Vr = np.asarray(K), np.asarray(Vr)
        assert 0 < K.shape[0] <= MB * 64, f"input {a}: {K.shape[0]} plaintext keys, the page takes 1..{MB * 64}"
        rq, cq, uq, fq = _coords_qk(K.shape[0], B)
        rp, cp, up, fp = _coords_pv(Vr.shape[0], B)
        for acc, vals in ((q, (rq + a * hid, cq, g * K[uq, fq])), (p, (rp + a * MB * hid, cp, Vr[up, fp]))):
            for lst, v in zip(acc, vals):
                lst.append(v)
    n = len(K_list)
    qk = encode_sparse(n * hid, MB * hid, *(np.concatenate(x) for x in q), B.t, ws_qk, shift_qk, f"b{layer}.xqk")
    pv = encode_sparse(n * MB * hid, hid, *(np.concatenate(x) for x in p), B.t, ws_pv, shift_pv, f"b{layer}.xpv")
    return qk, pv


def attention_dense_layout(rt, Qe, Ke, Ve, lin_qk, lin_pv, cfg, layout, n_e_cap, B, layer=0):
    """attention_dense_xpt for a page layout (E8): lin_qk / lin_pv from plain_linears_layout. With several inputs the
    PCMM inputs are masked per input (as groups > 1) so each lane meets its own input's plaintext keys; the own-key
    offsets cover n_e_cap tokens whatever the layout."""
    ops = rt.ops
    nb = len(Qe)
    g = gscale(B)
    N, t = rt.dims.N, B.t
    lay = tuple(tuple(int(v) for v in x) for x in layout)
    if any(len(x) == 4 and x[0] == 1 for x in lay):
        return _attention_two_chunks(rt, Qe, Ke, Ve, lin_qk, lin_pv, cfg, lay, n_e_cap, B, layer)
    lay = tuple(x[1:] if len(x) == 4 else x for x in lay)
    assert all(n_e <= n_e_cap for _s, n_e, _p in lay) and max(s + n_e for s, n_e, _p in lay) <= t
    multi = len(lay) > 1
    with rt.step("xqk"):
        q_in = Qe if not multi else [rt.mult_mask(Qe[k], ("xl.qgrp", s, n_e, a == len(lay) - 1 and pad_on()),
                                                  lambda s=s, n_e=n_e, a=a: layout_mask(N, t, s, n_e, a == len(lay) - 1))
                                     for a, (s, n_e, _p) in enumerate(lay) for k in range(nb)]
        Sx = [im_cleanse(ops, s) for s in pcmm_run(rt, q_in, lin_qk)]
    es = sorted({(i - j) % t for i in range(n_e_cap) for j in range(n_e_cap)})
    own, Pbs = [], []
    for k in range(nb):
        sub = sub_rt(rt, B, k)
        d = sub.dims
        if B.group_dims(k).H_real == 0:
            own.append(None)
            Pbs += [None] * MB
            continue
        with rt.step("qkt"):
            for X in (Qe[k], Ke[k], Ve[k]):
                ops.bootstrap_hint(X, ops.headroom(2), True)
            Q0, _ = split(ops, Qe[k])
            K0, _ = split(ops, Ke[k])
            V0, _ = split(ops, Ve[k])
            S0 = None
            for e in es:
                kc = lane_rotate(sub, K0, e, "xd.krot")
                s = rotsum(ops, ops.mult(Q0, kc), d.tH, d.N)
                v = sub.mult_mask(s, ("xl.qkt", lay, e), lambda e=e: np.where(layout_own_valid(d, lay, e), g, 0.0))
                S0 = v if S0 is None else ops.add(S0, v)
        S = [S0] + [Sx[MB * k + m] for m in range(MB)]
        valids = [layout_own_valid(d, lay)] + [layout_plain_valid(d, lay, m) for m in range(MB)]
        with rt.step("softmax"):
            P = softmax_x(sub, S, cfg, valids, layout_kcs(lay, n_e_cap), ("xl", lay, layer, k) + (("pad",) if pad_on() else ()))
        with rt.step("softmax_v"):
            acc = None
            for e in es:
                pb = sub.mult_mask(P[0], ("xd.bsel", e), lambda e=e: np.where(np.arange(d.N) // d.tH == e, 0.5, 0.0))
                pb = rotsum(ops, pb, d.tH, d.N)
                prod = ops.mult(pb, lane_rotate(sub, V0, e, "xd.vrot"))
                acc = prod if acc is None else ops.add(acc, prod)
            own.append(im_cleanse(ops, acc))
        Pbs += P[1:]
    with rt.step("xpv"):
        Pin = [p if p is not None else ops.sub(Qe[0], Qe[0]) for p in Pbs]
        if multi:
            Pin = [rt.mult_mask(p, ("xl.pgrp", s, n_e, a == len(lay) - 1 and pad_on()),
                                lambda s=s, n_e=n_e, a=a: layout_mask(N, t, s, n_e, a == len(lay) - 1))
                   for a, (s, n_e, _p) in enumerate(lay) for p in Pin]
        Ox = pcmm_run(rt, Pin, lin_pv)
    O = [ops.add(own[k], Ox[k]) if own[k] is not None else Ox[k] for k in range(nb)]
    return [[join(ops, o, o) for o in O]]


def _attention_two_chunks(rt, Qe, Ke, Ve, lin_qk, lin_pv, cfg, lay, n_e_cap, B, layer):
    """attention_dense_layout with inputs on both chunks (layout entries (chunk, first lane, n_e, n_p); the PCMM order is
    every chunk-0 input, then every chunk-1 input). The page splits into its real halves (2 Re = chunk 0, 2 Im = chunk 1);
    each half's inputs are masked (x 1/2, the page scale) into the scores PCMM, chunk 1's multiplied by i, so the
    PCMM's real part carries chunk 0's scores and its imaginary part chunk 1's (same outputs, same refreshes). Each
    half then runs its own-key products and softmax, and the values PCMM gathers both halves back into one page."""
    ops = rt.ops
    nb = len(Qe)
    g = gscale(B)
    N, t = rt.dims.N, B.t
    lays = [tuple(x[1:] for x in lay if x[0] == c) for c in (0, 1)]
    for lc in lays:
        assert lc and all(n_e <= n_e_cap for _s, n_e, _p in lc) and max(s + n_e for s, n_e, _p in lc) <= t
    halves = {}
    for k in range(nb):
        if B.group_dims(k).H_real == 0:
            continue
        for X in (Qe[k], Ke[k], Ve[k]):
            ops.bootstrap_hint(X, ops.headroom(2), True)
        halves[k] = [split(ops, X) for X in (Qe[k], Ke[k], Ve[k])]       # [(Q0, Q1), (K0, K1), (V0, V1)]
    with rt.step("xqk"):
        q_in = []
        for c in (0, 1):
            for a, (s_, n_e, _p) in enumerate(lays[c]):
                last = a == len(lays[c]) - 1
                for k in range(nb):
                    h = halves[k][0][c] if k in halves else split(ops, Qe[k])[c]
                    q = rt.mult_mask(h, ("xl2.q", c, s_, n_e, last and pad_on()),
                                     lambda s_=s_, n_e=n_e, last=last: 0.5 * layout_mask(N, t, s_, n_e, last))
                    q_in.append(ops.mult_i(q) if c else q)
        Sx = [split(ops, o) for o in pcmm_run(rt, q_in, lin_qk)]          # (2 Re, 2 Im): chunk 0 / chunk 1 scores
    es = sorted({(i - j) % t for i in range(n_e_cap) for j in range(n_e_cap)})
    own = {0: [], 1: []}
    Pbs = {0: [], 1: []}
    for k in range(nb):
        sub = sub_rt(rt, B, k)
        d = sub.dims
        if k not in halves:
            for c in (0, 1):
                own[c].append(None)
                Pbs[c] += [None] * MB
            continue
        (Q0, Q1), (K0, K1), (V0, V1) = halves[k]
        for c, (Qc, Kc, Vc) in enumerate(((Q0, K0, V0), (Q1, K1, V1))):
            lc = lays[c]
            with rt.step("qkt"):
                S0 = None
                for e in es:
                    kr = lane_rotate(sub, Kc, e, "xd.krot")
                    s = rotsum(ops, ops.mult(Qc, kr), d.tH, d.N)
                    v = sub.mult_mask(s, ("xl2.qkt", c, lc, e), lambda e=e, lc=lc: np.where(layout_own_valid(d, lc, e), g, 0.0))
                    S0 = v if S0 is None else ops.add(S0, v)
            S = [S0] + [Sx[MB * k + m][c] for m in range(MB)]
            valids = [layout_own_valid(d, lc)] + [layout_plain_valid(d, lc, m) for m in range(MB)]
            with rt.step("softmax"):
                P = softmax_x(sub, S, cfg, valids, layout_kcs(lc, n_e_cap), ("xl2", c, lc, layer, k)
                              + (("pad",) if pad_on() else ()))
            with rt.step("softmax_v"):
                acc = None
                for e in es:
                    pb = sub.mult_mask(P[0], ("xd.bsel", e), lambda e=e: np.where(np.arange(d.N) // d.tH == e, 0.5, 0.0))
                    pb = rotsum(ops, pb, d.tH, d.N)
                    prod = ops.mult(pb, lane_rotate(sub, Vc, e, "xd.vrot"))
                    acc = prod if acc is None else ops.add(acc, prod)
                own[c].append(im_cleanse(ops, acc))
            Pbs[c] += P[1:]
    with rt.step("xpv"):
        zero = None
        Pin = []
        for c in (0, 1):
            for a, (s_, n_e, _p) in enumerate(lays[c]):
                last = a == len(lays[c]) - 1
                for p in Pbs[c]:
                    if p is None:
                        zero = zero if zero is not None else ops.sub(Qe[0], Qe[0])
                        Pin.append(zero)
                        continue
                    q = rt.mult_mask(p, ("xl2.p", c, s_, n_e, last and pad_on()),
                                     lambda s_=s_, n_e=n_e, last=last: layout_mask(N, t, s_, n_e, last))
                    Pin.append(ops.mult_i(q) if c else q)
        Ox = pcmm_run(rt, Pin, lin_pv)                                    # chunk 0's values + i chunk 1's
    O = []
    for k in range(nb):
        if own[0][k] is None:
            O.append(Ox[k])
        else:
            O.append(ops.add(join(ops, own[0][k], own[1][k]), Ox[k]))
    return [O]
