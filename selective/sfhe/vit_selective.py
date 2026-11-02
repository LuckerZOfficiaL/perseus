"""Selective (one-way) ViT prefill on perseus.impl: the ENCRYPTED stream only, CLS + the sensitive patches on one
128-token page, whose attention at every layer also reads the plaintext stream's keys / values (the server's own
float64 ViT on the redacted image, perseus/examples/vit/.. selective/e3_reference.py in the fork) handed in as
blocked pages. Everything else is vit_block unchanged.

Model units -> attention pages (e3_probe_kv.py): a page holds q / k / v at scale 1/2 with page column c carrying
model column h * 64 + j for the interleaved feature r = j H + h, c = Blocks.head_perm()[r] (padded heads zero).
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

from perseus.impl import blocked as BL
from sfhe import ccmm_mha_xpt
from perseus.impl.pcmm import run as pcmm_run
from perseus.impl.prefill import Chunks, evict_block_masks, name_inputs

from perseus.examples.gpt2.prefill import _bias_blocks, _gelu_blocks, _tapped

from perseus.examples.vit import weights as V
from perseus.examples.vit.prefill import VitPrefill, add_rows, unscale_blocked

PAGE = 128


@dataclasses.dataclass(frozen=True)
class XChunks(Chunks):
    """Chunks whose lanes carry arbitrary absolute positions (the LayerNorm's per-position scales read them)."""
    positions: tuple = ()

    def lanes(self, k):
        return [self.positions[i] for i in self.idx(k)]


def cyclic_pad(rows, full):
    n = len(rows)
    if n >= full:
        return rows[:full]
    reps = -(-full // n)
    return np.concatenate([rows] * reps, axis=0)[:full]


def pack_two_chunks(rows_list, pos_list, t, chunk0):
    """The page rows / positions of inputs packed on both chunks: the first chunk0 inputs on chunk 0, the others on
    chunk 1, each from lane 0 on; free rows copy the used ones (no identically zero slot), free positions are PAGE - 1.
    Returns (rows (PAGE, d), positions (PAGE,), [(chunk, first lane, n_e)])."""
    ents, used, lane = [], [], [0, 0]
    for a, r in enumerate(rows_list):
        c = 0 if a < chunk0 else 1
        n_e = len(r)
        assert lane[c] + n_e <= t, f"chunk {c}: {lane[c] + n_e} encrypted tokens do not fit {t} lanes"
        ents.append((c, lane[c], n_e))
        used += [c * t + lane[c] + i for i in range(n_e)]
        lane[c] += n_e
    flat = np.concatenate([np.asarray(r, dtype=np.float64) for r in rows_list])
    rows = np.zeros((PAGE, flat.shape[1]))
    rows[used] = flat
    free = np.setdiff1d(np.arange(PAGE), used)
    rows[free] = flat[np.arange(len(free)) % len(flat)]
    pos = np.full(PAGE, PAGE - 1, dtype=np.int64)
    pos[used] = np.concatenate([np.asarray(p) for p in pos_list])
    return rows, pos, ents


def page_columns(B):
    """model column (h * 64 + j, or -1 for a padded head) of every page column."""
    hp = B.head_perm()
    inv = np.empty(B.d.hid, dtype=np.int64)
    inv[hp] = np.arange(B.d.hid)
    r = inv
    j, h = r // B.d.H, r % B.d.H
    return np.where(h < B.d.H_real, h * B.d.d_head + j, -1)


def to_page_rows(rows_model, B):
    """(n, 768) model-unit rows -> (n, hid) page rows at scale 1/2 in the head-grouped order."""
    cols = page_columns(B)
    out = np.zeros((rows_model.shape[0], B.d.hid))
    ok = cols >= 0
    out[:, ok] = 0.5 * np.asarray(rows_model, dtype=np.float64)[:, cols[ok]]
    return out


def plain_kv_sinc(rt, K_rows, V_rows, B):
    """The plaintext stream's K / V rows (n_p, 768) -> per page (K SinC vectors, V SinC vectors, keys in the page):
    exactly what ccmm_mha_paged.attention would make of encrypted pages, computed on the CPU twin's plaintext
    vectors (encode as blocked pages, split the halves, route to SinC, zero the padded value rows, dup for the
    merged CCMM) and read back as numpy. Call it outside the recorded stream."""
    from perseus.impl import ccmm as cm
    from perseus.impl import sinc as sc
    from perseus.impl import sinc_mha as smh
    from perseus.impl.ccmm_mha_paged import _key_rows
    from perseus.impl.prefill import split
    ops = rt.ops
    n_p = K_rows.shape[0]
    full = -(-n_p // PAGE) * PAGE
    ch_p = Chunks(full, B.t, bidir=True, keys=n_p)
    Kp = BL.encode_blocked(rt, cyclic_pad(to_page_rows(K_rows, B), full), ch_p, B)
    Vp = BL.encode_blocked(rt, cyclic_pad(to_page_rows(V_rows, B), full), ch_p, B)
    pages = []
    for g in range(ch_p.n_ct):
        kk = min(PAGE, n_p - PAGE * g)
        kj = smh.to_sinc_kv(rt, [list(split(ops, x)) for x in Kp[g]])
        vj = smh.to_sinc_kv(rt, [list(split(ops, x)) for x in Vp[g]])
        if kk < PAGE:
            vj = [rt.mult_mask(x, ("smhp.vrows", kk), lambda kk=kk: _key_rows(kk)) for x in vj]
        if cm.MERGE:
            kj, vj = sc.dup(rt, kj), sc.dup(rt, vj)
        vec = lambda x: np.zeros(sc.SLOTS, dtype=np.complex128) if x is None else np.asarray(ops.decrypt_slots_complex(x))
        pages.append(([vec(x) for x in kj], [vec(x) for x in vj], kk))
    return pages, ch_p


def vit_block_xpt(rt, Xs, w, cfgs, ch, B, pages, n_p, layer=0, img=0):
    """vit_block for the one-page encrypted stream, its attention reading the plaintext pages too."""
    ops, K = rt.ops, B.K
    ln1, ln2, sm, ge = cfgs
    skip = Xs
    with rt.step("ln_1"):
        N1 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln1, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift1"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N1)]
    with rt.step("qkv"):
        Q, Kc, Vv = [], [], []
        for g, y in enumerate(Y):
            o = _bias_blocks(rt, pcmm_run(rt, y, w["qkv"]), w["qkv"], ch, g, B)
            Kc.append(o[:K]); Vv.append(o[K:2 * K]); Q.append(o[2 * K:])
    assert len(Q) == 1, "the encrypted stream is one page"
    with rt.step("attn"):
        O = ccmm_mha_xpt.attention_xpt(rt, Q[0], Kc[0], Vv[0], pages, sm, ch.n_keys, n_p, layer=layer,
                                       h_real=B.d.H_real, img=img)
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
    Xs = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, skip)]
    skip = Xs
    with rt.step("ln_2"):
        N2 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln2, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift2"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N2)]
    out = []
    for g, y in enumerate(Y):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, y, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    return [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, skip)]


def vit_block_dense_xpt(rt, Xs, w, cfgs, ch, B, lin_qk, lin_pv, n_p, layer=0, groups=1, layout=None, n_e_cap=None):
    """vit_block for the one-page encrypted stream, route B: own keys by dense lane offsets, plaintext keys by PCMM.
    groups > 1 (E7): ch.n_keys = groups * n_e tokens of `groups` inputs on the page, lin_qk / lin_pv batched.
    layout (E10): the inputs on both chunks (dense_xpt.attention_dense_layout entries (chunk, lane, n_e, n_p))."""
    from sfhe.dense_xpt import attention_dense_layout, attention_dense_xpt
    ops, K = rt.ops, B.K
    ln1, ln2, sm, ge = cfgs
    skip = Xs
    with rt.step("ln_1"):
        N1 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln1, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift1"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N1)]
    with rt.step("qkv"):
        Q, Kc, Vv = [], [], []
        for g, y in enumerate(Y):
            o = _bias_blocks(rt, pcmm_run(rt, y, w["qkv"]), w["qkv"], ch, g, B)
            Kc.append(o[:K]); Vv.append(o[K:2 * K]); Q.append(o[2 * K:])
    assert len(Q) == 1, "the encrypted stream is one page"
    with rt.step("attn"):
        if layout is not None:
            cap = n_e_cap or max(x[-2] for x in layout)
            O = attention_dense_layout(rt, Q[0], Kc[0], Vv[0], lin_qk, lin_pv, sm, layout, cap, B, layer=layer)
        else:
            O = attention_dense_xpt(rt, Q[0], Kc[0], Vv[0], lin_qk, lin_pv, sm, ch.n_keys // groups, n_p, B, layer=layer,
                                    groups=groups)
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
    Xs = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, skip)]
    skip = Xs
    with rt.step("ln_2"):
        N2 = BL.norm_blocked(rt, unscale_blocked(rt, Xs, B), ln2, ch, B)
        Y = [[BL.shift_blocked(rt, n, w["shift2"], ch, g, k, B, 2.0) for k, n in enumerate(row)]
             for g, row in enumerate(N2)]
    out = []
    for g, y in enumerate(Y):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, y, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    return [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, skip)]


def plain_shift(name):
    """The band shift of a per-image linear from the PCMM calibration when it has one (the device), else 0 (fake)."""
    from perseus.impl.pcmm import band_shift
    try:
        return band_shift(name)
    except KeyError:
        return 0


class SelectivePrefill(VitPrefill):
    route = os.environ.get("SEL_ROUTE", "pcmm")          # "sinc": plaintext pages through the CCMM; "pcmm": route B
    ws_fixed = None                                      # {name: ws} fixed weight scales (the device session)

    def plain_linears(self, b, K_rows, V_rows):
        return self.plain_linears_batch(b, [K_rows], [V_rows])

    groups = 1                                           # E7: inputs sharing the encrypted page
    chunk0 = None                                        # E10: inputs on chunk 0 when the others go to chunk 1
    layout = None                                        # E10: the recorded page's layout entries (both chunks)

    def hidden(self, Xs, ch):
        """VitPrefill.hidden; with both chunks in use, the inputs' rows in input order (input a = rows a n_e ..)."""
        if self.layout is None:
            return super().hidden(Xs, ch)
        h = super(VitPrefill, self).hidden(Xs, ch)
        return np.concatenate([h[r:r + x[-2]] for r, x in zip(self.first_rows(self.layout), self.layout)])

    def first_rows(self, layout):
        """The page row of each input's first encrypted token (its CLS)."""
        t = self.blocks.t
        return [x[0] * t + x[1] if len(x) == 4 else x[0] for x in layout]

    n_e_cap = None                                       # E11: own-key offsets cover this many tokens (None: the page's max)
    _page_mode = False                                   # E11: the stream came from prefill_page (sizes vary per input)

    def plain_linears_layout(self, b, K_list, V_list):
        from sfhe.dense_xpt import plain_linears_layout
        ws = self.ws_fixed or {}
        return plain_linears_layout(K_list, V_list, self.blocks, b, ws_qk=ws.get(f"b{b}.xqk"), ws_pv=ws.get(f"b{b}.xpv"),
                                    shift_qk=plain_shift(f"b{b}.xqk"), shift_pv=plain_shift(f"b{b}.xpv"))

    def page(self, rows_list, pos_list, pkv_list, chunk0=None):
        """E11: inputs of different sizes on one page, BertSelectivePrefill.page's contract: rows_list[a] the pixel rows of
        input a's [CLS; private patches], pos_list[a] their token positions, pkv_list[a][b] = (K, V) its plaintext stream at
        layer b; chunk0 inputs on chunk 0 (default all), the rest on chunk 1. Returns (rows, positions, layout, P_kv)."""
        G = len(rows_list)
        c0 = G if chunk0 is None else int(chunk0)
        rows, pos, ents = pack_two_chunks(rows_list, pos_list, self.blocks.t, c0)
        n_p = [int(pk[0][0].shape[0]) for pk in pkv_list]
        layout = tuple((c, s_, n_e, n_p[a]) if c0 < G else (s_, n_e, n_p[a]) for a, (c, s_, n_e) in enumerate(ents))
        return rows, pos, layout, [[pk[b] for pk in pkv_list] for b in range(len(pkv_list[0]))]

    def prefill_page(self, rows, pos, layout, P_kv, on_block=None, img=0):
        """One page from `page`; hidden() then gives the inputs' rows in input order."""
        self.layout = tuple(tuple(x) for x in layout)
        self._page_mode = True
        n = sum(x[-2] for x in self.layout)
        pos = tuple(int(p) for p in pos)
        return self._run_page(np.asarray(rows, dtype=np.float64), pos, n, P_kv, on_block, img, len(self.layout))

    def two_chunks(self):
        return self.groups > 1 and self.chunk0 is not None and self.chunk0 < self.groups

    def page_rows(self, rows_e, positions):
        """The page rows / positions / layout entries (chunk, lane, n_e) of the batched inputs on both chunks."""
        n_e = len(rows_e) // self.groups
        split_ = lambda x: [x[a * n_e:(a + 1) * n_e] for a in range(self.groups)]
        return pack_two_chunks(split_(np.asarray(rows_e)), split_(np.asarray(positions)), self.blocks.t, self.chunk0)

    def plain_linears_batch(self, b, K_list, V_list):
        # the sparse encoder: bit-identical to plain_linears / plain_linears_batch (selective/e7_check_sparse.py), ~20x faster
        from sfhe.dense_xpt import plain_linears_sparse
        ws = self.ws_fixed or {}
        return plain_linears_sparse(K_list, V_list, self.blocks, b, ws_qk=ws.get(f"b{b}.xqk"), ws_pv=ws.get(f"b{b}.xpv"),
                                    shift_qk=plain_shift(f"b{b}.xqk"), shift_pv=plain_shift(f"b{b}.xpv"))

    def prefill_selective_batch(self, rows_list, pos_list, P_kv_list, on_block=None, img=0):
        """E7: several inputs' encrypted streams on one page. rows_list[a]: (n_e, 768) pixel rows of input a's [CLS;
        sensitive patches] (the same n_e for all), pos_list[a] its token positions, P_kv_list[a][b] = (K, V) its
        plaintext stream's rows at layer b (the same n_p for all). Lanes [a n_e, (a + 1) n_e) of chunk 0 are input a.
        Returns (Xs, ch, record); input a's hidden rows are rows a n_e .. of `hidden`."""
        n_e = len(rows_list[0])
        c0 = len(rows_list) if self.chunk0 is None else min(self.chunk0, len(rows_list))
        assert all(len(r) == n_e for r in rows_list) and max(c0, len(rows_list) - c0) * n_e <= self.blocks.t, \
            "inputs must fit their chunk"
        n_p = P_kv_list[0][0][0].shape[0]
        assert all(pk[b][0].shape[0] == n_p for pk in P_kv_list for b in range(len(pk))), "same plaintext keys per input"
        self.groups = len(rows_list)
        P_kv = [[pk[b] for pk in P_kv_list] for b in range(len(P_kv_list[0]))]
        try:
            return self.prefill_selective(np.concatenate(rows_list), np.concatenate(pos_list), P_kv, on_block, img)
        finally:
            self.groups = 1

    def prefill_selective(self, rows_e, positions, P_kv, on_block=None, img=0):
        """rows_e: (n_e, 768) pixel rows of [CLS; the sensitive patches]; positions: their token positions
        (0 for CLS, p + 1 for patch p); P_kv[b] = (K, V) model-unit rows of the plaintext tokens the encrypted
        stream reads at layer b. Returns (Xs, ch, record) like VitPrefill.prefill."""
        n = len(rows_e)
        full = -(-n // PAGE) * PAGE
        self.layout = None
        self._page_mode = False
        if self.two_chunks():
            rows, pos, ents = self.page_rows(rows_e, positions)
            pos = tuple(int(p) for p in pos)
            self.layout = tuple((c, s_, n_e, P_kv[0][a][0].shape[0]) for a, (c, s_, n_e) in enumerate(ents))
        else:
            rows = cyclic_pad(np.asarray(rows_e, dtype=np.float64), full)
            pos = tuple(int(p) for p in cyclic_pad(np.asarray(positions), full))
        return self._run_page(rows, pos, n, P_kv, on_block, img, self.groups)

    def _run_page(self, rows, pos, n, P_kv, on_block, img, groups):
        ch = XChunks(len(rows), self.blocks.t, bidir=True, keys=n, positions=pos)
        self.last_ch = ch
        self.start()
        self.token_begin(0)
        Xs = [x for row in BL.encode_blocked(self.rt, rows, ch, self.blocks) for x in row]
        d = self.rt.dims
        emb_off = (V.embed_offsets(self.store) * V.residual_scale(d.dim))[list(pos)]
        self._sel = (emb_off, P_kv, img, groups)
        stages = self.selective_stages(ch)
        if on_block is not None:
            stages = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(stages)]
        Xs = self.run_stages(Xs, 0, stages)
        return Xs, ch, self.token_end(0)

    def selective_stages(self, ch):
        """The stages of the encrypted stream; the per-image pieces (position rows, plaintext K/V, image tag) come
        from self._sel so the planned recorder can call this with the chunks alone."""
        emb_off, P_kv, img, groups = self._sel
        B = self.blocks

        def block_fn(b):
            def compute(Xs):
                if self._page_mode:
                    K_list, V_list = (list(x) for x in zip(*P_kv[b]))       # E11: each input its own plaintext keys
                    n_p = max(K.shape[0] for K in K_list)
                    lin_qk, lin_pv = self.plain_linears_layout(b, K_list, V_list)
                    self.__dict__.setdefault("plain_lins", {})[b] = (lin_qk, lin_pv)
                elif groups > 1:
                    assert self.route != "sinc", "E7 batching is route B only"
                    K_list, V_list = (list(x) for x in zip(*P_kv[b]))
                    n_p = K_list[0].shape[0]
                    lin_qk, lin_pv = self.plain_linears_batch(b, K_list, V_list)
                    self.__dict__.setdefault("plain_lins", {})[b] = (lin_qk, lin_pv)
                elif self.route == "sinc":
                    K_rows, V_rows = P_kv[b]
                    n_p = K_rows.shape[0]
                    pages, ch_p = plain_kv_sinc(self.rt, K_rows, V_rows, B)     # plaintext work, outside the stream
                else:
                    K_rows, V_rows = P_kv[b]
                    n_p = K_rows.shape[0]
                    lin_qk, lin_pv = self.plain_linears(b, K_rows, V_rows)
                    self.__dict__.setdefault("plain_lins", {})[b] = (lin_qk, lin_pv)
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, Xs)
                with self.rt.step(f"blk{b}"):
                    Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                    if b == 0:
                        with self.rt.step("embed"):
                            Xg = [pcmm_run(self.rt, x, self.embed_weights()) for x in Xg]
                            Xg = add_rows(self.rt, Xg, emb_off, ch, B, ("vit.emb.sel", tuple(ch.positions)))
                    if self.route == "sinc":
                        Xg = vit_block_xpt(self.rt, Xg, self.diag_weights(b), self.cfgs.block(b), ch, B, pages,
                                           n_p, layer=b, img=img)
                    else:
                        Xg = vit_block_dense_xpt(self.rt, Xg, self.diag_weights(b), self.cfgs.block(b), ch, B,
                                                 lin_qk, lin_pv, n_p, layer=b, groups=groups, layout=self.layout,
                                                 n_e_cap=self.n_e_cap)
                    Xs = [x for row in Xg for x in row]
                self.exit_stage(b, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        return [(f"blk{b}", f"b{b}.", block_fn(b)) for b in range(self.n_layers)]
