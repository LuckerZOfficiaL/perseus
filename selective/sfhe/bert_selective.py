"""Selective (one-way) BERT on perseus.impl (E8): the ENCRYPTED stream only, [CLS] + the sensitive tokens + [SEP] of one
or several sentences packed on chunk 0 of one 128-token page, each sentence's attention also reading its own plaintext
stream's keys / values (the server's float64 BERT on the redacted sentence, selective/e8_reference.py) through the
route-B PCMMs of impl/dense_xpt.py. Everything else is bert_block unchanged.

A page: inputs a = 0.. at lanes [s_a, s_a + n_e_a) of chunk 0, n_e_a <= n_e_cap; the rows are the client's embeddings
(word + true position + type, embeddings LayerNorm) of the encrypted tokens; the LayerNorm's per-position scales read the
true positions (XChunks). The op stream depends on (number of inputs, n_e_cap) only: the layout, the positions and the
plaintext K / V enter as plaintext values and the per-page linears.
"""
from __future__ import annotations

import numpy as np

from perseus.impl import blocked as BL
from sfhe.dense_xpt import attention_dense_layout, plain_linears_layout
from perseus.impl.pcmm import run as pcmm_run
from perseus.impl.prefill import evict_block_masks, name_inputs

from perseus.examples.gpt2.prefill import _bias_blocks, _gelu_blocks
from sfhe.vit_selective import PAGE, XChunks, cyclic_pad, plain_shift

from perseus.examples.bert.model import BertPrefill, _tapped, affine_blocked


def bert_block_dense(rt, Xs, w, cfgs, ch, B, lin_qk, lin_pv, layout, n_e_cap, b):
    """bert_block for the one-page encrypted stream: route-B attention on the page layout."""
    ops, K = rt.ops, B.K
    ln1, ln2, sm, ge = cfgs
    with rt.step("qkv"):
        Q, Kc, V = [], [], []
        for g, x in enumerate(Xs):
            o = _bias_blocks(rt, pcmm_run(rt, x, w["qkv"]), w["qkv"], ch, g, B)
            Kc.append(o[:K]); V.append(o[K:2 * K]); Q.append(o[2 * K:])
    assert len(Q) == 1, "the encrypted stream is one page"
    with rt.step("attn"):
        O = attention_dense_layout(rt, Q[0], Kc[0], V[0], lin_qk, lin_pv, sm, layout, n_e_cap, B, layer=b)
    with rt.step("out_proj"):
        A = [_bias_blocks(rt, pcmm_run(rt, o, w["out"]), w["out"], ch, g, B) for g, o in enumerate(O)]
    Y = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(A, Xs)]
    with rt.step("ln_1"):
        X1 = affine_blocked(rt, BL.norm_blocked(rt, Y, ln1, ch, B), w["g1"], w["b1"], ch, B, f"bert.ln1.{b}")
    out = []
    for g, x in enumerate(X1):
        with rt.step("up"):
            H = _bias_blocks(rt, pcmm_run(rt, x, w["up"]), w["up"], ch, g, B)
        with rt.step("gelu"):
            G = _gelu_blocks(rt, H, ge)
        with rt.step("down"):
            out.append(_bias_blocks(rt, pcmm_run(rt, G, w["down"]), w["down"], ch, g, B))
    Y = [[ops.add(a, s) for a, s in zip(ra, rs)] for ra, rs in zip(out, X1)]
    with rt.step("ln_2"):
        return affine_blocked(rt, BL.norm_blocked(rt, Y, ln2, ch, B), w["g2"], w["b2"], ch, B, f"bert.ln2.{b}")


class BertSelectivePrefill(BertPrefill):
    route = "pcmm"                                       # route B (the planned driver asks)
    ws_fixed = None                                      # {name: ws} fixed weight scales (the device session)
    n_e_cap = 10                                         # encrypted tokens per input the program is recorded for

    def plain_linears_layout(self, b, K_list, V_list):
        ws = self.ws_fixed or {}
        return plain_linears_layout(K_list, V_list, self.blocks, b, ws_qk=ws.get(f"b{b}.xqk"), ws_pv=ws.get(f"b{b}.xpv"),
                                    shift_qk=plain_shift(f"b{b}.xqk"), shift_pv=plain_shift(f"b{b}.xpv"))

    def page(self, rows_list, pos_list, pkv_list, chunk0=None):
        """Pack inputs on the page: (rows (PAGE, 768), positions (PAGE,), layout, P_kv[b] = [(K, V) per input]).
        rows_list[a]: the client's embedded rows of input a's encrypted tokens, pos_list[a] their true positions,
        pkv_list[a][b] = (K, V) model-unit rows of its plaintext stream at layer b. chunk0: the number of inputs on chunk 0
        (default: all, layout entries (lane, n_e, n_p)); the others go to chunk 1 (entries (chunk, lane, n_e, n_p))."""
        t = self.blocks.t
        G = len(rows_list)
        c0 = G if chunk0 is None else int(chunk0)
        layout, used, lane = [], [], [0, 0]
        for a, (rows, pkv) in enumerate(zip(rows_list, pkv_list)):
            c = 0 if a < c0 else 1
            n_e = len(rows)
            assert n_e <= self.n_e_cap, f"{n_e} encrypted tokens, the program takes {self.n_e_cap}"
            assert lane[c] + n_e <= t, f"chunk {c}: {lane[c] + n_e} encrypted tokens do not fit {t} lanes"
            layout.append((c, lane[c], n_e, int(pkv[0][0].shape[0])) if c0 < G else (lane[c], n_e, int(pkv[0][0].shape[0])))
            used += [c * t + lane[c] + i for i in range(n_e)]
            lane[c] += n_e
        flat = np.concatenate([np.asarray(r, dtype=np.float64) for r in rows_list])
        pos = np.full(PAGE, PAGE - 1, dtype=np.int64)     # padded lanes: one fixed position (never chunk 0's tuple)
        pos[used] = np.concatenate([np.asarray(p) for p in pos_list])
        if c0 >= G:
            rows = cyclic_pad(flat, PAGE)
        else:                                           # free rows: copies of the used ones (no identically zero slot)
            rows = np.zeros((PAGE, flat.shape[1]))
            rows[used] = flat
            free = np.setdiff1d(np.arange(PAGE), used)
            rows[free] = flat[np.arange(len(free)) % len(flat)]
        P_kv = [[pk[b] for pk in pkv_list] for b in range(len(pkv_list[0]))]
        return rows, pos, tuple(layout), P_kv

    def first_rows(self, layout):
        """The page row of each input's first encrypted token (its CLS)."""
        t = self.blocks.t
        return [x[0] * t + x[1] if len(x) == 4 else x[0] for x in layout]

    @staticmethod
    def entry(x):
        """(n_e, n_p) of a layout entry."""
        return tuple(x[-2:])

    def prefill_selective(self, rows, pos, layout, P_kv, on_block=None):
        """One page (from `page`): returns (Xs, ch, record); input a's hidden rows are rows s_a .. s_a + n_e_a."""
        n = sum(x[-2] for x in layout)
        ch = XChunks(PAGE, self.blocks.t, bidir=True, keys=n, positions=tuple(int(p) for p in pos))
        self.last_ch = ch
        self.start()
        self.token_begin(0)
        Xs = BL.encode_blocked(self.rt, np.asarray(rows, dtype=np.float64), ch, self.blocks)
        self._sel = (layout, P_kv)
        st = self.selective_stages(ch)
        if on_block is not None:
            st = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(st)]
        Xs = self.run_stages(Xs, 0, st)
        return Xs, ch, self.token_end(0)

    def prefill_page(self, rows, pos, layout, P_kv, on_block=None):
        return self.prefill_selective(rows, pos, layout, P_kv, on_block)

    def selective_stages(self, ch):
        layout, P_kv = self._sel
        B = self.blocks

        def block_fn(b):
            def compute(Xs):
                K_list, V_list = (list(x) for x in zip(*P_kv[b]))
                lin_qk, lin_pv = self.plain_linears_layout(b, K_list, V_list)
                self.__dict__.setdefault("plain_lins", {})[b] = (lin_qk, lin_pv)
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, [x for row in Xs for x in row])
                with self.rt.step(f"blk{b}"):
                    Xs = bert_block_dense(self.rt, Xs, self.weights(b), self.cfgs.block(b), ch, B, lin_qk, lin_pv,
                                          layout, self.n_e_cap, b)
                self.exit_stage(b, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        return [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]
