"""late-inject-replace on perseus.impl: the cheap mode of selective encryption. The server ran the plain ViT for the
first L - k layers with the private patches dropped from attention; the client's page holds the private patches'
pixel rows (zero elsewhere), encrypted. The hand-over stage computes the private tokens' raw embeddings under
encryption (block 0's patch-embedding PCMM plus the position offsets at the private rows) and adds the ordinary
tokens' plain residual rows as plaintext; the shipped last k blocks then run on all 197 tokens."""
from __future__ import annotations

import numpy as np

from perseus.impl import blocked as BL
from perseus.impl.pcmm import run as pcmm_run
from perseus.impl.prefill import Chunks, evict_block_masks, name_inputs

from perseus.examples.gpt2.prefill import _tapped

from perseus.examples.vit import weights as V
from perseus.examples.vit.prefill import VitPrefill, add_rows, vit_block


class LatePrefill(VitPrefill):
    def prefill_late(self, rows_page, S, hidden_plain, k, on_block=None, img=0):
        """rows_page: (197, 768) pixel rows, zero outside the private rows; S: private patch indices; hidden_plain:
        (197, 768) the plain model's residual rows entering layer L - k (model units; the private rows are ignored);
        k: encrypted last layers. Returns (Xs, ch, record) like VitPrefill.prefill."""
        rows = np.asarray(rows_page, dtype=np.float64)
        n = len(rows)
        full = -(-n // 128) * 128
        if full > n:
            rows = np.vstack([rows, rows[n - (full - n):]])            # VitPrefill.prefill's pad
        ch = Chunks(len(rows), self.blocks.t, bidir=True, keys=n)
        self.last_ch = ch
        self.start()
        self.token_begin(0)
        Xs = [x for row in BL.encode_blocked(self.rt, rows, ch, self.blocks) for x in row]
        d = self.rt.dims
        r = V.residual_scale(d.dim)
        tokS = np.asarray(sorted(int(s) for s in S), dtype=np.int64) + 1
        isS = np.zeros(n, dtype=bool)
        isS[tokS] = True
        off = np.where(isS[:, None], V.embed_offsets(self.store) * r, 0.0)
        hid = np.where(isS[:, None], 0.0, np.asarray(hidden_plain, dtype=np.float64) * r)
        self._late = (off, hid, k, img)
        stages = self.late_stages_bound(ch)
        if on_block is not None:
            stages = [(lab, pre, _tapped(fn, i, on_block)) for i, (lab, pre, fn) in enumerate(stages)]
        Xs = self.run_stages(Xs, 0, stages)
        return Xs, ch, self.token_end(0)

    def late_stages_bound(self, ch):
        """The stages with the per-image pieces taken from self._late (the planned recorder calls this with the
        chunks alone)."""
        return self.late_stages(ch, *self._late)

    def late_stages(self, ch, off, hid, k, img=0):
        B, L = self.blocks, self.n_layers

        def handover(Xs):
            cap = self.enter_stage(0, 0)
            name_inputs(self.inf, Xs)
            with self.rt.step("handover"):
                Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                with self.rt.step("embed"):
                    Xg = [pcmm_run(self.rt, x, self.embed_weights()) for x in Xg]
                    Xg = add_rows(self.rt, Xg, off, ch, B, ("vit.late.off", tuple(np.flatnonzero(off.any(1)))))
                with self.rt.step("plain"):
                    Xg = add_rows(self.rt, Xg, hid, ch, B, ("vit.late.hid", img))
                Xs = [x for row in Xg for x in row]
            self.exit_stage(0, cap)
            evict_block_masks(self.rt)
            return Xs

        def block_fn(b, i):
            def compute(Xs):
                cap = self.enter_stage(i, 0)
                name_inputs(self.inf, Xs)
                with self.rt.step(f"blk{b}"):
                    Xg = [Xs[g * B.K:(g + 1) * B.K] for g in range(len(Xs) // B.K)]
                    Xg = vit_block(self.rt, Xg, self.diag_weights(b), self.cfgs.block(b), ch, B, layer=b)
                    Xs = [x for row in Xg for x in row]
                self.exit_stage(i, cap)
                evict_block_masks(self.rt)
                return Xs
            return compute
        return [("handover", "h.", handover)] + \
            [(f"blk{b}", f"b{b}.", block_fn(b, i)) for i, b in enumerate(range(L - k, L), start=1)]
