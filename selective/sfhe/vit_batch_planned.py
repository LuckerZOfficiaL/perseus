"""E7: several inputs' encrypted two-stream tokens on ONE page, planned on the device (route B only).

A "page image" is `groups` inputs side by side: rows = their [CLS; sensitive patches] pixel rows concatenated (lanes
a n_e .. of chunk 0 are input a), positions likewise, and per layer b the list of every input's plaintext (K, V) rows.
The recorded stream is SelectivePrefill.prefill_selective with model.groups set, so everything else (recording,
compiling, the route-B fast path that rebuilds the per-image linears without the twin) is SelectivePlannedPrefill's,
with the linears stacked over the inputs (dense_xpt.plain_linears_batch).
"""
from __future__ import annotations

import math
import time

import numpy as np

from sfhe.vit_selective_planned import SelectivePlannedPrefill


def page_pkv(pkv_list):
    """[input][layer] (K, V) -> [layer][input] (K, V), the form prefill_selective reads when model.groups > 1."""
    return [[pk[b] for pk in pkv_list] for b in range(len(pkv_list[0]))]


class BatchSelectivePlannedPrefill(SelectivePlannedPrefill):
    def __init__(self, model, plan_dir, config, keys, *, positions, groups, chunk0=None, **kw):
        assert model.route != "sinc", "E7 batching is route B only"
        super().__init__(model, plan_dir, config, keys, positions=np.concatenate([positions] * groups), **kw)
        self.groups = int(groups)
        model.groups = self.groups                       # every recording of this driver is a batched page
        model.chunk0 = chunk0                            # E10: inputs past chunk0 on chunk 1

    def _rows_in(self, rows):
        if self.model.two_chunks():
            return self.model.page_rows(np.asarray(rows, dtype=np.float64), self.positions)[0]
        return super()._rows_in(rows)

    def hidden(self, b=None):
        """The inputs' rows in input order (both chunks: input a = rows a n_e ..)."""
        if self.model.layout is None:
            return super().hidden(b)
        from perseus.examples.bert.planned import EncoderPlannedPrefill
        h = EncoderPlannedPrefill.hidden(self, b)
        t = self.model.blocks.t
        return np.concatenate([h[c * t + s_:c * t + s_ + n_e] for c, s_, n_e, _p in self.model.layout])

    def fix_ws(self, probes, margin=1.5):
        """SelectivePlannedPrefill.fix_ws over page probes [(rows, page_pkv)]: max |W| over every input of every probe
        (the stacked weight's max is its inputs' max)."""
        from sfhe.dense_xpt import xqk_weights, xpv_weights
        from perseus.impl.pcmm import weight_bits
        wbits = weight_bits()
        nb = (wbits + 7) // 8
        top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
        mx = {}
        for _rows, pkv in probes:
            for b, per_input in enumerate(pkv):
                for K_rows, V_rows in per_input:
                    for name, W in ((f"b{b}.xqk", xqk_weights(K_rows, self.model.blocks)),
                                    (f"b{b}.xpv", xpv_weights(V_rows, self.model.blocks))):
                        mx[name] = max(mx.get(name, 0.0), float(np.abs(W).max()))
        ws = {k: 2.0 ** math.floor(math.log2(top / (margin * v))) for k, v in mx.items()}
        self.model.ws_fixed = ws
        return ws

    def stage_image(self, rows, pkv, img):
        """Route B fast path with the stacked linears; the twin path (SelectivePlannedPrefill) otherwise."""
        t0 = time.time()
        if any(self.dyn.values()):
            return super().stage_image(rows, pkv, img)
        self.set_image(pkv, img)
        self.model.plain_lins = {b: self.model.plain_linears_batch(b, [k for k, _ in per], [v for _, v in per])
                                 for b, per in enumerate(pkv)}
        return self.upload_plain(), time.time() - t0
