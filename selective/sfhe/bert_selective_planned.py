"""The selective BERT prefill PLANNED on the device (E8): `SelectivePlannedPrefill`'s per-image machinery over
`BertSelectivePrefill.prefill_selective`, one page of G sentences at a time.

One recorded stream serves every page with G inputs of at most n_e_cap encrypted tokens. What changes per page: block 0's
encrypted rows (the clients' embeddings), the 24 per-page linears (the sentences' plaintext K / V, built directly), and
the plaintext values that encode the page layout and the tokens' positions (validity masks, keys per lane, LayerNorm
position scales). Those are found by recording a second probe page (`find_dynamic`) and, per page, read off a re-record
of the page on the CPU twin (`page_plaintexts`, which can run in worker processes ahead of the device) and staged in place.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np

from sfhe.vit_selective_planned import SelectivePlannedPrefill


@contextlib.contextmanager
def _twin_values(on):
    """on=False: the twin records without ciphertext values (upstream's fakecap.CapCt.novec, one slot per ciphertext;
    ~2x faster and a fraction of a worker's memory): a page re-record needs the op stream and the plaintexts, never the
    message. Upstream's PCMM fake does not take one-slot inputs, so its outputs are stubbed (one zero slot) meanwhile."""
    if on:
        yield
        return
    from perseus.impl import fakecap, pcmm
    from perseus.impl.fake import FakeCt
    fake0, nv0 = pcmm.Pcmm._fake, fakecap.CapCt.novec

    def stub(eng, xr, lin):
        outs = []
        for m in range(lin.n_out):
            o = FakeCt(np.zeros(1, dtype=np.complex128), eng.land + 2 * pcmm._drop_of(lin, m), 1, xr[0].packing)
            if hasattr(xr[0], "ptag"):
                o.ptag = xr[m if m < lin.n_in else 0].ptag
            outs.append(o)
        return outs
    pcmm.Pcmm._fake, fakecap.CapCt.novec = stub, True
    try:
        with np.errstate(all="ignore"):                       # the one-slot values are meaningless (overflow, 0 / 0)
            yield
    finally:
        pcmm.Pcmm._fake, fakecap.CapCt.novec = fake0, nv0


class BertSelectivePlannedPrefill(SelectivePlannedPrefill):
    _stages_attr = "selective_stages"
    # the model gives page(...), prefill_page(rows, pos, layout, P_kv), plain_linears_layout(b, K_list, V_list):
    # BertSelectivePrefill, or the ViT SelectivePrefill (VitLayoutPlannedPrefill)

    def __init__(self, model, plan_dir, config, keys, *, page, **kw):
        rows, pos, layout, pkv = page
        super().__init__(model, plan_dir, config, keys, positions=pos, **kw)
        self.set_page(pos, layout, pkv)
        self.staged_shape = None

    def set_page(self, pos, layout, pkv):
        self.pos, self.layout, self._pkv = np.asarray(pos), tuple(layout), pkv

    def shape(self):
        return (self.layout, tuple(int(p) for p in self.pos))

    # ── the BERT pieces of the ViT driver ──
    def _rows_in(self, rows):
        return np.asarray(rows, dtype=np.float64)          # pages come padded (BertSelectivePrefill.page)

    def _forward(self, rows):
        return self.model.prefill_page(np.asarray(rows, dtype=np.float64), self.pos, self.layout, self._pkv)

    def linears(self):
        from perseus.impl.pcmm import PcmmLinear
        out = {}
        for b in range(self.n_layers):
            for lin in self.model.weights(b).values():
                if isinstance(lin, PcmmLinear):
                    out[lin.name] = lin
        for b, (lq, lp) in self.plain_lins().items():
            out[lq.name] = lq
            out[lp.name] = lp
        return out

    def hidden(self, b=None):
        from perseus.examples.bert.planned import EncoderPlannedPrefill
        return EncoderPlannedPrefill.hidden(self, b)       # all PAGE rows: input a is rows s_a .. s_a + n_e_a

    def fix_ws(self, pages, margin=1.5):
        """Fixed weight scales of the per-page linears from max |W| over every input of the pages [(rows, pos, layout,
        P_kv)] (W holds gscale * K and V entries)."""
        from sfhe.dense_xpt import gscale
        from perseus.impl.pcmm import weight_bits
        wbits = weight_bits()
        nb = (wbits + 7) // 8
        top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
        g = gscale(self.model.blocks)
        mx = {}
        for _rows, _pos, _lay, P_kv in pages:
            for b, per in enumerate(P_kv):
                for K, Vr in per:
                    mx[f"b{b}.xqk"] = max(mx.get(f"b{b}.xqk", 0.0), g * float(np.abs(K).max()))
                    mx[f"b{b}.xpv"] = max(mx.get(f"b{b}.xpv", 0.0), float(np.abs(Vr).max()))
        ws = {k: 2.0 ** math.floor(math.log2(top / (margin * v))) for k, v in mx.items()}
        self.model.ws_fixed = ws
        return ws

    # ── per-page plaintexts ──
    def record_page(self, page, values=True):
        """The page's recording on the twin: {block: BlockRec} (values=False: without ciphertext values, _twin_values)."""
        from perseus.impl.driver import Recorder
        rows, pos, layout, pkv = page
        self.model.rt.forget_masks()
        rec = Recorder(self.model, self.n_layers)
        self.set_page(pos, layout, pkv)
        with _twin_values(values):
            self.model.prefill_page(np.asarray(rows, dtype=np.float64), pos, layout, pkv)
        return rec.blocks

    def find_dynamic(self, page, page2=None, more=()):
        """The plaintexts that differ between the recorded page and another one: {block: {pt name}}; the two streams
        must otherwise bind (same ops, scalars and rotations). page2: a third page; dynamic plaintexts equal to each other
        on all the probe pages form one group (the same mask rebuilt in every block) and share one store slot
        (self.dyn_group: {(block, name): group}); every staged page is checked against the grouping. `more`: further
        probe pages for the grouping (value-free re-records; two probes can make masks coincide that differ on other
        layouts, E14)."""
        import hashlib
        from perseus.impl.driver import diff_block
        keep = (self.pos, self.layout, self._pkv)
        others = [self.record_page(pg) for pg in ([page] + ([page2] if page2 is not None else []))]
        others += [self.record_page(pg, values=False) for pg in more]
        self.set_page(*keep)
        self.dyn_names = {}
        for b, r in self.recs.items():
            for o in others:
                s_, k_, _z = diff_block(r, o[b])
                if s_ or k_:
                    raise RuntimeError(f"probe page block {b}: scalars / rotations change between pages: {sorted(s_) + sorted(k_)}")
                if set(r.pts) != set(o[b].pts):
                    raise RuntimeError(f"probe page block {b}: the plaintext names differ between pages")
            self.dyn_names[b] = {nm for nm, v in r.pts.items() if any(not np.array_equal(v, o[b].pts[nm]) for o in others)}
        self.dyn_group = {}
        if page2 is not None:
            h = lambda v: hashlib.blake2b(np.ascontiguousarray(v).tobytes(), digest_size=16).hexdigest()
            sig = {}
            for b in sorted(self.dyn_names):
                for nm in sorted(self.dyn_names[b]):
                    key = (h(self.recs[b].pts[nm]),) + tuple(h(o[b].pts[nm]) for o in others)
                    self.dyn_group[(b, nm)] = sig.setdefault(key, len(sig))
        self.lean_twin = self._lean_binds(page, others[0])
        self.set_page(*keep)
        return self.dyn_names

    def _lean_binds(self, page, full):
        """Whether the value-free twin re-records `page` exactly as the full one did (`full`): the same op stream and
        bit-equal dynamic plaintexts. page_plaintexts uses it only then (SFHE_TWIN_VALUES=1 forces the full twin)."""
        import os
        from perseus.impl.driver import diff_block
        if os.environ.get("SFHE_TWIN_VALUES", "0") not in ("", "0"):
            return False
        t0 = time.time()
        lean = self.record_page(page, values=False)
        bad = []
        for b, names in self.dyn_names.items():
            s_, k_, _z = diff_block(full[b], lean[b])
            if s_ or k_:
                bad.append(f"block {b}: scalars / rotations")
                continue
            bad += [f"block {b} {nm}" for nm in sorted(names)
                    if nm not in lean[b].pts or not np.array_equal(np.asarray(full[b].pts[nm]), np.asarray(lean[b].pts[nm]))]
        n = sum(len(v) for v in self.dyn_names.values())
        self.log(f"[twin] value-free re-record {time.time() - t0:.1f}s: "
                 + (f"{n}/{n} dynamic plaintexts bit-equal, used for the pages" if not bad
                    else f"{len(bad)} differ ({', '.join(bad[:3])}), keeping the full twin"))
        return not bad

    def page_plaintexts(self, page):
        """{block: {pt name: values}} of the dynamic plaintexts of `page` (a twin re-record; CPU only, so it can run in
        worker processes forked before the device session opens)."""
        from perseus.impl.driver import diff_block
        recs = self.record_page(page, values=not getattr(self, "lean_twin", False))
        out = {}
        for b, names in self.dyn_names.items():
            s, k, _z = diff_block(self.recs[b], recs[b])
            if s or k:
                raise RuntimeError(f"page block {b}: scalars / rotations change: {sorted(s) + sorted(k)}")
            stale = [nm for nm, v in self.recs[b].pts.items() if nm not in names and not np.array_equal(v, recs[b].pts[nm])]
            if stale:                                    # a plaintext the probes saw as static differs here (E15m)
                raise RuntimeError(f"page {page[2]} block {b}: {len(stale)} plaintexts classified static differ from the "
                                   f"recorded page ({stale[:4]}); add probe pages covering this layout")
            out[b] = {nm: np.ascontiguousarray(recs[b].pts[nm]) for nm in names}
        return out

    def stage_page(self, page, pts=None):
        """Stage one page: its dynamic plaintexts (`pts` from page_plaintexts, else a re-record here; skipped when the
        page's layout and positions are the staged ones) and its per-page linears. Returns (items staged, seconds)."""
        t0 = time.time()
        rows, pos, layout, pkv = page
        self.set_page(pos, layout, pkv)
        n = 0
        if any(self.dyn.values()) and self.shape() != self.staged_shape:
            pts = self.page_plaintexts(page) if pts is None else pts
            items, seen = [], {}
            for b in self.dyn:
                for key, lv, nm in self.dyn[b]:
                    v = pts[b][nm]
                    if (key, lv) in seen:                    # a group shares a slot: its members must agree here too
                        if not np.array_equal(seen[(key, lv)], v):
                            raise RuntimeError(f"page {self.layout}: dynamic plaintext group of block {b} {nm} disagrees; "
                                               "the grouping needs more probe pages")
                        continue
                    seen[(key, lv)] = v
                    items.append((key, lv, v))
            self.store.stage(items)
            n += len(items)
        self.staged_shape = self.shape()
        self.model.plain_lins = {b: self.model.plain_linears_layout(b, [k for k, _ in per], [v for _, v in per])
                                 for b, per in enumerate(pkv)}
        return n + self.upload_plain(), time.time() - t0
