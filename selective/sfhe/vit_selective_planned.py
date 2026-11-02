"""The selective ViT prefill PLANNED on the device: `VitPlannedPrefill` over `SelectivePrefill.prefill_selective`.

One recorded stream serves every image with the same sensitive positions (the op stream depends on the token counts
only). What changes per image: block 0's encrypted input page (the client's pixels of CLS + the sensitive patches,
`input_values`) and the plaintext key / value pages of the plaintext stream, which the recorded program reads as
plaintext operands. Those are found by recording a second probe image (`find_dynamic`), kept un-aliased in the
plaintext store, and re-staged per image after a re-recording on the CPU twin (`stage_image`), the decode driver's
`PlannedRun._patch` idea.
"""
from __future__ import annotations

import time

import numpy as np

from perseus.examples.vit.planned import VitPlannedPrefill
from sfhe.vit_selective import PAGE, cyclic_pad


class SelectivePlannedPrefill(VitPlannedPrefill):
    _stages_attr = "selective_stages"

    def __init__(self, model, plan_dir, config, keys, *, positions, **kw):
        super().__init__(model, plan_dir, config, keys, **kw)
        self.positions = np.asarray(positions, dtype=np.int64)
        self._pkv, self.img = None, 0
        self.dyn_names, self.dyn = {}, {}

    def set_image(self, pkv, img):
        self._pkv, self.img = pkv, img

    # ── route B: the per-image PCMM linears ──
    def plain_lins(self):
        return getattr(self.model, "plain_lins", {}) if self.model.route != "sinc" else {}

    def linears(self):
        out = super().linears()
        for b, (lq, lp) in self.plain_lins().items():
            out[lq.name] = lq
            out[lp.name] = lp
        return out

    def fix_ws(self, probes, margin=1.5):
        """Fixed weight scales for the per-image linears from the max |W| over probe images [(rows_e, pkv)]: the
        device's declared w_scale must hold for every image."""
        import math
        from sfhe.dense_xpt import xqk_weights, xpv_weights
        from perseus.impl.pcmm import weight_bits
        if self.model.route == "sinc":
            return {}
        wbits = weight_bits()
        nb = (wbits + 7) // 8
        top = min(2 ** (wbits - 1) - 1, 127 * (256 ** nb - 1) // 255)
        mx = {}
        for _rows, pkv in probes:
            for b, (K_rows, V_rows) in enumerate(pkv):
                for name, W in ((f"b{b}.xqk", xqk_weights(K_rows, self.model.blocks)),
                                (f"b{b}.xpv", xpv_weights(V_rows, self.model.blocks))):
                    mx[name] = max(mx.get(name, 0.0), float(np.abs(W).max()))
        ws = {k: 2.0 ** math.floor(math.log2(top / (margin * v))) for k, v in mx.items()}
        self.model.ws_fixed = ws
        return ws

    def upload_plain(self):
        """Upload the current image's per-image linears (built by the twin re-record) under their fixed names."""
        n = 0
        for b, (lq, lp) in self.plain_lins().items():
            for lin in (lq, lp):
                self.sess.pcmm_upload(lin.name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
                n += 1
        return n

    def _forward(self, rows):
        return self.model.prefill_selective(np.asarray(rows, dtype=np.float64), self.positions, self._pkv, img=self.img)

    def _rows_in(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        return cyclic_pad(rows, -(-len(rows) // PAGE) * PAGE)

    # ── per-image plaintexts ──
    def record_pts(self, rows, pkv, img):
        """Another image's recording on the twin: {block: BlockRec} with its plaintext values."""
        from perseus.impl.driver import Recorder
        self.model.rt.forget_masks()
        rec = Recorder(self.model, self.n_layers)
        self.set_image(pkv, img)
        self.model.prefill_selective(np.asarray(rows, dtype=np.float64), self.positions, pkv, img=img)
        return rec.blocks

    def find_dynamic(self, rows1, pkv1):
        """The plaintexts that differ between the recorded image and another one: {block: {pt name}}; the two streams
        must otherwise bind (same ops, scalars and rotations)."""
        from perseus.impl.driver import diff_block
        other = self.record_pts(rows1, pkv1, img=1000)
        self.dyn_names = {}
        for b, r in self.recs.items():
            s, k, _z = diff_block(r, other[b])
            if s or k:
                raise RuntimeError(f"probe image block {b}: scalars / rotations change between images: {sorted(s) + sorted(k)}")
            self.dyn_names[b] = {nm for nm, v in r.pts.items() if not np.array_equal(v, other[b].pts[nm])}
        return self.dyn_names

    def setup(self, rows):
        """The upstream EncoderPlannedPrefill.setup, with every dynamic plaintext (per image / per page) under its own
        store key: upstream keys plaintexts by value, so a per-image one could share a slot with an equal static one and
        restaging it would overwrite both. Each dynamic value is nudged by a few ulps (a unique bit pattern) while the
        upstream setup runs, then its true value is staged under the key it got (and restaged per image / page)."""
        if not hasattr(self, "progs"):
            self.prepare(rows)
        saved, j = {}, 0
        groups = getattr(self, "dyn_group", {}) or {}            # (block, name) -> group: one slot per group
        nudge = {}
        for b in sorted(self.dyn_names):
            r = self.recs[b]
            for nm in sorted(self.dyn_names[b]):
                g = groups.get((b, nm))
                if g is None or g not in nudge:
                    j += 1
                    if g is not None:
                        nudge[g] = j
                step = nudge[g] if g is not None else j
                v = np.array(r.pts[nm], copy=True)
                flat = v.reshape(-1).view(np.float64)             # complex -> (re, im) pairs; real data as is
                flat[0] = np.frombuffer((flat[0:1].view(np.int64) + step).tobytes(), dtype=np.float64)[0]
                saved[(b, nm)], r.pts[nm] = r.pts[nm], v
        try:
            super().setup(rows)
        finally:
            for (b, nm), v in saved.items():
                self.recs[b].pts[nm] = v
        self.dyn, items = {}, []
        for b, p in enumerate(self.progs[lab] for lab in self.labels):
            names, lab = self.dyn_names.get(b, set()), self.labels[b]
            self.dyn[b] = []
            for nm, lv, _d, form in p.pt_refs:
                if nm in names:
                    key = self.alias[lab][f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm]
                    self.dyn[b].append((key, int(lv), nm))
                    items.append((key, int(lv), np.ascontiguousarray(self.recs[b].pts[nm])))
        if items:
            self.store.stage(items)
        self.log(f"[planned] {len(items)} per-image / per-page plaintexts under {len({(k, lv) for k, lv, _v in items})} "
                 f"keys ({j} groups)")
        return self

    def stage_image(self, rows, pkv, img):
        """Re-record this image on the twin, check its stream binds to the programs, stage its plaintexts in place.
        Returns (plaintexts staged, twin seconds)."""
        from perseus.impl.driver import diff_block
        t0 = time.time()
        if not any(self.dyn.values()):
            # route B: no plaintext-store entry changes per image; the image enters through block 0's encrypted
            # page and the 24 per-image linears, built directly from the plaintext stream's K/V (no twin pass)
            self.set_image(pkv, img)
            self.model.plain_lins = {b: self.model.plain_linears(b, K, V) for b, (K, V) in enumerate(pkv)}
            return self.upload_plain(), time.time() - t0
        recs = self.record_pts(rows, pkv, img)
        items = []
        for b, r in self.recs.items():
            s, k, _z = diff_block(r, recs[b])
            if s or k:
                raise RuntimeError(f"image {img} block {b}: scalars / rotations change: {sorted(s) + sorted(k)}")
            for key, lv, nm in self.dyn[b]:
                items.append((key, lv, np.ascontiguousarray(recs[b].pts[nm])))
        if items:
            self.store.stage(items)
        n_lin = self.upload_plain() if hasattr(self, "sess") else 0
        return len(items) + n_lin, time.time() - t0
