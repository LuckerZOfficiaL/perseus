"""late-inject-replace PLANNED on the device: `VitPlannedPrefill` over `LatePrefill.prefill_late`. One recorded
stream (hand-over + the last k blocks) serves every image with the same private positions; per image the client's
pixel page changes (block 0's input) and the ordinary tokens' plain residual rows change (the hand-over's plaintext
add), found and restaged like SelectivePlannedPrefill's dynamic plaintexts."""
from __future__ import annotations

import numpy as np

from perseus.examples.vit import weights as V
from perseus.examples.vit.prefill import add_rows
from sfhe.vit_selective_planned import SelectivePlannedPrefill


class LatePlannedPrefill(SelectivePlannedPrefill):
    _stages_attr = "late_stages_bound"

    def __init__(self, model, plan_dir, config, keys, *, positions, k, direct_handover=False, **kw):
        super().__init__(model, plan_dir, config, keys, positions=positions, **kw)
        self.k = int(k)
        self._hid = None
        # opt-in: build the per-image hand-over plaintexts directly (handover_vectors) instead of re-recording the
        # whole stream on the twin; falls back to the twin if the vectors do not reproduce the recorded image's
        self.direct_handover = bool(direct_handover)
        self._dyn_map = None
        # the recorded stages are the hand-over + the last k blocks, not the model's 12 layers (labels are fixed here,
        # PlannedPrefill.__init__ built them from model.n_layers)
        self.n_layers = self.n_stages = self.k + 1
        self.labels = [f"block_{b}" for b in range(self.n_stages)]

    def set_image(self, hidden_plain, img, positions=None):
        """positions: this image's private patches (E11 faces: they vary per image; the op stream does not)."""
        self._hid, self.img = hidden_plain, img
        if positions is not None:
            self.positions = np.asarray(sorted(int(s) for s in positions), dtype=np.int64)

    def _forward(self, rows):
        return self.model.prefill_late(np.asarray(rows, dtype=np.float64), self.positions, self._hid, self.k, img=self.img)

    def _rows_in(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        n = len(rows)
        full = -(-n // 128) * 128
        return np.vstack([rows, rows[n - (full - n):]]) if full > n else rows

    def plain_lins(self):
        return {}

    def linears(self):
        """The PCMM linears the stages name: the patch embedding (hand-over) + the model's last k blocks. The base
        class enumerates blocks by stage index (0..k), which would declare blocks 0..k instead of L-k..L-1."""
        from perseus.impl.pcmm import PcmmLinear
        L = self.model.n_layers
        out = {}
        for b in range(L - self.k, L):
            for lin in self.model.diag_weights(b).values():
                if isinstance(lin, PcmmLinear):
                    out[lin.name] = lin
        emb = self.model.embed_weights()
        out[emb.name] = emb
        return out

    def fix_ws(self, probes, margin=1.5):
        return {}

    def record(self, rows):
        self._hid0 = self._hid                           # the recorded image's plain rows (_map_dynamic matches them)
        self._pos0 = self.positions
        return super().record(rows)

    def record_pts(self, rows, hidden_plain, img):
        from perseus.impl.driver import Recorder
        self.model.rt.forget_masks()
        rec = Recorder(self.model, self.n_layers)
        self.set_image(hidden_plain, img)
        self.model.prefill_late(np.asarray(rows, dtype=np.float64), self.positions, hidden_plain, self.k, img=img)
        return rec.blocks

    # -- per-image hand-over plaintexts without the twin --
    def handover_vectors(self, hidden_plain, positions=None):
        """{(site, g, k): complex slot vector} of the hand-over's two plaintext adds, exactly as `LatePrefill.late_stages`
        builds them through `add_rows`: the position offsets at the private rows ("off") and the ordinary tokens' plain
        rows ("hid": private rows zero, residual scale applied)."""
        ch, B = self.ch, self.model.blocks
        hp = np.asarray(hidden_plain, dtype=np.float64)
        n = len(hp)
        pos = self.positions if positions is None else positions
        isS = np.zeros(n, dtype=bool)
        isS[np.asarray(sorted(int(s) for s in pos), dtype=np.int64) + 1] = True
        r = V.residual_scale(self.model.rt.dims.dim)
        rows = {"off": np.where(isS[:, None], V.embed_offsets(self.model.store) * r, 0.0),
                "hid": np.where(isS[:, None], 0.0, hp * r)}
        out = {}
        n_groups = len(self.in_names[0]) // B.K           # the hand-over input: [g][k] blocked rows, row-major
        for site, rr in rows.items():
            class _Grab:                                # add_rows' rt: keep each mask's build() instead of adding it
                ops = None

                @staticmethod
                def add_mask(x, key, build, site=site):
                    out[(site,) + tuple(key[1:])] = np.asarray(build())
                    return x
            add_rows(_Grab, [[None] * B.K for _ in range(n_groups)], rr, ch, B, None)
        return out

    def _map_dynamic(self):
        """{program pt name: (g, k)} for the dynamic plaintexts, by matching the recorded image's own values; None if
        anything does not match (then the twin path is used)."""
        dyn = {b: [nm for _key, _lv, nm in v] for b, v in self.dyn.items() if v}
        if set(dyn) != {0}:
            return None
        vecs = self.handover_vectors(self._hid0, self._pos0)
        m = {}
        for nm in dyn[0]:
            want = np.asarray(self.recs[0].pts[nm])
            hit = [gk for gk, v in vecs.items() if v.shape == want.shape
                   and np.allclose(v, want, rtol=0, atol=1e-9 * max(1.0, float(np.abs(v).max())))]
            if len(hit) != 1:
                return None
            m[nm] = hit[0]
        return m

    def stage_image(self, rows, hidden_plain, img, positions=None):
        """SelectivePlannedPrefill.stage_image; with direct_handover the 4 hand-over plaintexts are built directly
        (no twin pass). Returns (plaintexts staged, seconds)."""
        import time
        if positions is not None:
            self.set_image(hidden_plain, img, positions)
        if not self.direct_handover:
            return super().stage_image(rows, hidden_plain, img)
        t0 = time.time()
        if self._dyn_map is None:
            self._dyn_map = self._map_dynamic() or {}
            self.log(f"[late] direct hand-over staging: {'on, ' + str(len(self._dyn_map)) + ' plaintexts' if self._dyn_map else 'no match, twin path'}")
        if not self._dyn_map:
            return super().stage_image(rows, hidden_plain, img)
        self.set_image(hidden_plain, img, positions)
        vecs = self.handover_vectors(hidden_plain)
        items = [(key, lv, np.ascontiguousarray(vecs[self._dyn_map[nm]])) for key, lv, nm in self.dyn[0]]
        self.store.stage(items)
        return len(items), time.time() - t0
