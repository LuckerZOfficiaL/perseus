"""E18: the two-stream prefill on Perseus's Llama port (paged blocked GQA, CCMM attention), upstream code unmodified.

The public tokens run as ordinary pages (their own causal attention over public tokens only: the store's K / V are
the plaintext stream's); the encrypted stream (the private tokens and the readout token) is one more page whose
tokens sit at their TRUE positions (PosChunks: RoPE per lane), attending to every stored page and to itself with the
causal rule taken from those positions (Geometry: validity per stored block, and each query row's key count kc).
`install()` rebuilds ccmm_paged.attention_paged from its own source with three substitutions, active only while a
Geometry is set (`set_geometry`): kc from the geometry, validity per (stored block j, key ciphertext o) from the
geometry, and every mask key tagged with the geometry (masks differ per block and per prompt); with `Geometry.prep` set, the stored pages arrive as raw encryptions
of their public K / V (the stage's inputs), take the level / degree of the page's own K / V (the attention's repeated
per-block steps compile only if every block's operands enter alike) and are brought to the CCMM's operand form inside
the stage, right where upstream stores the page's own (recorded)."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import os
import textwrap

import numpy as np

from perseus.impl.prefill import Chunks

BIG = 1 << 30           # the position of a padding token: no real query sees it


@dataclasses.dataclass(frozen=True)
class PosChunks(Chunks):
    """Chunks whose lanes carry explicit absolute positions (pos[i] = token i's position)."""
    pos: tuple = ()

    def lanes(self, k):
        return [int(self.pos[i]) for i in self.idx(k)]

    @property
    def key_tag(self):
        return (self.T, self.base, "pos", hashlib.sha1(np.asarray(self.pos, np.int64).tobytes()).hexdigest()[:12])


class Geometry:
    """The encrypted page's view: kpos[j] = the 128 key positions of stored page j (page 0: tokens 0..126, then the
    BOS at sinc.BOS_TOK = 127, position 0) and of the page itself (j = p), qpos = the page's query positions."""

    def __init__(self, kpos, qpos):
        self.kpos = [np.asarray(k, np.int64) for k in kpos]
        self.qpos = np.asarray(qpos, np.int64)
        allk = np.concatenate(self.kpos)
        self.count = np.array([int((allk <= q).sum()) for q in self.qpos])
        self.key = hashlib.sha1(b"".join(k.tobytes() for k in self.kpos) + self.qpos.tobytes()).hexdigest()[:16]

    prep = None             # (rt, eng, (Ks, Vs) raw store entry, gscale, merge) -> CCMM operands; set by the driver

    def kc(self, tq, qv):
        return np.where(qv, self.count[np.where(qv, tq, 0)], 1)

    def valid(self, j, tq, tk, qv, o):
        q = self.qpos[np.where(qv, tq, 0)]
        return qv & (self.kpos[j][tk[o]] <= q)


_GEOM = {}


def set_geometry(page, geom):
    """The geometry of encrypted page `page` (None clears)."""
    if geom is None:
        _GEOM.pop(page, None)
    else:
        _GEOM[page] = geom


def install():
    """ccmm_paged.attention_paged -> the geometry-aware version (idempotent)."""
    from perseus.impl import ccmm_paged as CP
    if getattr(CP, "_sfhe_ts", False):
        return
    src = textwrap.dedent(inspect.getsource(CP.attention_paged))
    subs = [
        ("    tq, tk, qv, kc = page_meta(p)\n",
         "    tq, tk, qv, kc = page_meta(p)\n    _TS = _sfhe_geom.get(p)\n    if _TS is not None:\n"
         "        kc = _TS.kc(tq, qv)\n"),
        ('tag = ("pg", p, diag)', 'tag = ("pg", p, diag) if _TS is None else ("pgts", _TS.key, p, j)'),
        ("_valid(p, diag, tq, tk, qv, o), (\"sm\", o) + tag)",
         "(_valid(p, diag, tq, tk, qv, o) if _TS is None else _TS.valid(j, tq, tk, qv, o)), (\"sm\", o) + tag)"),
        ('("sm", layer, r, "pg", p), cv)', '("sm", layer, r, "pg", p) + (() if _TS is None else (_TS.key,)), cv)'),
        ("        kw = dict(keep=True, export=not final) if _DEVKV else {}\n",
         "        _TS0 = _sfhe_geom.get(p)\n        if _TS0 is not None and _TS0.prep is not None:\n"
         "            store[:] = [_TS0.prep(rt, eng, x, gscale, merge, (Ks[0].level, Ks[0].noise_deg),\n"
         "                                  (Vs[0].level, Vs[0].noise_deg)) for x in store[:p]]\n"
         "        kw = dict(keep=True, export=not final) if _DEVKV else {}\n"),
    ]
    # pages >= 1: one step per stored block (upstream's repeated steps reuse the block's names so the page-0 plan binds
    # any page count on the eager path; a compiled Program needs every op's name once)
    subs += [("    ops, p = rt.ops, page\n", "    ops, p = rt.ops, page\n    _U = p > 0\n"),
             ('rt.step("kb", repeat=True)', 'rt.step("kb" + (str(j) if _U else ""), repeat=not _U)'),
             ('rt.step(f"r{r}", repeat=True)', 'rt.step(f"r{r}" + (f"_{j}" if _U else ""), repeat=not _U)'),
             ('rt.step(f"u{r}", repeat=True)', 'rt.step(f"u{r}" + (f"_{j}" if _U else ""), repeat=not _U)'),
             ('rt.step(f"y{r}", repeat=True)', 'rt.step(f"y{r}" + (f"_{j}" if _U else ""), repeat=not _U)'),
             ('rt.step("pv", repeat=True)', 'rt.step("pv" + (str(j) if _U else ""), repeat=not _U)')]
    # SFHE_TS_ZERO_MASKED=1 (E26 fix): the exp series' input is score + mask, so a masked entry (an invalid key, or any
    # entry of a padding query row) keeps its raw score on top of the mask's bottom-of-domain constant; a negative raw
    # score lands below -1, where the Chebyshev exp and its squarings blow up (prompts whose stored page is mostly
    # padding: max error ~1e4, NaN on the twin). Zero the masked scores first (one 0 / 1 plaintext product per score
    # ciphertext): every masked entry then sits exactly on the domain's bottom, as the mask intends.
    if os.environ.get("SFHE_TS_ZERO_MASKED", "0") not in ("", "0"):
        subs.append(("        if merge:           # the scores carry alpha (unmerge's scale): the domain map's product rode its masks\n",
                     "        if merge:           # the scores carry alpha (unmerge's scale): the domain map's product rode its masks\n"
                     "            s = rt.mult_mask(s, site + (\"zm\",), lambda: np.where(valid, 1.0, 0.0))\n"))
    for a, b in subs:
        if src.count(a) != 1:
            raise RuntimeError(f"llama_ts.install: upstream attention_paged changed ({a!r} found {src.count(a)}x)")
        src = src.replace(a, b)
    ns = vars(CP)
    ns["_sfhe_geom"] = _GEOM
    exec(compile(src, "<sfhe.llama_ts attention_paged>", "exec"), ns)
    CP._sfhe_ts = True
