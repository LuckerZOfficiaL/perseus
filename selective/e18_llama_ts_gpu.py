#!/usr/bin/env python
"""E18 phase 2, milestone 2(a): the two-stream Llama-3.1 prefill PLANNED on the device, E15 drug prompts.

Only the encrypted page runs encrypted (the private tokens + the readout token r, at their true positions); the public
part's K / V come from the float64 public stream (on the server: plaintext) and enter every block as encryptions of
public vectors (the block's extra input ciphertexts, e18_llama_ts.raw_store), brought to the CCMM's operand form inside
the block (recorded). One recorded stream serves every prompt with the same number of stored public pages (2: 128-255
public tokens): what changes per prompt is block 0's encrypted rows, the stored K / V inputs, and the plaintexts that
encode positions (RoPE, validity, key counts); those are found by recording probe prompts (`find_dynamic`), kept under
their own store keys, and restaged per prompt from a re-record on the CPU twin. Each prompt's encrypted readout logits
are compared with the float64 two-stream model (KL, argmax, the answer letter).

  env $RT PF_LAST_QUERY=0 PF_PCMM_CALIB_IN=<plan>/pcmm_calib.json \\
      python selective/e18_llama_ts_gpu.py --plan bootstrap_placements/selective_e18/llama_ts_p2 --n 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e18_llama_ts as TS

REPO, D = TS.REPO, TS.D


def build_prompt(model, idx, split, tokenizer, owt=0, owt_page=1, enron=0, cat=1):
    """Everything one prompt needs: tokens, layout, float64 reference, the encrypted page, geometry, store patterns
    (owt = L: the full prefill's page 1 on openwebtext window idx, e18_llama_ts.prompt)."""
    from sfhe.llama_ts import BIG, Geometry
    a = argparse.Namespace(idx=idx, split=split, tokenizer=tokenizer, owt=owt, owt_page=owt_page, enron=enron, cat=cat)
    ids, S, letters, label = TS.prompt(a)
    pub, E = TS.layout(ids, S)
    ref, kvs = TS.exact_hidden_ts(model, ids, pub, E, model.n_layers)
    pages = TS.pages_of(ids, pub, E, BIG, model.blocks.t)
    npub = len(pages) - 1
    kpos = [list(kp) for _t, _c, kp in pages]
    kpos[0] = kpos[0] + [0]
    geom = Geometry(kpos, TS.query_positions(pages[-1][2], len(ids) - 1))
    geom.prep = TS.prep_store
    rows, k = [], 1
    for j, (_t, _c, kp) in enumerate(pages[:-1]):
        rp = [k + i if x < BIG else (k + TS.n_real(kp) - 1 if TS.PAD_COPY else None) for i, x in enumerate(kp)]
        k += sum(x < BIG for x in kp)
        rows.append(rp + ([0] if j == 0 else []))
    pats = TS.store_patterns(model, kvs, rows)
    return dict(idx=idx, ids=ids, S=S, E=E, letters=letters, label=label, npub=npub, toks=pages[-1][0], ch=pages[-1][1],
                geom=geom, pats=pats, refE=ref[-1][len(pub):], n_pub=len(pub))


def make_class():
    from perseus.examples.llama.planned import LlamaPlannedPrefill, _key  # noqa

    class TsPlanned(LlamaPlannedPrefill):
        """LlamaPlannedPrefill over the encrypted page of a two-stream prompt (self.P)."""

        def _loop(self, P, on_block=None):
            from perseus.examples.llama import prefill as LP
            from perseus.impl import blocked as BL
            from perseus.impl.prefill import evict_block_masks
            from sfhe.llama_ts import set_geometry
            m = self.model
            B = m.blocks
            set_geometry(P["npub"], P["geom"])
            m.token_begin(0)
            try:
                Xs = BL.encode_blocked(m.rt, np.stack([m.w.embedding(t) for t in P["toks"]]), P["ch"], B)
                for b in range(m.n_layers):
                    m.rt._scope_visits.clear()
                    store = TS.raw_store(m, P["pats"][b])
                    cap = m.enter_stage(b, 1)
                    BL.name_blocked(m.inf, Xs)
                    with m.rt.step(f"blk{b}"):
                        Xs = LP.llama_prefill_block_blocked(m.rt, Xs, m.pcmm_weights(b), m.cfgs.block(b), P["ch"], m.rope,
                                                            m.prefix_kv(b), B, None, layer=b,
                                                            paged=(P["npub"], store, True), last=None)
                    m.exit_stage(b, cap)
                    if on_block is not None:
                        on_block(b, Xs)
                    evict_block_masks(m.rt)
            finally:
                set_geometry(P["npub"], None)
                m.token_end(0)
            return Xs

        def record_raw(self, P, values=True, reduce=None):
            """The prompt's per-block recording: (blocks {b: BlockRec}, io {b: names / metas / values}). `reduce(b, name,
            value)` replaces each plaintext value right after its block (a hash, a store key): a whole recording of
            full plaintext arrays holds ~200 GB."""
            from perseus.examples.llama import prefill as LP
            from perseus.impl import fakecap
            from perseus.impl.driver import Recorder
            m = self.model
            m.rt.forget_masks()
            rec = Recorder(m, self.n_layers)
            io, orig = {}, LP.llama_prefill_block_blocked

            def block(rt, Xs, Pw, cfgs, ch, rope, prefixes, B, kv=None, layer=0, paged=None, last=None):
                flat = [x for blk in Xs for x in blk]
                st = [c for Ks, Vs in list(paged[1]) for c in list(Ks) + list(Vs)]
                metas = [(int(c.level), int(c.noise_deg)) for c in flat]
                vals = [np.array(c.vec) for c in flat] if (layer == 0 and values) else None
                Y = orig(rt, Xs, Pw, cfgs, ch, rope, prefixes, B, kv=kv, layer=layer, paged=paged, last=last)
                metas += [(int(c.level), int(c.noise_deg)) for c in st]       # as prep_store set them
                io[layer] = ([c.name for c in flat] + [c.name for c in st], [c.name for blk in Y for c in blk], metas,
                             vals, len(flat))
                return Y
            def on_block(b, _Xs):
                r = rec.blocks[b]
                if reduce is not None:
                    for nm in list(r.pts):
                        r.pts[nm] = reduce(b, nm, r.pts[nm])
                r.objs = {}
            LP.llama_prefill_block_blocked = block
            nv = fakecap.CapCt.novec
            fakecap.CapCt.novec = not values
            try:
                self._loop(P, on_block=on_block)
            finally:
                LP.llama_prefill_block_blocked = orig
                fakecap.CapCt.novec = nv
            return rec.blocks, io

        def record(self, toks=None):
            t0 = time.time()
            self.uniq = {}
            dyn = getattr(self, "dyn_names", {})

            grp = getattr(self, "dyn_group", {})

            rot = getattr(self, "rotate_dyn", False)

            def to_key(b, nm, v):
                v = np.ascontiguousarray(v)
                if nm in dyn.get(b, ()):
                    # rotate: one slot per mask name, restaged per block (every block runs the same op stream, so a
                    # name is one op's mask at one level in every block); else one slot per same-name group
                    k = f"dynslot/{nm}" if rot else f"dyn/g{grp[(b, nm)]}"
                else:
                    k = _key(v)
                self.uniq.setdefault(k, v)
                return k
            blocks, io = self.record_raw(self.P, reduce=to_key)
            self.ch = self.P["ch"]
            self.in_metas, self.in_names, self.nh_in, self.kv_names = {}, {}, {}, {}
            for i, r in blocks.items():
                names_in, names_out, metas, vals, nh = io[i]
                r.out_names = list(names_out)
                self.in_names[i], self.in_metas[i], self.nh_in[i] = names_in, metas, nh
                if i == 0:
                    self.in_vals = vals
            self.n_hidden = len(io[0][1])
            self.recs = blocks
            self.record_s = time.time() - t0
            return self.recs

        def find_dynamic(self, probes):
            """The plaintexts that differ between the recorded prompt and the probes: {block: {name}}; the streams must
            otherwise bind (same ops, scalars, rotations)."""
            from perseus.impl.driver import diff_block
            h = lambda b, nm, v: _key(np.ascontiguousarray(v))
            base, _ = self.record_raw(self.P, reduce=h)
            self.dyn_names = {b: set() for b in base}
            sig = {(b, nm): (v,) for b, r in base.items() for nm, v in r.pts.items()}
            for Q in probes:
                other, _ = self.record_raw(Q, reduce=h)
                for b, r in other.items():
                    for nm, v in r.pts.items():
                        sig[(b, nm)] += (v,)
                for b, r in base.items():
                    s_, k_, _z = diff_block(r, other[b])
                    if s_ or k_:
                        raise RuntimeError(f"probe {Q['idx']} block {b}: scalars / rotations change: {sorted(s_) + sorted(k_)}")
                    if set(r.pts) != set(other[b].pts):
                        raise RuntimeError(f"probe {Q['idx']} block {b}: the plaintext names differ")
                    self.dyn_names[b] |= {nm for nm, v in r.pts.items() if v != other[b].pts[nm]}
                del other
            # the same mask rebuilt in every block shares one store slot: the same plaintext name (every block runs the
            # same op stream and numbers its plaintexts from 0, so one name = one op's mask) with equal values on every
            # probe; a layer-dependent mask (per-layer softmax config) keeps one slot per value. Checked per prompt.
            ids = {}
            self.dyn_group = {(b, nm): ids.setdefault((nm,) + sig[(b, nm)], len(ids)) for b, names in self.dyn_names.items()
                              for nm in names}
            self.log(f"[e18-gpu] {sum(len(v) for v in self.dyn_names.values())} dynamic plaintexts in {len(ids)} groups")
            return self.dyn_names

        def setup(self, toks=None):
            super().setup(toks)
            self.dyn = {}
            for b, p in enumerate(self.progs[lab] for lab in self.labels):
                lab = self.labels[b]
                self.dyn[b] = []
                for nm, lv, _d, form in p.pt_refs:
                    if nm in self.dyn_names.get(b, ()):
                        key = self.alias[lab][f"{lab}/{nm}" if (form == "at_use" or nm.startswith("pt_")) else nm]
                        self.dyn[b].append((key, int(lv), nm))
            return self

        def stage_prompt(self, P):
            """Re-record P on the twin (values: block 0's rows), check it binds, stage its dynamic plaintexts.
            Returns (block-0 values, seconds)."""
            from perseus.impl.driver import diff_block
            t0 = time.time()
            blocks, io = self.record_raw(P, reduce=lambda b, nm, v: np.ascontiguousarray(v) if nm in self.dyn_names[b]
                                         else _key(np.ascontiguousarray(v)))
            items, seen = [], {}
            self.pending = {}
            for b, r in blocks.items():
                s_, k_, _z = diff_block(self.recs[b], r)
                if s_ or k_:
                    raise RuntimeError(f"prompt {P['idx']} block {b}: scalars / rotations change: {sorted(s_) + sorted(k_)}")
                # a plaintext static on the probes must equal the recorded value here too (else it needed restaging)
                bad = [nm for nm, k in r.pts.items() if nm not in self.dyn_names[b] and k != self.recs[b].pts[nm]]
                if bad:
                    raise RuntimeError(f"prompt {P['idx']} block {b}: {len(bad)} plaintexts static on the probes differ "
                                       f"here ({bad[:3]}): add this prompt to the probes")
                if getattr(self, "rotate_dyn", False):  # staged per block, right before the block runs (run_prompt)
                    mine = {}
                    for key, lv, nm in self.dyn[b]:
                        mine.setdefault((key, lv), np.ascontiguousarray(r.pts[nm]))
                    self.pending[b] = [(key, lv, v) for (key, lv), v in mine.items()]
                    continue
                for key, lv, nm in self.dyn[b]:
                    v = np.ascontiguousarray(r.pts[nm])
                    if (key, lv) in seen:
                        if not np.array_equal(seen[(key, lv)], v):
                            raise RuntimeError(f"prompt {P['idx']} block {b} {nm}: its dynamic plaintext group disagrees "
                                               "here (needs more probe prompts)")
                        continue
                    seen[(key, lv)] = v
                    items.append((key, lv, v))
            if items:
                self.store.stage(items)
            self.P, self.ch = P, P["ch"]
            return io[0][3], len(items), time.time() - t0

        def run_prompt(self, x0_vals, P):
            """The encrypted page on the device: block 0's rows and every block's stored public K / V encrypted
            here (the public ones by the server), then the blocks. Returns (wall s, encryption s of the store)."""
            fhe = self.fhe
            t0 = time.time()
            x0 = [fhe.encrypt(v, lv, dg) for v, (lv, dg) in zip(x0_vals, self.in_metas[0][:self.nh_in[0]])]
            t_enc = t_stage = 0.0
            for b in range(self.n_stages):
                p = self.progs[self.labels[b]]
                nh = self.nh_in[b]
                if getattr(self, "pending", None):    # this block's dynamic plaintexts into the shared slots
                    ts_ = time.time()
                    fhe.device_sync()               # the previous block has read them
                    self.store.stage(self.pending[b])
                    t_stage += time.time() - ts_
                te = time.time()
                svals = [v[o] for kp, vp in P["pats"][b] for v in (kp, vp) for o in range(len(v))]
                st = [fhe.encrypt(np.asarray(v, dtype=np.complex128), lv, dg)
                      for v, (lv, dg) in zip(svals, self.in_metas[b][nh:])]
                t_enc += time.time() - te
                src = x0 if b == 0 else [reg for _j, reg in self.reg[b - 1]]
                ins = [src[j] if j < nh else st[j - nh] for j in self.in_pos[b]]
                p.run(ins, list(self.reg[b]))
                if st or b == 0:            # free this block's eager inputs once it has read them (the pool is small)
                    fhe.device_sync()
                    del ins, st
                    if b == 0:
                        x0 = None
            fhe.device_sync()
            self.t_stage = t_stage
            return time.time() - t0, t_enc

    return TsPlanned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--configs", default=TS.CONFIGS)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=20, help="validation prompts (128-255 public tokens) to run")
    ap.add_argument("--first", type=int, default=0, help="first validation index to consider (E26: extend the set)")
    ap.add_argument("--probes", default="train:0,train:1,train:2", help="prompts for find_dynamic (first = recorded)")
    ap.add_argument("--owt", type=int, default=0, help="L: the full prefill's page (--owt-page) on openwebtext windows "
                    "of L tokens instead (probes: windows 0..2, runs: windows 3..)")
    ap.add_argument("--owt-page", type=int, default=1, help="0 / 1 (with --owt), or p >= 2: the full prefill's page p")
    ap.add_argument("--enron", type=int, default=0, help="L: E14 Enron prompts of L tokens (e18_llama_ts.enron_prompts)")
    ap.add_argument("--cat", type=int, default=1)
    ap.add_argument("--probe-idx", default=None, help="--enron / --owt-page >= 2: probe prompt indices (first = recorded)")
    ap.add_argument("--eval-idx", default=None, help="--enron / --owt-page >= 2: evaluated prompt indices")
    ap.add_argument("--rotate-dyn", action="store_true", help="per-prompt plaintexts: one slot per mask name, restaged "
                    "before every block (device memory: the plaintext store holds one block's masks, not 32)")
    ap.add_argument("--compile-only", action="store_true", help="CPU only: probes, dynamic plaintexts, compile, then stop")
    ap.add_argument("--work-dir", default=f"{D}/e18/p2/work")
    ap.add_argument("--out", default=f"{D}/e18/p2/gpu_ts.json")
    a = ap.parse_args()
    os.chdir(REPO)
    from perseus import _pc, config as pcfg
    from perseus._paths import resolve
    from perseus.examples.llama.run_prefill import _plan_config
    RT = pcfg.runtime_env(a.plan)
    over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
            "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": "1", "FIDESLIB_ENCODE_NOSYNC": "0", **RT,
            "LLAMA_PACKING": "cachemir_complex", "PF_LAST_QUERY": "0"}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ[k] = str(v)
    from perseus.examples.llama import env, hf, weights as W
    from perseus.examples.llama.config import load_configs
    from perseus.examples.llama.prefill import LlamaPrefill, plain_logits
    from perseus.impl import fakecap
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.layout import ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.prefill import prefill_rot_steps
    from perseus.impl.sinc import rot_steps as sinc_rot_steps
    from sfhe.llama_ts import install
    install()
    cfgs = load_configs(a.configs)
    store = W.SafetensorsStore(hf.NAME)
    dims = W.llama_dims(store.config)
    rot = (prefill_rot_steps(dims) | pcmm_rot_steps() | ladder_radix_steps(dims, ladder_radix_bits())
           | sinc_rot_steps() | {16, -16} | blocked_rot_steps(Blocks(dims, int(os.environ.get("PF_NCH", "512"))),
                                                            ladder_radix_bits()))
    env.export_env(None, chain="n32", **{"CKKS_COMPLEX": "1", **over})   # CUDA_VISIBLE_DEVICES: the launcher's
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    _, inf = fakecap.make_session(dims, set(env.rot_steps(dims)) | rot)
    sess = fakecap.FakeSession(inf)
    lw = W.LlamaWeights(store, cfgs, a.configs, dims, "cachemir_complex", pinned=None)
    model = LlamaPrefill(inf, lw, cfgs, core=fakecap.core, packing="cachemir_complex")
    log = lambda m: print(m, flush=True)
    t0 = time.time()
    if a.enron or a.owt_page >= 2:
        probes = [("x", int(i)) for i in a.probe_idx.split(",")]
    else:
        probes = [tuple(x.split(":")) for x in a.probes.split(",")] if not a.owt else [("owt", 0), ("owt", 1)]
    bp = lambda i, sp: build_prompt(model, int(i), sp, a.tokenizer, owt=a.owt, owt_page=a.owt_page, enron=a.enron,
                                    cat=a.cat)
    P0 = bp(probes[0][1], probes[0][0])
    Ps = [bp(i, sp) for sp, i in probes[1:]]
    log(f"[e18-gpu] probes built {time.time() - t0:.0f}s: public {[P0['n_pub']] + [p['n_pub'] for p in Ps]}, "
        f"stored pages {[P0['npub']] + [p['npub'] for p in Ps]}")
    Cls = make_class()
    keys = _pc.KeySource.generate([], conj=True, layout="packed")
    run = Cls(model, resolve(a.plan), _plan_config(a.plan), keys, work_dir=a.work_dir, pt_rows=28, stc_cache=8, log=log)
    run.P = P0
    run.rotate_dyn = a.rotate_dyn
    t1 = time.time()
    dyn = run.find_dynamic(Ps)
    log(f"[e18-gpu] dynamic plaintexts {sum(len(v) for v in dyn.values())} over {len(dyn)} blocks ({time.time() - t1:.0f}s)")
    if a.compile_only:                      # report every block's compile-check errors instead of the first
        import collections, re
        orig_compile = _pc.Program.compile

        def compile_kg(*args, **kw):
            progs = orig_compile(*args, **{**kw, "keep_going": True})
            for lab in sorted(progs, key=lambda x: int(x.split("_")[1])):
                errs = progs[lab].check[1]
                if errs:
                    c = collections.Counter(re.sub(r"[0-9]+", "N", str(e))[:150] for e in errs)
                    log(f"[e18-gpu] {lab}: {len(errs)} check errors: " + "; ".join(f"{v}x {k}" for k, v in c.most_common(4))
                        + f" | e.g. {[str(e)[:120] for e in errs[1:3]]}")
            return progs
        _pc.Program.compile = compile_kg
    run.prepare(None)
    n_bts = sum(run.progs[lab].n_bootstraps for lab in run.labels)
    n_fin = [sum(run.progs[lab].n_finishes[i] for lab in run.labels) for i in (0, 1)]
    log(f"[e18-gpu] compiled: {n_bts} refresh sites + {n_fin[1]} PCMM / CCMM outputs = {n_bts + n_fin[1]} bts")
    if a.compile_only:
        return
    n_in = max(len(run.in_names[b]) for b in run.in_names)
    run.pool = max(run.pool, n_in + (2 if a.rotate_dyn else 8))   # every block's eager inputs (rows + stored K / V)
    log(f"[e18-gpu] eager pool {run.pool} (largest block input count {n_in})")
    tries = [dict(), dict(stc_cache=4), dict(stc_cache=1), dict(stc_cache=1, schedule="program")]
    for k, kw in enumerate(tries):          # device memory: a smaller StC-first cache, then the capture-order schedule
        for key, v in kw.items():
            setattr(run, key, v)
        if "schedule" in kw:                # the schedule is compiled in: recompile
            del run.progs
            run.prepare(None)
        try:
            run.setup(None)
            break
        except RuntimeError as e:
            if "memory" not in str(e) or k == len(tries) - 1:
                raise
            log(f"[e18-gpu] setup did not fit with {kw or 'defaults'} ({str(e)[:120]}); retrying with {tries[k + 1]}")
    # the validation prompts with the recorded geometry (2 stored public pages)
    val = json.load(open(f"{D}/e15/drugs_t128.json"))["validation"]
    todo = [int(x) for x in a.eval_idx.split(",")] if (a.enron or a.owt_page >= 2 or a.eval_idx) else None
    res, i = [], (2 if a.owt else a.first)
    while len(res) < a.n and (i < len(val) if todo is None else bool(todo)):
        if todo is not None:
            i = todo.pop(0)
        P = bp(i, "validation")
        i += 1
        if P["npub"] != P0["npub"]:
            log(f"[e18-gpu] validation {P['idx']}: {P['n_pub']} public tokens ({P['npub']} pages), skipped")
            continue
        x0, n_st, s_tw = run.stage_prompt(P)
        wall, t_enc = run.run_prompt(x0, P)
        h = run.hidden()[:len(P["E"])]
        lg = plain_logits(model, h[-1:])[0]
        lr = plain_logits(model, P["refE"][-1:])[0]
        pz = np.exp(lr - lr.max()); pz /= pz.sum()
        qz = np.exp(lg - lg.max()); qz /= qz.sum()
        kl = float((pz * (np.log(pz + 1e-300) - np.log(qz + 1e-300))).sum())
        let = np.array(P["letters"], dtype=int)
        lt = (lambda z: int(np.argmax(z[let]))) if len(let) else (lambda z: -1)
        row = dict(idx=P["idx"], n_tokens=len(P["ids"]), n_enc=len(P["E"]), kl=kl, argmax_enc=int(lg.argmax()),
                   argmax_ref=int(lr.argmax()), letter_enc=lt(lg), letter_ref=lt(lr),
                   label=int(P["label"]), wall=wall, store_enc_s=t_enc, stage_s=getattr(run, "t_stage", 0.0), twin_s=s_tw,
                   staged=n_st,
                   maxerr=float(np.abs(h - P["refE"]).max()))
        res.append(row)
        log(f"[e18-gpu] {len(res)}/{a.n} validation {P['idx']}: {row['n_tokens']} tokens, {row['n_enc']} encrypted, wall "
            f"{wall:.2f}s (store encryption {t_enc:.2f}s, mask staging {getattr(run, 't_stage', 0.0):.2f}s), KL {kl:.3g}, letter enc {row['letter_enc']} ref "
            f"{row['letter_ref']} label {row['label']}, twin {s_tw:.0f}s")
        json.dump(dict(results=res, bts=n_bts + n_fin[1]), open(a.out, "w"), indent=1)
    same = sum(r["letter_enc"] == r["letter_ref"] for r in res)
    walls = sorted(r["wall"] for r in res)
    log(f"[e18-gpu] SUMMARY prompts={len(res)} letter enc=ref {same}/{len(res)} argmax enc=ref "
        f"{sum(r['argmax_enc'] == r['argmax_ref'] for r in res)}/{len(res)} median KL "
        f"{np.median([r['kl'] for r in res]):.3g} max KL {max(r['kl'] for r in res):.3g} median wall {walls[len(walls) // 2]:.2f}s "
        f"bts={n_bts + n_fin[1]} correct enc {sum(r['letter_enc'] == r['label'] for r in res)}")
    run.close()
    model.close()
    sess.close()


if __name__ == "__main__":
    main()
