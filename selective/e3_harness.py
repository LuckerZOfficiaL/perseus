#!/usr/bin/env python
"""E3 harness: the ViT fake session and model of `perseus.examples.vit.run_prefill --fake`, as a function.

setup() replicates the runner's environment (the shipped vit_b16_224 runtime switches, the PCMM calibration, the
rotation steps) and returns the pieces the selective-attention work needs: the VitPrefill model on the numpy CPU
twin, its store, configs, dims and Blocks. No GPU is touched (the fake session is numpy).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))                   # this checkout's perseus, before any installed one
import sfhe                                                        # noqa: E402,F401  (runtime patches)
from sfhe import runtime as RT                                     # noqa: E402

RT.apply_pins()
UPSTREAM_VIT = "bootstrap_placements/vit_b16_224"                    # upstream's ViT plan: its runtime, recipe, PCMM bands
# VIT_CONFIGS: another approximation config (sfhe/configs/vit_base_224_photos: LayerNorm bands refitted on photos by
# e12_vit_ln_bands.py from upstream's config; re-derive it after a pull)
CONFIGS = os.environ.get("VIT_CONFIGS", "configs/model/approximation/vit_base_224/configs.json")


def setup(layers=12, capture=None, structure_only=False, nch=512, overrides=None, cls=None, plan=None, weights=None):
    """plan: a plan directory -> the device path's environment (its runtime line, the fast bundle) instead of the
    capture's; the model is still the CPU twin, the planned driver adds the device session."""
    os.chdir(REPO)
    os.environ["PF_NCH"] = str(nch)
    cal = f"{UPSTREAM_VIT}/pcmm_calib.json"
    if os.path.exists(cal):
        os.environ.setdefault("PF_PCMM_CALIB_IN", cal)
    os.environ.setdefault("PERSEUS_DATA", os.path.expanduser("~/data/selective-fhe/perseus_data"))
    os.environ.setdefault("KV_ARENA_GB", "0")
    from perseus.examples.gpt2 import env
    if plan:
        from perseus._paths import resolve
        plan = resolve(plan)
        fast = {k: os.environ.get(k, v) for k, v in (("FIDESLIB_ENCODE_NOSYNC", "1"), ("FIDESLIB_CCMM_ASYNC", "1"),
                                                     ("FIDESLIB_PCMM_NOSYNC", "1"), ("FIDESLIB_ENCODE_PINNED_RING", "256"),
                                                     ("FIDESLIB_STC_POW2_KEY", "1"))}
        over = {**fast, **env.plan_runtime(plan, "n32"), "GPT2_PACKING": "cachemir", **RT.PINS, **(overrides or {})}
    else:
        from perseus._paths import resolve
        over = {**env.plan_runtime(resolve(UPSTREAM_VIT), "n32"), "GPT2_PACKING": "cachemir", **RT.PINS, **(overrides or {})}
    for k, v in over.items():
        if k.startswith("PF_"):
            os.environ.setdefault(k, str(v))
    pcmm = os.environ.get("PF_LINEAR", "") == "pcmm"
    if pcmm:
        over = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
                "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": os.environ.get("PF_PCMM_STAGED_ONLY", "1"), **over}
    env.export_env(None, chain="n32", CKKS_COMPLEX="1", **over)
    if capture:
        os.environ["FHE_GRAPH_DIR"] = capture
    from perseus.impl import config
    from perseus.impl.layout import Dims, ladder_radix_bits, ladder_radix_steps
    from perseus.impl.pcmm import rot_steps as pcmm_rot_steps
    from perseus.impl.blocked import Blocks, blocked_rot_steps
    from perseus.impl.prefill import diag_rot_steps, prefill_rot_steps
    from perseus.examples.vit import weights as V
    from perseus.examples.vit.prefill import VitPrefill
    d0 = Dims.gpt2()
    extra = prefill_rot_steps(d0) | diag_rot_steps(d0) | (pcmm_rot_steps() if pcmm else set()) \
        | blocked_rot_steps(Blocks(d0, nch), ladder_radix_bits())
    ccmm_on = os.environ.get("PF_ATTN", "") == "ccmm"
    if ccmm_on:
        from perseus.impl import sinc_mha
        extra |= sinc_mha.rot_steps(2) | {16, -16}
    from perseus.impl import fakecap
    fakecap.session_env(env.ENV, {"CKKS_COMPLEX": "1", **over})
    rot = extra | fakecap.GPT2_BAND | (ladder_radix_steps(d0, ladder_radix_bits())
                                       if ladder_radix_bits() >= 2 else set())
    _, inf = fakecap.make_session(d0, rot)
    sess = fakecap.FakeSession(inf)
    store = V.VitStore(os.path.expanduser(weights)) if weights else V.VitStore()
    cfgs = config.load_configs(CONFIGS)
    model = (cls or VitPrefill)(inf, store, cfgs, core=fakecap.core, n_layers=layers)
    if capture:
        model.set_capture(capture)
        from perseus.impl.magtape import save_rot_steps
        save_rot_steps(capture, inf.fhe.loaded_rot_steps)
        if structure_only:
            inf.fhe.magnitude_suppressed = True
    os.environ.setdefault("STEPS_T", "128")
    os.environ["MULTI_T"] = str(V.T)
    model.preload_prefill()
    if pcmm:
        from perseus.impl.pcmm import engine
        engine(model.rt).prepare(model.rt.dims.dim)
    if ccmm_on:
        from perseus.impl.ccmm import engine as ccmm_engine
        ccmm_engine(model.rt).prepare(model.rt.dims.dim)
    return SimpleNamespace(model=model, store=store, cfgs=cfgs, inf=inf, sess=sess, B=model.blocks, V=V,
                           d=model.rt.dims, rt=model.rt, pcmm=pcmm, ccmm=ccmm_on)


def sample_rows(V, i=0):
    """Patch rows of one of the shipped EuroSAT sample images (pre-normalized (224, 224, 3) arrays)."""
    import glob
    import numpy as np
    f = sorted(glob.glob(os.path.join(REPO, "scripts/repro/data/vit_eurosat/*.npy")))[i]
    return V.patch_rows(np.load(f))
