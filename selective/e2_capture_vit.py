#!/usr/bin/env python
"""E2: record the ViT prefill circuit for a prompt of T' tokens on the CPU twin (no GPU, no encryption).

Wraps perseus.examples.vit.run_prefill --fake --capture with the first T' rows of a sample image, so the
recorded op stream is what Perseus would execute for an encrypted stream of T' tokens (CLS + T'-1 sensitive
patches). The plaintext-key attention of the real selective block is not in it; this is the lower bound of
the encrypted work. PF_NCH picks the channel-block width (512: two ciphertexts per 128-token page; 1024: one
ciphertext per 64-token page).

  python selective/e2_capture_vit.py --tokens 17 --nch 512 --out ~/data/selective-fhe/e2/graphs/vit_T17_nch512
"""
from __future__ import annotations

import argparse
import os
import sys

RUNTIME = ("FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 SM_DEN_RECIP=1 FIDESLIB_BTS_REAL=1 "
           "GELU_FOLD=1 LN_CHEB=1 SM_GS_FIRST=1 SM_PERIODIC=1 SM_FOLD=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32 "
           "KV_LANES=1 CHEB_BSGS=1 QKV_LANES=1 BTS_MAX_INPUT_LEVEL=50 SM_HRS_PACK=1 CUTMAX_PAIR=1 IMPL_DEAD_OPS=1 "
           "LADDER_RADIX_BITS=3 PF_LINEAR=pcmm PF_ACT_FOLD=1 PF_GELU_JOINT=1 PF_JOINT_FORCE=1 PF_LAYOUT=blocked "
           "SM_CHEB_GS=1 PF_ATTN=ccmm PF_CCMM_MERGE=1 PF_SM_UNPACK_FOLD=1 PF_SM_R_REFRESH=1 PF_SM_R_SCALE=256 "
           "PF_SM_R_ITER=1 PF_SM_R_ITER_P=8 PF_SM_PAIR=1 PF_SM_PAIR_ROUNDS=1 PF_SINC_A2=1 FIDESLIB_BTS_STC_FIRST=2 "
           "FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_STC_FIRST_STAGED_ONLY=0 "
           "PF_PCMM_STAGED_ONLY=0")  # the shipped vit_b16_224 plan's runtime line (PLAN_CMD.txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, required=True, help="T': rows of the prompt (CLS included)")
    ap.add_argument("--nch", type=int, default=512)
    ap.add_argument("--out", required=True, help="capture directory (block_<b>/graph.json)")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--pad", default="1", help="VIT_PAD: 1 pads to whole 128-token pages (the shipped setting)")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo)
    os.environ["PF_NCH"] = str(a.nch)
    os.environ["VIT_PAD"] = a.pad
    os.environ.setdefault("PERSEUS_DATA", os.path.expanduser("~/data/selective-fhe/perseus_data"))
    cal = "bootstrap_placements/vit_b16_224/pcmm_calib.json"      # the shipped PCMM calibration (run_prefill reads
    if os.path.exists(cal):                                       # it from --plan, which a capture does not pass)
        os.environ.setdefault("PF_PCMM_CALIB_IN", cal)
    from perseus.examples.vit import weights as V
    orig_rows, orig_embed = V.patch_rows, V.embed
    V.patch_rows = lambda px: orig_rows(px)[:a.tokens]
    V.embed = lambda store, rows: rows @ V.patch_matrix(store) + V.embed_offsets(store)[:len(rows)]
    from perseus.examples.vit.prefill import VitPrefill
    import numpy as np

    def prefill(self, rows, on_block=None):
        """VitPrefill.prefill with a cyclic pad to whole 128-token pages (the shipped pad assumes >= 64 real rows and
        leaves zero slots below that, which the planner refuses: 'no captured magnitude'). keys = the T' real rows."""
        from perseus.impl.prefill import Chunks
        from perseus.impl import blocked as BL
        from perseus.examples.gpt2.prefill import _tapped
        n = len(rows)
        full = -(-n // 128) * 128
        rows = np.vstack([rows] + [rows[(i * n) % n:][:min(n, full - n - i * n)] for i in range((full - n + n - 1) // n)])[:full]
        ch = Chunks(len(rows), self.blocks.t, bidir=True, keys=n)
        self.last_ch = ch
        self.start()
        self.token_begin(0)
        Xs = [x for row in BL.encode_blocked(self.rt, rows, ch, self.blocks) for x in row]
        stages = self.prefill_stages(ch)
        if on_block is not None:
            stages = [(lab, pre, _tapped(fn, b, on_block)) for b, (lab, pre, fn) in enumerate(stages)]
        Xs = self.run_stages(Xs, 0, stages)
        return Xs, ch, self.token_end(0)

    VitPrefill.prefill = prefill
    from perseus.examples.vit import run_prefill
    argv = ["run_prefill", "--fake", "--images", "scripts/repro/data/vit_eurosat", "--n", "1", "--capture", a.out,
            "--layers", str(a.layers), "--configs", "configs/model/approximation/vit_base_224/configs.json"]
    for kv in RUNTIME.split():
        argv += ["--set", kv]
    if a.structure_only:
        argv.append("--structure-only")
    sys.argv = argv
    print(f"[e2] T'={a.tokens} nch={a.nch} pad={a.pad} -> {a.out}", flush=True)
    return run_prefill.main()


if __name__ == "__main__":
    sys.exit(main())
