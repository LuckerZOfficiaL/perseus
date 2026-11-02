#!/usr/bin/env python
"""E2 summary: op counts and planned bootstraps per captured ViT configuration, plus a time estimate calibrated on
the shipped 197-token run (59.0 s per image on one RTX PRO 6000, README). Per-op device costs from the README's
perseus-core table (ms, batch of 1): bootstrap dense 20.3, sparse 17.4; relinearized mult 0.540; rotation 0.512;
square ~ mult; add 0.022; plaintext mult 0.019; conjugate ~ rotation.

  python selective/e2_summarize.py vit_b16_224=graphs/vit_b16_224:bootstrap_placements/vit_b16_224 \
      T17=~/data/selective-fhe/e2/graphs/vit_T17_nch512:bootstrap_placements/selective_e2/vit_T17_nch512
"""
import collections
import glob
import json
import os
import sys

MS = dict(auto_bootstrap=20.3, deliberate_bootstrap=20.3, mult=0.540, square=0.540, rotate=0.512, rotate_sum=0.512,
          conjugate=0.512, mult_i=0.019, add=0.022, add_inplace=0.022, sub_ct=0.022, negate=0.022, hint=0.0, level_hint=0.0)
SHIPPED_S = 59.0


def counts(graph_dir):
    tot = collections.Counter()
    for f in sorted(glob.glob(os.path.join(graph_dir, "block_*/graph.json"))):
        for n in json.load(open(f))["nodes"]:
            tot[n["op_type"]] += 1
    return tot


def planned_bts(plan_dir):
    """The plan's own total per block summary (placed refreshes + hint bootstraps + deliberate fixed points); matches
    the bootstrap count the GPU runner reports (2116 full model, 694 two-stream route B). Returns (total, deliberate)."""
    tot, delib = 0, 0
    for f in sorted(glob.glob(os.path.join(plan_dir, "block_*_placement.json"))):
        s = json.load(open(f))["summary"]
        tot += s["total_bootstraps"]
        delib += s["num_deliberate_bootstrap_nodes"]
    return tot, delib


def est_ms(c, bts):
    """planned bootstraps at the dense price + every other recorded op at its price."""
    return bts * MS["auto_bootstrap"] + sum(MS.get(k, 0.5) * v for k, v in c.items() if "bootstrap" not in k)


def main():
    rows = []
    for arg in sys.argv[1:]:
        name, rest = arg.split("=", 1)
        g, p = (os.path.expanduser(x) for x in rest.split(":"))
        c = counts(g)
        b, sp = planned_bts(p) if os.path.isdir(p) else (None, None)
        rows.append((name, c, b, sp))
    ref = next((r for r in rows if r[0] == "vit_b16_224"), rows[0])
    ref_ms = est_ms(ref[1], ref[2] or 0)
    scale = SHIPPED_S * 1000 / ref_ms
    print(f"calibration: {ref[0]} estimated {ref_ms/1000:.1f} s from counts vs {SHIPPED_S} s measured -> scale x{scale:.2f}\n")
    print("| Config | Nodes | Mult+square | Rotations | Recorded bts | Planned bts (incl. kept deliberate) | Est. s/image (calibrated) | vs shipped |")
    print("|---|---|---|---|---|---|---|---|")
    for name, c, b, sp in rows:
        ms = est_ms(c, b or 0) * scale
        print(f"| {name} | {sum(c.values())} | {c['mult'] + c['square']} | {c['rotate'] + c['rotate_sum'] + c['conjugate']} | "
              f"{c['auto_bootstrap'] + c['deliberate_bootstrap']} | {b if b is not None else 'n/a'} | {ms/1000:.1f} | "
              f"{SHIPPED_S / (ms/1000):.2f}x |")


if __name__ == "__main__":
    main()
