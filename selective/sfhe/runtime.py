"""Runtime switches. Our circuits run with upstream's CURRENT plan runtime (backend, bootstrap variants, PCMM raises:
their optimizations carry over on a pull) except the layout switches our code is written against, pinned here. The
full-model baselines run upstream's own runtime untouched (they never import sfhe).

  PF_QKV_PACK    upstream packs V's last block onto K's in the fused QKV output; our blocks split it K | V | Q
  PF_ATTN_PACK, PF_ATTN_BSGS, PF_SM_HPACK
                 internals of upstream's blocked attention (lane rotations, softmax head packing) that our route-B
                 attention reuses (impl.prefill.lane_rotate / recip_lanes)
  PF_LAST_QUERY  the last block computes the last query only: our stages read every encrypted token
"""
from __future__ import annotations

import os

PINS = {"PF_QKV_PACK": "0", "PF_ATTN_PACK": "0", "PF_ATTN_BSGS": "0", "PF_SM_HPACK": "0", "PF_LAST_QUERY": "0"}


def apply_pins():
    """Set the pins in this process (before any perseus module that reads them at import)."""
    os.environ.update(PINS)


def _plan_cmd(plan):
    from perseus._paths import ASSETS
    d = plan if os.path.isabs(plan) or os.path.exists(plan) else os.path.join(ASSETS, plan)
    return open(os.path.join(d, "PLAN_CMD.txt")).read().splitlines()


def plan_runtime_line(upstream_plan):
    """The `runtime=` line for one of our plans: upstream's plan's runtime with the pins replacing their switches."""
    line = ""
    for ln in _plan_cmd(upstream_plan):
        if ln.startswith("runtime="):
            line = ln[len("runtime="):]
    kv = [x for x in line.split() if x.split("=", 1)[0] not in PINS]
    return "runtime=" + " ".join(kv + [f"{k}={v}" for k, v in PINS.items()])


def plan_recipe(upstream_plan):
    """upstream's planner recipe of `upstream_plan` (its PLAN_CMD `recipe=` line)."""
    for ln in _plan_cmd(upstream_plan):
        if ln.startswith("recipe="):
            return ln[len("recipe="):]
    raise KeyError(f"{upstream_plan}: no recipe= line in PLAN_CMD.txt")


if __name__ == "__main__":                 # selective/*.sh: python -m sfhe.runtime {runtime|recipe} UPSTREAM_PLAN
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
    print(plan_runtime_line(sys.argv[2])[len("runtime="):] if sys.argv[1] == "runtime" else plan_recipe(sys.argv[2]))
