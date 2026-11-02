#!/usr/bin/env python
"""E25: collect the attacker views (Table 5 completeness), the leakage-vs-cost frontier, the S-only attacker and the
re-costed detector margins into e25/summary.json and e25/frontier.json (plot data). An operating point "near the
majority rate and clearly cheaper than full encryption" = attacker within 0.05 of the majority rate (the CI-free
criterion stated in NOTES.md) at <= 50% of the full model's cost.

  python selective/e25_summary.py
"""
import json
import os

OUT = os.path.expanduser("~/data/selective-fhe/e25")
E16 = os.path.expanduser("~/data/selective-fhe/e16")
FULL = {"drugsc": 14.40, "ages": 36.57}
LEVELS = {"drugsc": ("ours", "ours_w1", "ours_w3", "ours_s", "all"),
          "ages": ("ours", "ours_b1.25", "ours_b1.5", "ours_b2", "ours_b3", "ours_ub", "all")}
NAMES = {"ours": "private part only", "ours_w1": "+-1 word", "ours_w3": "+-3 words", "ours_s": "whole sentences",
         "all": "everything (full encryption)", "ours_b1.25": "face box x1.25", "ours_b1.5": "face box x1.5",
         "ours_b2": "face box x2", "ours_b3": "face box x3", "ours_ub": "upper body"}


def J(f):
    p = f"{OUT}/{f}"
    return json.load(open(p)) if os.path.exists(p) else None


def strip(v):
    return {k: x for k, x in v.items() if k != "correct"}


def main():
    s = dict(note="costs are s/input DERIVED from measured per-layout page costs (e16_leakage tables), not new encrypted "
                  "runs; attacker accuracies are measured (E16 protocol: start from the server's task model, 2 epochs on the "
                  "train split's views, validation accuracy with bootstrap 95% CI)", attackers={}, frontier={})
    for tag in ("drugsc", "drugsc_drug", "hatex", "ages", "ages_genface"):
        r = J(f"attr_{tag}.json")
        if r:
            s["attackers"][tag] = dict(majority=r["majority"], n_train=r["n_train"], n_val=r["n_val"],
                                       views={k: strip(v) for k, v in r["views"].items()})
    idt = json.load(open(f"{E16}/identity_ages.json"))
    clip = idt["models"]["openai/clip-vit-large-patch14-336"]
    s["attackers"]["ages_identity_clip"] = dict(chance=idt["chance"], note="E16 (unchanged): ours = face patches removed "
                                                "(the black / patch-removal view); genface = B6b's generated face",
                                                views={k: dict(top1=v["top1"]) for k, v in clip.items()})
    frontier = {}
    for task, attr_tag in (("drugsc", "drugsc"), ("ages", "ages")):
        att, ow = J(f"attr_{attr_tag}.json"), J(f"owte_frontier_{task}.json")
        drug = J("attr_drugsc_drug.json") if task == "drugsc" else None
        if not (att and ow):
            continue
        pts, ok = [], []
        for L in LEVELS[task]:
            a = att["views"].get(L)
            o = ow["levels"].get(L)
            if a is None or o is None:
                continue
            p = dict(level=L, name=NAMES[L], attacker=a["acc"], attacker_ci=a["ci"], majority=att["majority"],
                     owte_acc=o["owte"], encrypted=o["tokens"], s=o["s"], full_s=FULL[task], cost_frac=o["s"] / FULL[task])
            if drug and L in drug["views"]:
                p.update(drug_attacker=drug["views"][L]["acc"], drug_majority=drug["majority"])
            pts.append(p)
            if L != "all" and a["acc"] <= att["majority"] + 0.05 and o["s"] <= 0.5 * FULL[task]:
                ok.append(L)
        frontier[task] = dict(points=pts, operating_points=ok,
                              full_model_acc=ow.get("full") or ow["levels"]["all"].get("full"))
    s["frontier"] = frontier
    s["s_only"] = J("s_only.json")
    dm = J("detector_margins.json")
    if dm:
        s["detector"] = dict(note=dm["cost_note"], crc=dm["crc"], deployed=dm["deployed"], paper_targets=dm["paper_targets"])
    json.dump(frontier, open(f"{OUT}/frontier.json", "w"), indent=1)
    json.dump(s, open(f"{OUT}/summary.json", "w"), indent=1)
    for task, f in frontier.items():
        print(f"[e25] frontier {task}: operating points {f['operating_points'] or 'none'}")
        for p in f["points"]:
            print(f"   {p['name']:28s} attacker {p['attacker']:.3f} (maj {p['majority']:.3f})"
                  + (f" drug {p['drug_attacker']:.3f} (maj {p['drug_majority']:.3f})" if "drug_attacker" in p else "")
                  + f"  OWTE acc {p['owte_acc']:.3f}  encrypted {p['encrypted']:6.1f}  s {p['s']:6.2f} ({100 * p['cost_frac']:.0f}% of full)")
    for tag, a in s["attackers"].items():
        if "views" in a and "majority" in a:
            print(f"[e25] {tag:14s} maj {a['majority']:.3f} | " + "  ".join(f"{k} {v['acc']:.3f}" for k, v in a["views"].items()))


if __name__ == "__main__":
    main()
