#!/usr/bin/env python
"""E28 summary (CPU): privacy-utility points of the parametrised text hiding baselines per (task, backbone): accuracy per
level at test time (B5 / B5b on the one-way weights) and trained on noisy inputs (one level), OWTE for reference, OWTE
minus each (paired bootstrap 95% CI) and minus the best of them. Writes e28/summary.json, prints a table.

  python selective/e28_summary.py
"""
import json
import os

D = os.path.expanduser("~/data/selective-fhe")
E = f"{D}/e28"
METHODS = {"dp": ("SanText-style word substitution", "epsilon (stronger privacy = smaller)", "b5"),
           "custext": ("CusText word substitution (K = 20)", "epsilon (stronger privacy = smaller)", "b5b"),
           "emb": ("Embedding noise (Split-and-Denoise, no denoiser)", "noise norm / mean word distance (stronger "
                   "privacy = larger)", "b5b")}
GRID = {"dp": [1, 3, 10], "custext": [1, 3], "emb": [4, 8, 16]}


def J(f):
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    out = {"methods": {k: dict(name=v[0], level=v[1], grid_levels=GRID[k]) for k, v in METHODS.items()}, "results": {}}
    for task in ("drugsc", "mts"):
        for bk in ("bert-base", "bert-large"):
            r = {}
            files = {s: J(f"{E}/tt_{task}_{bk}_{s}.json") for s in ("b5", "b5b")}
            for m, (_n, _l, src) in METHODS.items():
                f = files[src]
                pts = []
                if f:
                    for k, v in f.items():
                        if isinstance(v, dict) and k.startswith(m) and k[len(m):].replace(".", "").isdigit():
                            pts.append(dict(level=float(k[len(m):]), acc=v["acc"], owte_minus=v["two_minus"],
                                            in_grid=float(k[len(m):]) in GRID[m]))
                pts.sort(key=lambda p: p["level"])
                tr = [J(f"{E}/{x}") for x in sorted(os.listdir(E)) if x.startswith(f"trained_{task}_{bk}_{m}")]
                r[m] = dict(test_time=pts, trained=[dict(level=t["level"], acc=t["acc"], owte_minus=t["owte_minus"])
                                                    for t in tr if t])
            owte = (files["b5"] or files["b5b"] or {}).get("two_stream")
            cands = [(p["acc"], f"{m}{p['level']:g}" + (" trained" if t else ""), p["owte_minus"])
                     for m in METHODS for t, ps in ((0, r[m]["test_time"]), (1, r[m]["trained"])) for p in ps]
            best = max(cands) if cands else None
            out["results"][f"{task}/{bk}"] = dict(owte=owte, n=(files["b5"] or {}).get("n"), methods=r,
                                                  best=dict(acc=best[0], name=best[1], owte_minus=best[2]) if best else None)
            print(f"\n### {task} {bk}: OWTE {owte}")
            for m in METHODS:
                print(f"  {m:8s} test-time: " + ", ".join(f"{p['level']:g}:{p['acc']:.3f}" for p in r[m]["test_time"])
                      + " | trained: " + ", ".join(f"{p['level']:g}:{p['acc']:.3f} (OWTE-it {100 * p['owte_minus'][0]:+.1f} "
                                                    f"[{100 * p['owte_minus'][1]:+.1f}, {100 * p['owte_minus'][2]:+.1f}])"
                                                    for p in r[m]["trained"]))
            if best:
                print(f"  best: {best[1]} {best[0]:.3f}, OWTE minus it {100 * best[2][0]:+.1f} "
                      f"[{100 * best[2][1]:+.1f}, {100 * best[2][2]:+.1f}]")
    json.dump(out, open(f"{E}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
