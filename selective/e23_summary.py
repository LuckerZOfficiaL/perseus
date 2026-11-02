#!/usr/bin/env python
"""E23 summary (CPU): per (task, backbone) the grid's OWTE and full model, the best test-time hiding method on the
one-way weights (the grid's number), the best hiding-aware model (trained on its own hidden inputs), the private-only
model, and late fusion (the client adds w * log p(public-only) + (1 - w) * log p(private-only), w tuned on the held 10%
of the train split; public-only = the delete / drop model), each with OWTE minus it (paired bootstrap 95% CI); the
one-way seeds 0-2 as mean and std. Writes e23/summary.json and prints a table per task.

  python selective/e23_summary.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e12_report import ci

D = os.path.expanduser("~/data/selective-fhe")
E = f"{D}/e23"
TASKS = {"drugsc": ("bert-base", "bert-large"), "mts": ("bert-base", "bert-large"),
         "ages": ("vit-base", "vit-large", "deit-base", "deit-small")}
HIDE = {"drugsc": ("mask", "delete", "typed"), "mts": ("mask", "delete", "typed"),
        "ages": ("black", "gray", "blur", "pixelate", "drop")}


def J(f):
    return json.load(open(f)) if os.path.exists(f) else None


def owte_ref(task, bk):
    if task == "ages":
        r = J(f"{D}/e21/vision/ages_{bk}_oneway.json")
        return dict(zip(range(r["n"]), zip(r["labels"], r["preds"]["two_stream"], r["preds"]["full"])))
    r = J(f"{E}/owte_{task}_{bk}.json")
    return dict(zip(r["rows"], zip(r["labels"], r["two_stream"], r["full"])))


def vs_owte(ref, rows, pred):
    y = np.array([ref[r][0] for r in rows])
    two = np.array([ref[r][1] for r in rows])
    m, lo, hi = ci(two == y, pred == y)
    return dict(acc=float((pred == y).mean()), owte=float((two == y).mean()), n=len(rows), owte_minus=[m, lo, hi])


def grid_test_time(task, bk):
    """The grid's best hiding method on the one-way weights (accuracy, name)."""
    if task == "ages":
        r = J(f"{D}/e21/vision/ages_{bk}_oneway.json")
        k = max(("redaction", "black", "gray", "blur", "pixelate", "inpaint", "genface"), key=lambda k: r["acc"][k])
        return r["acc"][k], k, r["acc"]["two_stream"], r["acc"]["full"]
    best = (None, None)
    for f in (f"{D}/e21/text/{task}_{bk}_oneway_b5.json", f"{D}/e21/text/{task}_{bk}_oneway_b5b.json"):
        for k, v in (J(f) or {}).items():
            if isinstance(v, dict) and "acc" in v and (best[0] is None or v["acc"] > best[0]):
                best = (v["acc"], k)
    w = J(f"{D}/e21/text/{task}_{bk}_oneway_acc.json")["with_private"]
    return best[0], best[1], w["two_stream"], w["full"]


def seeds(task, bk):
    out = []
    for s in (0, 1, 2):
        if task == "ages":
            r = J(f"{D}/e21/vision/ages_{bk}_oneway.json" if s == 0 else f"{E}/seeds/ages_{bk}_oneway_s{s}.json")
            if r is None:
                continue
            k = max(("redaction", "black", "gray", "blur", "pixelate", "inpaint", "genface"), key=lambda k: r["acc"][k])
            out.append((r["acc"]["two_stream"], r["acc"]["full"], r["acc"][k]))
        else:
            d, g = (f"{D}/e21/text", f"{bk}_oneway") if s == 0 else (f"{E}/seeds", f"{bk}_oneway_s{s}")
            a, b5, b5b = J(f"{d}/{task}_{g}_acc.json"), J(f"{d}/{task}_{g}_b5.json"), J(f"{d}/{task}_{g}_b5b.json")
            if not (a and b5 and b5b):
                continue
            hide = max(v["acc"] for f in (b5, b5b) for v in f.values() if isinstance(v, dict) and "acc" in v)
            out.append((a["with_private"]["two_stream"], a["with_private"]["full"], hide))
    if not out:
        return None
    x = 100 * np.array(out)
    gap = x[:, 0] - x[:, 2]
    return dict(n=len(out), owte=[float(x[:, 0].mean()), float(x[:, 0].std())], full=[float(x[:, 1].mean()), float(x[:, 1].std())],
                hide=[float(x[:, 2].mean()), float(x[:, 2].std())], owte_minus_hide=[float(gap.mean()), float(gap.std())],
                owte_minus_full=[float((x[:, 0] - x[:, 1]).mean()), float((x[:, 0] - x[:, 1]).std())])


def fusion(task, bk, ref):
    pub = J(f"{E}/{task}_{bk}_{'drop' if task == 'ages' else 'delete'}.json")
    prv = J(f"{E}/{task}_{bk}_private.json")
    if not (pub and prv):
        return None
    assert pub["rows"] == prv["rows"] and pub["held_labels"] == prv["held_labels"]
    hp, hq, yh = np.array(pub["logp_held"]), np.array(prv["logp_held"]), np.array(pub["held_labels"])
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    accs = [((w * hp + (1 - w) * hq).argmax(1) == yh).mean() for w in ws]
    w = float(ws[int(np.argmax(accs))])
    pred = (w * np.array(pub["logp_val"]) + (1 - w) * np.array(prv["logp_val"])).argmax(1)
    return dict(vs_owte(ref, pub["rows"], pred), w_public=w, held_acc=float(max(accs)))


def main():
    out = {}
    for task, bks in TASKS.items():
        out[task] = {}
        print(f"\n### {task}\n| Backbone | OWTE | Full | Best test-time hiding | Best trained hiding | OWTE - it [95% CI] | "
              f"Private-only | OWTE - it | Late fusion (w) | OWTE - it | Seeds: OWTE / full / hiding / OWTE - hiding |")
        print("|" + "---|" * 11)
        for bk in bks:
            ref = owte_ref(task, bk)
            tt_acc, tt_name, owte, full = grid_test_time(task, bk)
            row = dict(owte=owte, full=full, test_time_hiding=[tt_acc, tt_name])
            trained = {}
            for c in HIDE[task]:
                r = J(f"{E}/{task}_{bk}_{c}.json")
                if r:
                    trained[c] = vs_owte(ref, r["rows"], np.array(r["logp_val"]).argmax(1))
            if trained:
                b = max(trained, key=lambda c: trained[c]["acc"])
                row["trained_hiding"] = dict(trained[b], method=b, all={c: v["acc"] for c, v in trained.items()})
            r = J(f"{E}/{task}_{bk}_private.json")
            if r:
                row["private_only"] = vs_owte(ref, r["rows"], np.array(r["logp_val"]).argmax(1))
            row["late_fusion"] = fusion(task, bk, ref)
            row["seeds"] = seeds(task, bk)
            out[task][bk] = row
            f = lambda d: (f"{d['acc']:.3f}", f"{100 * d['owte_minus'][0]:+.1f} [{100 * d['owte_minus'][1]:+.1f}, "
                                              f"{100 * d['owte_minus'][2]:+.1f}]") if d else ("-", "-")
            th = row.get("trained_hiding")
            po, lf, sd = row.get("private_only"), row["late_fusion"], row["seeds"]
            print(f"| {bk} | {owte:.3f} | {full:.3f} | {tt_acc:.3f} ({tt_name}) | "
                  + (f"{th['acc']:.3f} ({th['method']})" if th else "-") + f" | {f(th)[1]} | {f(po)[0]} | {f(po)[1]} | "
                  + (f"{lf['acc']:.3f} ({lf['w_public']:.2f})" if lf else "-") + f" | {f(lf)[1]} | "
                  + (f"{sd['owte'][0]:.1f}±{sd['owte'][1]:.1f} / {sd['full'][0]:.1f}±{sd['full'][1]:.1f} / "
                     f"{sd['hide'][0]:.1f}±{sd['hide'][1]:.1f} / {sd['owte_minus_hide'][0]:+.1f}±{sd['owte_minus_hide'][1]:.1f} "
                     f"(n={sd['n']})" if sd else "-") + " |")
    json.dump(out, open(f"{E}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
