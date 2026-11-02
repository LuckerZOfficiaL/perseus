#!/usr/bin/env python
"""E25 task 4: E4's detector-margin chains re-costed with the CURRENT measured per-layout costs (e16_leakage
TEXT_COST / FACE_COST: text <= 13 private pieces 1.78 s, <= 29 5.51 s, else the full model 14.40 s; photos <= 20
patches 2.51 s, <= 31 5.54 s, <= 63 10.76 s, else 36.57 s; costs DERIVED from measured page costs), with per-input
any-miss shares and conformal risk control (Angelopoulos et al. 2024) for two monotone losses: the expected missed
fraction (as E4) and any-miss (1 if any private piece / >10% of a face is left in plaintext). Same chains, targets and
200 random calibration / test halves as E4. CPU only.

  python selective/e25_detector.py        (sat venv) -> ~/data/selective-fhe/e25/detector_margins.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_detector_recall as E4
from e16_leakage import FACE_COST, FULL_BERT, FULL_VIT, TEXT_COST, cost

OUT = os.path.expanduser("~/data/selective-fhe/e25")
# E4's text chain, labelled: "deployed kinds" is the tagger WITHOUT the condition list (52% missed on drug reviews);
# the detector of the paper's experiments is "+condition names".
TEXT_LABEL = {"deployed kinds": "tagger only (no condition list)", "+condition names": "+condition names (paper's detector)"}
PAPER_TARGET = {"faces_imdb_d320_min16": 0.01, "faces_wider_d1280_min16": 0.05, "text_cadec": 0.10, "text_drugs": 0.05}


def picks(L, C, alpha, splits=200, seed=0):
    rng = np.random.default_rng(seed)
    pick, te_risk, te_cost = [], [], []
    for _ in range(splits):
        perm = rng.permutation(len(L))
        cal, te = perm[:len(L) // 2], perm[len(L) // 2:]
        j = E4.crc(L, alpha, cal)
        pick.append(j)
        te_risk.append(L[te, j].mean())
        te_cost.append(C[te, j].mean())
    return int(np.bincount(pick).argmax()), float(np.mean(te_risk)), float(np.percentile(te_risk, 95)), float(np.mean(te_cost))


def table(ref, losses, labels, costfn, deployed_j=None):
    L = np.array([x["L"] for x in losses])
    A = np.array([x["A"] for x in losses])
    K = np.array([x["K"] for x in losses])
    C = np.vectorize(costfn)(K).astype(float)
    rows = []
    for j, lab in enumerate(labels):
        rows.append(dict(reference=ref, setting=lab, setting_index=j, deployed=(j == deployed_j), n=len(losses),
                         miss=float(L[:, j].mean()), any_miss=float(A[:, j].mean()), private=float(K[:, j].mean()),
                         s_input=float(C[:, j].mean()), picks=[]))
    crc = {}
    for lname, M in (("miss", L), ("any", A)):
        for alpha in E4.ALPHAS:
            j, r, r95, c = picks(M, C, alpha)
            crc[f"{lname}@{alpha}"] = dict(setting=labels[j], setting_index=j, test_risk=r, test_risk_p95=r95, s_input=c,
                                           test_any_miss=float(A[:, j].mean()), test_miss=float(L[:, j].mean()))
            rows[j]["picks"].append(f"{lname}<={alpha}")
    return rows, crc


def main():
    os.makedirs(OUT, exist_ok=True)
    fc = lambda k: cost(k, FACE_COST, FULL_VIT)
    tc = lambda k: cost(k, TEXT_COST, FULL_BERT)
    face_labels = [f"score {t}, grow {g}, dilate {d}" for t, g, d in E4.FACE_CHAIN] + ["all (full encryption)"]
    text_labels = [TEXT_LABEL.get(c[0], c[0]) for c in E4.TEXT_CHAIN] + ["all (full encryption)"]
    out = dict(cost_note="s/input DERIVED from the current measured per-layout page costs (e16_leakage TEXT_COST / "
                         "FACE_COST); E4's own report used stale tables (text: only <= 13 -> 1.77 s, else full)",
               paper_targets=PAPER_TARGET, rows=[], crc={}, deployed={})
    for name in ("imdb", "wider"):
        rows = json.load(open(f"{E4.OUT}/faces_{name}.json"))
        for det in ("d320", "d1280"):
            for min_px in ((16,) if name == "imdb" else (16, 32)):
                ref = f"faces_{name}_{det}_min{min_px}"
                r, c = table(ref, E4.face_losses(rows, det, min_px), face_labels, fc,
                             E4.FACE_CHAIN.index((0.7, 0.1, 0)) if det == "d320" else None)
                out["rows"] += r
                out["crc"][ref] = c
    for name in ("cadec", "drugs"):
        ls = E4.text_losses(json.load(open(f"{E4.OUT}/text_{name}.json")))
        ref = f"text_{name}"
        r, c = table(ref, ls, text_labels, tc, 1)                 # +condition names = the paper's detector
        out["rows"] += r
        out["crc"][ref] = c
        d = np.array([x["dep"] for x in ls])
        out["deployed"][ref] = dict(note="E15's deployed detector function (as stored by E4)", miss=float(d[:, 0].mean()),
                                    any_miss=float(d[:, 1].mean()), private=float(d[:, 2].mean()),
                                    s_input=float(np.mean([tc(k) for k in d[:, 2]])))
    json.dump(out, open(f"{OUT}/detector_margins.json", "w"), indent=1)
    for ref, c in out["crc"].items():
        tgt = PAPER_TARGET.get(ref)
        for key, v in c.items():
            mark = " <- paper target" if tgt is not None and key.endswith(f"@{tgt}") else ""
            print(f"[e25] {ref:26s} CRC {key:9s} {v['setting']:38s} risk {v['test_risk']:.4f} (p95 {v['test_risk_p95']:.4f}) "
                  f"any-miss {v['test_any_miss']:.3f} miss {v['test_miss']:.4f} s/input {v['s_input']:6.2f}{mark}")
    for ref, d in out["deployed"].items():
        print(f"[e25] {ref} deployed function: miss {d['miss']:.4f} any-miss {d['any_miss']:.3f} private {d['private']:.1f} "
              f"s/input {d['s_input']:.2f}")
    for r in out["rows"]:
        if r["deployed"] or r["picks"]:
            print(f"[e25] {r['reference']:26s} {r['setting']:38s} miss {r['miss']:.4f} any {r['any_miss']:.3f} "
                  f"private {r['private']:6.1f} s {r['s_input']:6.2f} {'DEPLOYED ' if r['deployed'] else ''}{r['picks']}")


if __name__ == "__main__":
    main()
