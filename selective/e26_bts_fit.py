#!/usr/bin/env python
"""E26: is encrypted time proportional to the number of bootstraps? Collects every measured page (median device time
per page, compiled bootstrap count incl. PCMM / CCMM outputs) from the E8-E22 BERT / ViT logs and the E18 Llama pages,
fits time = a * bts (+ b) per family, and writes the points, fits and residuals. CPU only.

  python selective/e26_bts_fit.py
"""
import glob
import json
import os
import re

import numpy as np

D = os.path.expanduser("~/data/selective-fhe")


def bert_vit_points():
    pts = []
    for log in sorted(glob.glob(f"{D}/e*/*.log") + glob.glob(f"{D}/e*/*/*.log")):
        txt = open(log, errors="ignore").read()
        m = re.search(r"\[e8-gpu\] compiled .*= (\d+) bts per page", txt)
        js = log[:-4] + ".json"
        if m and os.path.exists(js):
            try:
                d = json.load(open(js))
            except Exception:
                continue
            res = d["results"] if isinstance(d, dict) else d
            pw = sorted({r["page"]: r["page_wall"] for r in res if "page_wall" in r}.values())
            if len(pw) >= 2:
                pts.append(dict(family="BERT page (two-stream)", src=os.path.relpath(log, D), bts=int(m.group(1)),
                                s=float(np.median(pw[1:])), pages=len(pw)))
        for m in re.finditer(r"\[e11-gpu\] SUMMARY .*?s/page=([\d.]+) .*?bts/page=(\d+)", txt):
            pts.append(dict(family="ViT page (two-stream)", src=os.path.relpath(log, D), bts=int(m.group(2)),
                            s=float(m.group(1)), pages=None))
        walls = [(float(w), int(b)) for w, b in re.findall(r"\[bert\] SUMMARY .*? wall=([\d.]+)s .*?bts=(\d+)", txt)]
        if walls:
            pts.append(dict(family="BERT full model", src=os.path.relpath(log, D), bts=walls[0][1],
                            s=float(np.median([w for w, _b in walls])), pages=len(walls)))
    return pts


def llama_points():
    pts = []
    for p in range(5):
        d = json.load(open(f"{D}/e18/p2/gpu_fullL_p{p}.json"))
        r = d["results"][-1]
        pts.append(dict(family="Llama page", src=f"e18/p2/gpu_fullL_p{p}.json", bts=d["bts"],
                        s=r["wall"] - r["store_enc_s"], pages=1))
    for f in ("gpu_ts.json",):
        d = json.load(open(f"{D}/e18/p2/{f}"))
        pts.append(dict(family="Llama page", src=f"e18/p2/{f}", bts=d["bts"],
                        s=float(np.median([r["wall"] - r["store_enc_s"] for r in d["results"]])), pages=len(d["results"])))
    d = json.load(open(f"{D}/e18/p2/gpu_ts_e510_all20.json"))
    pts.append(dict(family="Llama page", src="e18/p2/gpu_ts_e510_all20.json", bts=d["bts"],
                    s=float(np.median([r["twin_s"] if False else r["wall"] - r["store_enc_s"] - r.get("stage_s", 0)
                                       for r in d["results"]])), pages=len(d["results"])))
    return pts


def fit(pts):
    x = np.array([p["bts"] for p in pts], float)
    y = np.array([p["s"] for p in pts], float)
    a0 = float((x @ y) / (x @ x))                                        # through the origin
    A = np.vstack([x, np.ones_like(x)]).T
    a1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
    r0 = y - a0 * x
    r2 = 1 - ((y - (a1 * x + b1)) ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(y) > 2 else float("nan")
    return dict(n=len(pts), ms_per_bts_origin=1000 * a0, rel_resid_origin_max=float(np.abs(r0 / y).max()),
                rel_resid_origin_median=float(np.median(np.abs(r0 / y))), ms_per_bts=1000 * float(a1),
                intercept_s=float(b1), r2=float(r2))


def main():
    pts = bert_vit_points() + llama_points()
    out = dict(points=pts, fits={})
    for fam in sorted({p["family"] for p in pts}):
        P = [p for p in pts if p["family"] == fam]
        if len(P) >= 2:
            out["fits"][fam] = fit(P)
            print(f"[e26-bts] {fam}: {out['fits'][fam]}", flush=True)
    os.makedirs(f"{D}/e26", exist_ok=True)
    json.dump(out, open(f"{D}/e26/bts_fit.json", "w"), indent=1)
    for p in [p for p in pts if p["family"] == "Llama page"]:
        print(f"  {p['src']}: {p['bts']} bts {p['s']:.1f} s -> {1000 * p['s'] / p['bts']:.2f} ms/bts")


if __name__ == "__main__":
    main()
