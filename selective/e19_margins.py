#!/usr/bin/env python
"""E19: E15's tasks with the private sets of E4's calibrated margins, so accuracy and cost hold with the miss-rate
guarantee in force. Ages: YuNet at score 0.3, boxes grown by 20% (E4's conformal pick for <= 1% missed face area on
IMDB-WIKI); drug reviews: the fixed detector + NER Sign_symptom (E4's pick for <= 5% missed health words). Plaintext
accuracy of full / two-stream / redaction with the one-way models (the encrypted answers equal plaintext, E12-E15), and
the encrypted s / input from E15's measured page shapes (more than the largest shape: the full model).

  python selective/e19_margins.py --task ages      (anon venv, cv2; GPU if visible)
  python selective/e19_margins.py --task drugs     (sat venv)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e16_leakage import FACE_COST, FULL_BERT, FULL_VIT, TEXT_COST, cost

OUT = os.path.expanduser("~/data/selective-fhe/e19")


def ages(a):
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification
    import e1_oneway_vit as EV
    import e11_faces as E11
    import e15_ages as E15
    from e12_report import ci
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = json.load(open(f"{E15.OUT}/ages.json"))["validation"]
    old = json.load(open(f"{E15.OUT}/faces.json"))["detections"]
    files = [f"{E15.OUT}/{rel}" for rel, _l, _a in val]
    det = E11.detect(files, a.score, a.grow)
    y = np.array([lab for _r, lab, _a in val])
    res = {}
    for mname in a.models.split(","):
        vit = EV.Vit(ViTForImageClassification.from_pretrained(os.path.expanduser(mname)).eval().to(dev))
        P = {k: [] for k in ("full", "two_stream", "redaction", "two_stream_deployed")}
        K, Kd = [], []
        for (rel, _l, _a), f in zip(val, files):
            S_new = det.get(f, {}).get("patches") or list(range(196))    # no face found: encrypt the whole photo
            S_old = old[rel]["patches"]
            K.append(len(S_new))
            Kd.append(len(S_old))
            px = EV.preprocess([Image.open(f).convert("RGB")]).to(dev)
            with torch.no_grad():
                P["full"].append(int(vit.full(px)[0].argmax()))
                lr, lo = vit.two_stream(px, torch.tensor([S_new], device=dev), "drop") if len(S_new) < 196 \
                    else (vit.full(px)[0] * 0, vit.full(px)[0])
                P["two_stream"].append(int(lo.argmax()))
                P["redaction"].append(int(lr.argmax()) if len(S_new) < 196 else -1)
                _lr, lo = vit.two_stream(px, torch.tensor([S_old], device=dev), "drop")
                P["two_stream_deployed"].append(int(lo.argmax()))
        P = {k: np.array(v) for k, v in P.items()}
        tag = os.path.basename(os.path.normpath(mname))
        s_new = np.mean([cost(k, FACE_COST, FULL_VIT) for k in K])
        s_old = np.mean([cost(k, FACE_COST, FULL_VIT) for k in Kd])
        m, lo_, hi = ci(P["two_stream"] == y, P["redaction"] == y)
        res[tag] = dict({k: float((v == y).mean()) for k, v in P.items()}, patches=float(np.mean(K)),
                        patches_deployed=float(np.mean(Kd)), s=float(s_new), s_deployed=float(s_old),
                        two_minus_redaction=[m, lo_, hi], over_63=int(sum(k > 63 for k in K)))
        print(f"[e19] ages {tag}: " + ", ".join(f"{k} {(v == y).mean():.3f}" for k, v in P.items()) +
              f" | private patches {np.mean(K):.1f} (deployed {np.mean(Kd):.1f}), s/photo {s_new:.2f} (deployed "
              f"{s_old:.2f}), {sum(k > 63 for k in K)} over 63 | two-stream minus redaction {m:+.3f} [{lo_:+.3f}, {hi:+.3f}]",
              flush=True)
    return res


def drugs(a):
    from transformers import AutoTokenizer, pipeline
    from datasets import load_dataset
    import e13_hatexplain as H13
    import e15_drugs as E15
    D = os.path.expanduser("~/data/selective-fhe/e15")
    data = json.load(open(f"{D}/drugs_t128.json"))
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    ds = load_dataset("lewtun/drug-reviews")
    lexicon, cond = E15.lexicon_of(ds), E15.phrases(E15.condition_names(ds))
    ner = pipeline("token-classification", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=-1)
    texts = [t for t, _l in data["validation"]]
    spans = E15.detect(texts, lexicon, ner, 0.5, cond)
    for t, sp, es in zip(texts, spans, ner(texts, batch_size=64)):
        sp += [(*E15.widen(t, int(e["start"]), int(e["end"])), "Sign_symptom") for e in es
               if e["entity_group"] == "Sign_symptom" and e["score"] >= 0.5]
    pos = {}
    for j, (t, sp) in enumerate(zip(texts, spans)):
        enc = tok(t, truncation=True, max_length=128, return_offsets_mapping=True)
        pos[str(j)] = [k for k, (c0, c1) in enumerate(enc["offset_mapping"])
                       if c1 > c0 and any(c0 < e1 and c1 > e0 for e0, e1, _ in sp)]
    priv = {"validation": pos}
    json.dump(priv, open(f"{OUT}/drugs_t128cs_private.json", "w"))
    k = [len(v) for v in pos.values() if v]
    s = float(np.mean([cost(x, TEXT_COST, FULL_BERT) for x in k]))
    print(f"[e19] drugs: {len(k)} reviews with private data, median {np.median(k):.0f} private tokens, "
          f"{sum(x > 29 for x in k)} over 29, s/review {s:.2f}", flush=True)
    res = dict(reviews=len(k), private_median=float(np.median(k)), s=s)
    for mname in a.models.split(","):
        tag = os.path.basename(os.path.normpath(mname))
        H13.accuracy(os.path.expanduser(mname), tok, data, priv, f"{OUT}/drugs_cs_accuracy_{tag}.json", max_len=128,
                     focus=("birth_control", 0))
        res[tag] = json.load(open(f"{OUT}/drugs_cs_accuracy_{tag}.json"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("ages", "drugs"))
    ap.add_argument("--models", default=None)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--grow", type=float, default=0.2)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    E = os.path.expanduser("~/data/selective-fhe/e15")
    if a.task == "ages":
        a.models = a.models or f"{E}/ages_oneway,{E}/vit_age"
        res = ages(a)
    else:
        a.models = a.models or f"{E}/drugs_oneway,{E}/drugs_bert"
        res = drugs(a)
    json.dump(res, open(f"{OUT}/{a.task}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
