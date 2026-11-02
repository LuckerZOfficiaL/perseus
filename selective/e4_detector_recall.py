#!/usr/bin/env python
"""E4 (the client's detector misses things): how much of the private part the client's detector leaves in plaintext,
against human or independent labels, and what a safety margin costs. Faces: OpenCV YuNet as in E11 / E15 (deployed:
score 0.7, boxes grown by 10%, on the 320 px photo); references are WIDER FACE val (human boxes) and IMDB-WIKI test
photos (the dataset's own face box, an independent detector; the photos are NOT filtered by YuNet, unlike E15's). Text:
E15's drug detector (drug lexicon + biomedical NER, Medication / Disease_disorder, score >= 0.5); references are CADEC
(AskaPatient drug reviews, human Drug / Disease spans) and E15's drug reviews themselves (silver: the review's own drug
name and every condition word, string matches).

Margins form one chain per detector, each setting encrypting a superset of the previous: lower detector thresholds, then
grown boxes / neighbouring words, then everything. A miss = private area (faces: face-box pixels at the 224 px model
input; text: word pieces) left in plaintext. Conformal risk control (Angelopoulos et al. 2024) picks the least
conservative setting whose calibrated miss rate is <= alpha, over random calibration / test halves. Cost = E15's
encrypted s per input by private-token count (faces: buckets s / m / l, more = full ViT; text: <= 13 tokens, more = full
BERT). CPU only.

  python selective/e4_detector_recall.py --faces wider      (sat venv: cv2)
  python selective/e4_detector_recall.py --faces imdb --n 2000
  python selective/e4_detector_recall.py --text cadec
  python selective/e4_detector_recall.py --text drugs
  python selective/e4_detector_recall.py --report
"""
import argparse
import glob
import io
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e4")
WIDER = f"{OUT}/raw/wider/data"
CADEC = f"{OUT}/raw/akramRedjdal/data"
FACE_COST = ((0, 0.0), (20, 2.51), (31, 5.54), (63, 10.75))      # E15 encrypted s per photo by private patches
FULL_VIT = 36.57
TEXT_COST = ((0, 0.0), (13, 1.77))                               # E15 drugs: <= 13 private pieces on a page of 12
FULL_BERT = 14.40
FLOOR = 0.3                                                      # YuNet score floor; higher thresholds filter it
FACE_CHAIN = ([(t, 0.1, 0) for t in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)] + [(0.3, g, 0) for g in (0.2, 0.3, 0.5)]
              + [(0.3, 0.5, 1), (0.3, 0.5, 2)])                  # (score, grow, patch dilation); deployed (0.7, 0.1, 0)
TEXT_CHAIN = (("deployed kinds", 0.5, 0, 0, 0), ("+condition names", 0.5, 1, 0, 0), ("+symptoms", 0.5, 1, 1, 0),
              ("+misspellings", 0.5, 1, 1, 1), ("score 0.1", 0.1, 1, 1, 1), ("score 0.01", 0.01, 1, 1, 1),
              ("+1 word", 0.01, 1, 1, 1, 1), ("+2 words", 0.01, 1, 1, 1, 2))
# (name, NER score, condition lexicon, NER Sign_symptom too, edit-distance-1 lexicon matches[, neighbouring words]):
# each adds to the previous; the drug lexicon and NER Medication / Disease_disorder are always on.
ALPHAS = (0.1, 0.05, 0.01)


def cost(k, table, full):
    for kmax, s in table:
        if k <= kmax:
            return s
    return full


# ---------------------------------------------------------------- faces

_YUNET = []


def yunet(im):
    """[(x0, y0, x1, y1, score)] of the faces YuNet finds in a BGR image (score >= FLOOR), as fractions of the photo."""
    import cv2
    import e11_faces as E11
    if not _YUNET:
        _YUNET.append(cv2.FaceDetectorYN.create(E11.YUNET, "", (320, 320), FLOOR))
    h, w = im.shape[:2]
    _YUNET[0].setInputSize((w, h))
    _n, faces = _YUNET[0].detect(im)
    return [] if faces is None else [(float(x / w), float(y / h), float((x + bw) / w), float((y + bh) / h), float(s))
                                     for x, y, bw, bh, s in faces[:, [0, 1, 2, 3, 14]]]


def detections(im):
    """YuNet on the photo as deployed (320 px thumbnail) and at up to 1280 px."""
    import cv2
    out = {}
    for name, side in (("d320", 320), ("d1280", 1280)):
        h, w = im.shape[:2]
        f = min(1.0, side / max(h, w))
        x = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else im
        out[name] = yunet(x)
    return out


def build_wider(a):
    import cv2
    lines = open(f"{WIDER}/wider_face_split/wider_face_val_bbx_gt.txt").read().split("\n")
    rows, i = [], 0
    while i < len(lines) and lines[i].strip():
        f, n = lines[i].strip(), int(lines[i + 1])
        boxes = [list(map(float, l.split()[:10])) for l in lines[i + 2:i + 2 + max(n, 1)]] if n else []
        i += 2 + max(n, 1)
        im = cv2.imread(f"{WIDER}/WIDER_val/images/{f}")
        h, w = im.shape[:2]
        gold = [(x / w, y / h, (x + bw) / w, (y + bh) / h) for x, y, bw, bh, *_r, inv, _o, _p in boxes
                if inv == 0 and bw > 0 and bh > 0]
        rows.append(dict(file=f, size=[w, h], gold=gold, **detections(im)))
        if len(rows) % 500 == 0:
            print(f"[e4] wider {len(rows)}", flush=True)
    return rows


def build_imdb(a):
    import cv2
    from datasets import Image as ImageFeature, load_dataset
    tok = open(os.path.expanduser("~/.credentials/hf_token")).read().strip()
    ds = load_dataset("systemk-ai/imdb-wiki", "default", split="test", streaming=True, token=tok).select_columns(
        ["age", "face_location", "face_score", "second_face_score", "image"]).cast_column("image", ImageFeature(decode=False))
    rows = []
    for j, r in enumerate(ds):                                   # E15's filters, minus the YuNet one
        if len(rows) >= a.n:
            break
        second, fs = r["second_face_score"], r["face_score"]
        if r["age"] is None or not 15 <= r["age"] <= 80 or (second is not None and not math.isnan(second)) \
                or fs is None or not fs > 0:
            continue
        im = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        x0, y0, x1, y1 = r["face_location"]
        rows.append(dict(file=j, size=[w, h], gold=[(x0 / w, y0 / h, x1 / w, y1 / h)], **detections(im)))
        if len(rows) % 500 == 0:
            print(f"[e4] imdb {len(rows)} of {j + 1} streamed", flush=True)
    return rows


def patch_mask(boxes, grow, dil):
    """14 x 14 mask of the patches E11's rule makes private (overlap with a box grown by `grow`), dilated by `dil`."""
    m = np.zeros((14, 14), bool)
    e = np.arange(15) * 16
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        X0, X1 = (x0 - grow * bw) * 224, (x1 + grow * bw) * 224
        Y0, Y1 = (y0 - grow * bh) * 224, (y1 + grow * bh) * 224
        m |= np.outer((e[:-1] < Y1) & (e[1:] > Y0), (e[:-1] < X1) & (e[1:] > X0))
    for _ in range(dil):
        p = np.pad(m, 1)
        m = np.max([p[1 + dr:15 + dr, 1 + dc:15 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
    return m


def overlap(box):
    """14 x 14 pixel areas of a face box (224 px coordinates) inside each patch."""
    x0, y0, x1, y1 = (np.clip(v * 224, 0, 224) for v in box)
    e = np.arange(15) * 16
    oy = np.clip(np.minimum(e[1:], y1) - np.maximum(e[:-1], y0), 0, None)
    ox = np.clip(np.minimum(e[1:], x1) - np.maximum(e[:-1], x0), 0, None)
    return np.outer(oy, ox)


def face_losses(rows, det, min_px):
    """Per photo with a reference face of side >= min_px at 224 px, per chain setting: missed face area fraction, whether
    some face is > 10% exposed, private patches."""
    out = []
    for r in rows:
        ov = [overlap(b) for b in r["gold"]]
        ov = [o for o, b in zip(ov, r["gold"]) if min((b[2] - b[0]) * 224, (b[3] - b[1]) * 224) >= min_px and o.sum() > 0]
        if not ov:
            continue
        area = sum(o.sum() for o in ov)
        L, A, K = [], [], []
        for t, g, d in FACE_CHAIN:
            m = patch_mask([b[:4] for b in r[det] if b[4] >= t], g, d)
            miss = [o.sum() - (o * m).sum() for o in ov]
            L.append(sum(miss) / area)
            A.append(float(max(mi / o.sum() for mi, o in zip(miss, ov)) > 0.1))
            K.append(int(m.sum()))
        out.append(dict(L=L + [0.0], A=A + [0.0], K=K + [196]))
    return out


# ---------------------------------------------------------------- text

CONDITION_WORDS = r"birth control|contraceptives?|depression|depressed|pain|anxiety|anxious|acne|bipolar|insomnia|" \
                  r"weight loss|obesity|obese|adhd|diabetes|diabetic"


def words_of(t):
    return [(m.start(), m.end()) for m in re.finditer(r"[^\W_]+", t)]


def text_rows(name):
    """[(text the model sees, [gold char spans])]."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    out = []
    if name == "cadec":
        import pandas as pd
        for f in sorted(glob.glob(f"{CADEC}/*.parquet")):
            for toks, labs in zip(*pd.read_parquet(f)[["tokens", "labels"]].values.T):
                t, sp, c = "", [], 0
                for w, l in zip(toks, labs):
                    if t:
                        t += " "
                    if l[2:] in ("Drug", "Disease"):
                        sp.append((len(t), len(t) + len(w)))
                    t += w
                enc = tok(t, truncation=True, max_length=128, return_offsets_mapping=True)
                end = max(c1 for c0, c1 in enc["offset_mapping"])
                out.append((t[:end], [(s, e) for s, e in sp if s < end]))
        return out
    from datasets import load_dataset
    import e15_drugs as E15
    D = os.path.expanduser("~/data/selective-fhe/e15")
    texts = [t for t, _l in json.load(open(f"{D}/drugs_t128.json"))["validation"]]
    ds = load_dataset("lewtun/drug-reviews")["test"]               # the reviews' own drug names, by text
    drug = {E15.clean(r)[:200]: d for r, d in zip(ds["review"], ds["drugName"])}
    for t in texts:
        names = [p.strip() for p in re.split(r"[/,]", drug.get(t[:200], "")) if len(p.strip()) >= 3]
        rx = re.compile(r"(?<![\w])(?:" + "|".join([CONDITION_WORDS] + [re.escape(n) for n in names]) + r")(?![\w])", re.I)
        out.append((t, [E15.widen(t, m.start(), m.end()) for m in rx.finditer(t)]))
    return out


def deletes(w):
    return {w} | {w[:i] + w[i + 1:] for i in range(len(w))}


def build_text(name):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from datasets import load_dataset
    import e15_drugs as E15
    rows = text_rows(name)
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    deployed = E15.detector()
    ds = load_dataset("lewtun/drug-reviews")
    lexicon = E15.lexicon_of(ds)
    conds = E15.condition_names(ds)
    clex = E15.phrases(conds)
    singles = [n for n in E15.drug_names(ds) + conds if n.isalpha() and len(n) >= 5]
    near = {}                                                    # symmetric deletes: edit distance <= 1
    for n in singles:
        for v in deletes(n.lower()):
            near.setdefault(v, n)
    ntok = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    ner = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all").eval()
    keep = [i for l, i in ner.config.label2id.items() if l[2:] in E15.NER_KINDS]
    symp = [i for l, i in ner.config.label2id.items() if l[2:] == "Sign_symptom"]
    out = []
    for j in range(0, len(rows), 32):
        chunk = rows[j:j + 32]
        texts = [t for t, _g in chunk]
        dep = deployed(texts)
        enc = ntok(texts, return_offsets_mapping=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
        off = enc.pop("offset_mapping")
        with torch.no_grad():
            pr = ner(**enc).logits.softmax(-1)
            p, q = pr[..., keep].sum(-1), pr[..., keep + symp].sum(-1)
        for b, (t, gold) in enumerate(chunk):
            W = words_of(t)
            ob = off[b].tolist()
            sc = lambda P: [max([float(P[b, i]) for i, (c0, c1) in enumerate(ob) if c1 > c0 and c0 < e and c1 > s]
                                or [0.0]) for s, e in W]
            on = lambda rx: [any(s < m.end() and e > m.start() for m in ms) for s, e in W] if (ms := list(rx.finditer(t))) \
                else [False] * len(W)
            fuzzy = [len(w) >= 5 and w.isalpha() and bool(deletes(w) & near.keys())
                     for w in (t[s:e].lower() for s, e in W)]
            bert = tok(t, truncation=True, max_length=128, return_offsets_mapping=True)["offset_mapping"]
            pieces = [(c0, c1) for c0, c1 in bert if c1 > c0]
            hit = lambda spans: [int(any(c0 < e and c1 > s for s, e in spans)) for c0, c1 in pieces]
            out.append(dict(words=W, score=sc(p), score_symp=sc(q), lex=on(lexicon), lex_cond=on(clex), fuzzy=fuzzy,
                            pieces=pieces, gold=hit(gold),
                            deployed=hit([(s, e) for s, e, _k in dep[b]])))
        print(f"[e4] {name} {min(j + 32, len(rows))}/{len(rows)}", flush=True)
    return out


def text_losses(rows):
    """Per text with gold pieces, per chain setting (+ encrypt-all): missed gold fraction, any miss, private pieces."""
    out = []
    for r in rows:
        g = np.array(r["gold"], bool)
        if not g.any():
            continue
        W, P = r["words"], r["pieces"]
        wp = [[k for k, (c0, c1) in enumerate(P) if c0 < e and c1 > s] for s, e in W]
        L, A, K = [], [], []
        for _name, t, cond, symp, fz, *d in TEXT_CHAIN:
            d = d[0] if d else 0
            S = np.array(r["score_symp"] if symp else r["score"])
            on = (S >= t) | np.array(r["lex"], bool)
            on |= np.array(r["lex_cond"], bool) & bool(cond)
            on |= np.array(r["fuzzy"], bool) & bool(fz)
            for _ in range(d):
                on = on | np.r_[on[1:], False] | np.r_[False, on[:-1]]
            m = np.zeros(len(P), bool)
            for w in np.flatnonzero(on):
                m[wp[w]] = True
            L.append(float((g & ~m).sum() / g.sum()))
            A.append(float((g & ~m).any()))
            K.append(int(m.sum()))
        dep = np.array(r["deployed"], bool)
        out.append(dict(L=L + [0.0], A=A + [0.0], K=K + [len(P)], dep=[float((g & ~dep).sum() / g.sum()),
                                                                    float((g & ~dep).any()), int(dep.sum())]))
    return out


# ---------------------------------------------------------------- report

def crc(L, alpha, idx):
    """Conformal risk control: the first chain setting whose calibrated risk on idx is <= alpha (last = encrypt all)."""
    n = len(idx)
    risk = L[idx].mean(0)
    ok = np.flatnonzero(n / (n + 1) * risk + 1 / (n + 1) <= alpha)
    return int(ok[0]) if len(ok) else L.shape[1] - 1


def summarize(tag, losses, chain, costfn, deployed_j=None, splits=200, seed=0):
    L = np.array([x["L"] for x in losses])
    A = np.array([x["A"] for x in losses])
    C = np.array([[costfn(k) for k in x["K"]] for x in losses])
    K = np.array([x["K"] for x in losses])
    lab = [c[0] if isinstance(c[0], str) else str(c) for c in chain] + ["all"]
    res = dict(n=len(losses), chain=lab, miss=L.mean(0).tolist(), any=A.mean(0).tolist(), cost=C.mean(0).tolist(),
               tokens=K.mean(0).tolist())
    print(f"[e4] {tag}: {len(losses)} inputs with private content")
    for j, c in enumerate(lab):
        star = " (deployed)" if j == deployed_j else ""
        print(f"[e4] {tag}   {c:18s} missed {L[:, j].mean():.4f}  inputs with a miss {A[:, j].mean():.3f}  "
              f"private {K[:, j].mean():6.1f}  s/input {C[:, j].mean():6.2f}{star}")
    rng = np.random.default_rng(seed)
    res["crc"] = {}
    for name, M in (("miss", L), ("any", A)):
        for alpha in ALPHAS:
            pick, te_risk, te_cost = [], [], []
            for _ in range(splits):
                perm = rng.permutation(len(M))
                cal, te = perm[:len(M) // 2], perm[len(M) // 2:]
                j = crc(M, alpha, cal)
                pick.append(j)
                te_risk.append(M[te, j].mean())
                te_cost.append(C[te, j].mean())
            mode = int(np.bincount(pick).argmax())
            res["crc"][f"{name}@{alpha}"] = dict(setting=lab[mode], risk=float(np.mean(te_risk)),
                                                 risk_p95=float(np.percentile(te_risk, 95)), cost=float(np.mean(te_cost)))
            print(f"[e4] {tag}   CRC {name:4s} <= {alpha:<4}: setting {lab[mode]:18s} test {name} {np.mean(te_risk):.4f} "
                  f"(95th pct over splits {np.percentile(te_risk, 95):.4f})  s/input {np.mean(te_cost):.2f}")
    return res


def report(a):
    res = {}
    fc = lambda k: cost(k, FACE_COST, FULL_VIT)
    tc = lambda k: cost(k, TEXT_COST, FULL_BERT)
    for name in ("imdb", "wider"):
        f = f"{OUT}/faces_{name}.json"
        try:
            rows = json.load(open(f))
        except (OSError, ValueError):
            print(f"[e4] {f}: missing or still being written, skipped")
            continue
        for det in ("d320", "d1280"):
            for min_px in ((16,) if name == "imdb" else (16, 32)):
                tag = f"faces_{name}_{det}_min{min_px}"
                res[tag] = summarize(tag, face_losses(rows, det, min_px), FACE_CHAIN, fc,
                                     deployed_j=FACE_CHAIN.index((0.7, 0.1, 0)) if det == "d320" else None)
    for name in ("cadec", "drugs"):
        f = f"{OUT}/text_{name}.json"
        if not os.path.exists(f):
            continue
        ls = text_losses(json.load(open(f)))
        d = np.array([x["dep"] for x in ls])
        print(f"[e4] text_{name} deployed detector: missed {d[:, 0].mean():.4f}  inputs with a miss {d[:, 1].mean():.3f}  "
              f"private {d[:, 2].mean():.1f}  s/input {np.mean([tc(k) for k in d[:, 2]]):.2f}")
        res[f"text_{name}"] = summarize(f"text_{name}", ls, TEXT_CHAIN, tc)
        res[f"text_{name}"]["deployed"] = dict(miss=float(d[:, 0].mean()), any=float(d[:, 1].mean()),
                                               tokens=float(d[:, 2].mean()), cost=float(np.mean([tc(k) for k in d[:, 2]])))
    json.dump(res, open(f"{OUT}/e4_report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", choices=("wider", "imdb"))
    ap.add_argument("--text", choices=("cadec", "drugs"))
    ap.add_argument("--n", type=int, default=2000, help="--faces imdb: photos")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.faces:
        rows = build_wider(a) if a.faces == "wider" else build_imdb(a)
        json.dump(rows, open(f"{OUT}/faces_{a.faces}.json", "w"))
        print(f"[e4] faces {a.faces}: {len(rows)} photos written", flush=True)
        os._exit(0)                                              # the streaming dataset's threads crash at exit
    if a.text:
        json.dump(build_text(a.text), open(f"{OUT}/text_{a.text}.json", "w"))
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
