#!/usr/bin/env python
"""E25 task 2: what the private SET itself reveals. OWTE's server sees, besides the public tokens, which positions are
private: the number of private pieces / patches, span lengths, positions, the page layout bucket and whether the input
falls back to full encryption. An attacker that sees ONLY these (no public content) predicts the drug-review condition,
the reviewed drug (30 most frequent, as E16) and the age group. Models: gradient boosting and logistic regression on hand
features, plus (photos) gradient boosting on the raw 14 x 14 mask and a small CNN on it; train split -> validation, best
model per target reported with the majority rate. Then padding: text spans extended to the next length in {1, 2, 4, 8,
16, 32} pieces (encrypting the following, else preceding, word pieces), and also with the count raised to its page layout's
cap (13 or 29 pieces, so no extra cost); faces snapped to the smallest box of a fixed set
(4x4, 5x5, 6x6, 7x7, 7x9 patches; larger -> the whole photo) that contains the face. Attacker re-run on the padded sets;
extra encrypted pieces / patches and the per-input cost from the measured per-layout costs (DERIVED, e16 tables).

  python selective/e25_s_attack.py --export          (sat venv: tokenizer, datasets)
  python selective/e25_s_attack.py --attack          (p02 venv: sklearn, torch CPU)
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/data/selective-fhe/e25")
D = os.path.expanduser("~/data/selective-fhe")
TEXT_COST = ((0, 0.0), (13, 1.78), (29, 5.51))                  # = e16_leakage (kept here: this venv lacks e16's deps)
FACE_COST = ((0, 0.0), (20, 2.51), (31, 5.54), (63, 10.76))
FULL_BERT, FULL_VIT = 14.40, 36.57
SPAN_BUCKETS = (1, 2, 4, 8, 16, 32)
BOXES = ((4, 4), (5, 5), (6, 6), (7, 7), (7, 9), (9, 7))


def cost(k, table, full):
    for kmax, s in table:
        if k <= kmax:
            return s
    return full


def export():
    from transformers import AutoTokenizer
    import e16_leakage as E16
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    data = json.load(open(f"{D}/e15/drugs_t128.json"))
    priv = json.load(open(f"{D}/e15/drugs_t128c_private.json"))
    out = {}
    for split in ("train", "validation"):
        drug = E16.drug_names_of(split)
        rows = []
        for i, (t, y) in enumerate(data[split]):
            P = priv[split].get(str(i), [])
            if not P:
                continue
            n = len(tok(t, truncation=True, max_length=128)["input_ids"])
            rows.append(dict(i=i, y=int(y), drug=drug.get(t[:200]), n=n, P=sorted(p for p in P if p < n)))
        out[split] = rows
        print(f"[e25] export {split}: {len(rows)} reviews with private pieces", flush=True)
    json.dump(out, open(f"{OUT}/s_only_drugs.json", "w"))


# ---------------------------------------------------------------- features

def spans(P):
    out, cur = [], []
    for p in sorted(P):
        if cur and p == cur[-1] + 1:
            cur.append(p)
        else:
            if cur:
                out.append(cur)
            cur = [p]
    if cur:
        out.append(cur)
    return out


def text_feats(P, n):
    sp = spans(P)
    L = sorted((len(s) for s in sp), reverse=True)[:6]
    L += [0] * (6 - len(L))
    first = [s[0] / n for s in sp][:4]
    first += [-1] * (4 - len(first))
    hist = np.histogram([p / n for p in P], bins=8, range=(0, 1))[0].tolist()
    k = len(P)
    layout = 0 if k <= 13 else 1 if k <= 29 else 2
    return [k, len(sp), n, layout, int(layout == 2)] + L + first + hist


def pad_text(P, n):
    """Each span extended to the next bucket length (following pieces, else preceding), inside [1, n - 2]."""
    S = set(P)
    for s in spans(P):
        want = next((b for b in SPAN_BUCKETS if b >= len(s)), len(s))
        lo, hi = s[0], s[-1]
        while hi - lo + 1 < want:
            if hi + 1 <= n - 2:
                hi += 1
            elif lo - 1 >= 1:
                lo -= 1
            else:
                break
        S |= set(range(lo, hi + 1))
    return sorted(S)


def pad_cap(P, n):
    """Span buckets, then more neighbouring pieces until the count reaches its page layout's cap (13 or 29): the count
    no longer varies inside a layout, at no extra cost (the layout, so the page cost, is unchanged)."""
    S = set(pad_text(P, n))
    cap = 13 if len(S) <= 13 else 29 if len(S) <= 29 else None
    if cap is None:
        return sorted(S)
    r = 1
    while len(S) < cap and r < n:
        for p in sorted(S):
            for q in (p + r, p - r):
                if len(S) < cap and 1 <= q <= n - 2:
                    S.add(q)
        r += 1
    return sorted(S)


def mask_of(P):
    m = np.zeros((14, 14), np.float32)
    for p in P:
        m[divmod(int(p), 14)] = 1
    return m


def face_feats(P):
    rr = [int(p) // 14 for p in P]
    cc = [int(p) % 14 for p in P]
    k = len(P)
    layout = 0 if k <= 20 else 1 if k <= 31 else 2 if k <= 63 else 3
    return [k, min(rr), max(rr), min(cc), max(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1, np.mean(rr), np.mean(cc),
            layout, int(layout == 3)]


def pad_face(P):
    """The smallest fixed box (rows x cols) containing the face's bounding box, placed over it and clipped to the photo;
    no box fits -> the whole photo."""
    rr = [int(p) // 14 for p in P]
    cc = [int(p) % 14 for p in P]
    r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
    for h, w in BOXES:
        if r1 - r0 <= h and c1 - c0 <= w:
            R0 = min(max(0, r0 - (h - (r1 - r0)) // 2), 14 - h)
            C0 = min(max(0, c0 - (w - (c1 - c0)) // 2), 14 - w)
            return sorted({r * 14 + c for r in range(R0, R0 + h) for c in range(C0, C0 + w)} | {int(p) for p in P})
    return list(range(196))


# ---------------------------------------------------------------- attackers

def fit_eval(Xtr, ytr, Xva, yva, masks=None, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
    res["gboost"] = float((gb.predict(Xva) == yva).mean())
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), ytr)
    res["logreg"] = float((lr.predict(sc.transform(Xva)) == yva).mean())
    if masks is not None:
        Mtr, Mva = masks
        gm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed).fit(
            np.c_[Mtr.reshape(len(Mtr), -1), Xtr], ytr)
        res["gboost_mask"] = float((gm.predict(np.c_[Mva.reshape(len(Mva), -1), Xva]) == yva).mean())
        res["cnn_mask"] = cnn(Mtr, ytr, Mva, yva, seed)
    return res


def cnn(Mtr, ytr, Mva, yva, seed=0, epochs=15):
    import torch
    torch.manual_seed(seed)
    n_cls = int(max(ytr.max(), yva.max())) + 1
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 32, 3, padding=1),
                              torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(512, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, n_cls))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X, Y = torch.tensor(Mtr)[:, None], torch.tensor(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), 128):
            b = perm[j:j + 128]
            loss = torch.nn.functional.cross_entropy(net(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((net(torch.tensor(Mva)[:, None]).argmax(1).numpy() == yva).mean())


def attack():
    torch_threads()
    res = dict(cost_note="s/input DERIVED from measured per-layout page costs (e16_leakage tables), not new runs", tasks={})
    d = json.load(open(f"{OUT}/s_only_drugs.json"))
    tr, va = d["train"], d["validation"]
    top = [g for g, _c in Counter(r["drug"] for r in tr if r["drug"]).most_common(30)]
    for target in ("condition", "drug"):
        for padded in (False, True, "cap"):
            def rows(split):
                X, y, K = [], [], []
                for r in split:
                    if target == "drug" and r["drug"] not in top:
                        continue
                    P = pad_cap(r["P"], r["n"]) if padded == "cap" else pad_text(r["P"], r["n"]) if padded else r["P"]
                    X.append(text_feats(P, r["n"]))
                    y.append(r["y"] if target == "condition" else top.index(r["drug"]))
                    K.append(len(P))
                return np.array(X, float), np.array(y), np.array(K)
            Xtr, ytr, _ = rows(tr)
            Xva, yva, K = rows(va)
            acc = fit_eval(Xtr, ytr, Xva, yva)
            maj = Counter(yva.tolist()).most_common(1)[0][1] / len(yva)
            key = f"drugs_{target}{'_padded_cap' if padded == 'cap' else '_padded' if padded else ''}"
            res["tasks"][key] = dict(n_train=len(ytr), n_val=len(yva), majority=maj, acc=acc, best=max(acc.values()),
                                     private=float(K.mean()),
                                     s=float(np.mean([cost(k, TEXT_COST, FULL_BERT) for k in K])))
            print(f"[e25] S-only {key:24s} best {max(acc.values()):.3f} (majority {maj:.3f}) {acc}  private {K.mean():5.1f} "
                  f"s/input {res['tasks'][key]['s']:.2f}", flush=True)
    meta = json.load(open(f"{D}/e15/ages/ages.json"))
    det = json.load(open(f"{D}/e15/ages/faces.json"))["detections"]
    for padded in (False, True):
        def rows(split):
            X, M, y, K = [], [], [], []
            for rel, lab, _age in meta[split]:
                P = pad_face(det[rel]["patches"]) if padded else det[rel]["patches"]
                X.append(face_feats(P))
                M.append(mask_of(P))
                y.append(lab)
                K.append(len(P))
            return np.array(X, float), np.stack(M), np.array(y), np.array(K)
        Xtr, Mtr, ytr, _ = rows("train")
        Xva, Mva, yva, K = rows("validation")
        acc = fit_eval(Xtr, ytr, Xva, yva, (Mtr, Mva))
        maj = Counter(yva.tolist()).most_common(1)[0][1] / len(yva)
        key = f"ages_group{'_padded' if padded else ''}"
        res["tasks"][key] = dict(n_train=len(ytr), n_val=len(yva), majority=maj, acc=acc, best=max(acc.values()),
                                 private=float(K.mean()), s=float(np.mean([cost(k, FACE_COST, FULL_VIT) for k in K])),
                                 full_fallback=int((K > 63).sum()))
        print(f"[e25] S-only {key:24s} best {max(acc.values()):.3f} (majority {maj:.3f}) {acc}  private {K.mean():5.1f} "
              f"s/photo {res['tasks'][key]['s']:.2f}  over 63: {(K > 63).sum()}", flush=True)
    res["identity_note"] = ("identity from S alone not attempted: 831 distinct names among the 900 validation photos (E16), "
                            "so a set-only classifier has almost no examples per name")
    json.dump(res, open(f"{OUT}/s_only.json", "w"), indent=1)


def torch_threads():
    try:
        import torch
        torch.set_num_threads(16)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--attack", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.export:
        export()
    if a.attack:
        attack()


if __name__ == "__main__":
    main()
