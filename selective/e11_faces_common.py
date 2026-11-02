"""E11 faces: the photos' private patch sets (e11_faces.py) and their two-stream oracles, shared by the twin test and the
GPU runner. SFHE_FACES=ages switches every user to E15's IMDB-WIKI age photos (e15_ages.py): the validation photos are
scored, and each size bucket ends with AGES_TAIL train photos as its held-out tail (calibration / capture pages; run
with --heldout AGES_TAIL / 2); calibration pools and blob pages come from the train photos (their own faces)."""
import glob
import json
import os

import numpy as np

SET = os.environ.get("SFHE_FACES", "imagenette")
if SET == "ages":
    ROOT = os.path.expanduser("~/data/selective-fhe/e15/ages")
    FACES = f"{ROOT}/faces.json"
else:
    ROOT = os.path.expanduser("~/data/selective-fhe/data/imagenette_dl/imagenette2-320/val")
    FACES = os.path.expanduser("~/data/selective-fhe/e11/faces.json")
MIN_PRIVATE = 4             # the plaintext keys must fit 3 key blocks (196 - k <= 192)
AGES_TAIL = 48


def grow(S, k):
    """S grown by its nearest patches to k patches."""
    S = list(S)
    r0, c0 = divmod(S[0], 14)
    for _d, p in sorted((abs(r - r0) + abs(c - c0), r * 14 + c) for r in range(14) for c in range(14)):
        if len(S) >= k:
            break
        if p not in S:
            S.append(p)
    return sorted(S)


def private_sets(k_max, k_min=1):
    """[(file, label, private patches)] of the photos with a face and k_min..k_max private patches (sets below
    MIN_PRIVATE grown by their nearest patches: encrypting more is always allowed). Ages: the validation photos, then the bucket's AGES_TAIL train photos."""
    j = json.load(open(FACES))
    if SET == "ages":
        ages = json.load(open(f"{ROOT}/ages.json"))
        pick = lambda rows: [(r[0], r[1], j["detections"][r[0]]["patches"]) for r in sorted(rows)
                             if k_min <= len(j["detections"][r[0]]["patches"]) <= k_max]
        tail = pick(ages["train"])[:AGES_TAIL]
        return [(os.path.join(ROOT, rel), lab, S if len(S) >= MIN_PRIVATE else grow(S, MIN_PRIVATE))
                for rel, lab, S in pick(ages["validation"]) + tail]
    wnid2idx = {v[0]: int(k) for k, v in json.load(open(os.path.expanduser(
        "~/data/selective-fhe/data/imagenet_class_index.json"))).items()}
    out = []
    for rel, d in sorted(j["detections"].items()):
        S = list(d["patches"])
        if not (k_min <= len(S) <= k_max):
            continue
        if len(S) < MIN_PRIVATE:
            S = grow(S, MIN_PRIVATE)
        out.append((os.path.join(ROOT, rel), wnid2idx[rel.split("/")[0]], S))
    return out


def inputs(V, store, sets, layers=12, full=True):
    """Per photo: the client's pixel rows of [CLS; private patches], their token positions, the oracle (P_kv, E_hidden,
    logits) and the full model's logits."""
    import e3_reference as R
    out = []
    for f, label, S in sets:
        rows = V.patch_rows(V.pixels(f))
        x = V.embed(store, rows)
        ref = R.two_stream(store, x, S, layers)
        out.append(dict(file=f, label=label, S=S, rows_e=rows[ref["E_idx"]], pos=ref["E_idx"], ref=ref,
                        full=V.logits(store, V.exact_blocks(store, x, layers)[-1])[0] if (full and layers == 12) else None))
    return out


def calib_photos():
    """Photos never evaluated, for calibration pools: Imagenette's validation photos without a face; ages: the train photos."""
    if SET == "ages":
        return sorted(glob.glob(f"{ROOT}/train/*.jpg"))
    faces = {os.path.join(ROOT, r) for r in json.load(open(FACES))["detections"]}
    return sorted(set(glob.glob(f"{ROOT}/*/*.JPEG")) - faces)


def blob_sets(n, k_min, k_max, seed, label=-1):
    """n photos WITHOUT a face (never evaluated), each with a random face-like private region of k_min..k_max patches: a
    connected blob grown from a random patch (4-neighbours, random order). For calibration / magnitude captures.
    Ages: n train photos with their own face of k_min..k_max patches (outside the buckets' held-out tails)."""
    rng = np.random.default_rng(seed)
    if SET == "ages":
        det = json.load(open(FACES))["detections"]
        tail = {f for f, _lab, _S in private_sets(k_max, k_min)[-AGES_TAIL:]}
        fs = [f for f in calib_photos() if f not in tail
              and k_min <= len(det[os.path.relpath(f, ROOT)]["patches"]) <= k_max]
        out = []
        for f in rng.choice(fs, size=n, replace=False):
            S = det[os.path.relpath(f, ROOT)]["patches"]
            out.append((str(f), label, S if len(S) >= MIN_PRIVATE else grow(S, MIN_PRIVATE)))
        return out
    out = []
    for f in rng.choice(calib_photos(), size=n, replace=False):
        k = int(rng.integers(max(k_min, MIN_PRIVATE), k_max + 1))
        S = {int(rng.integers(0, 196))}
        while len(S) < k:
            p = sorted(S)[int(rng.integers(0, len(S)))]
            r, c = divmod(p, 14)
            nb = [(r + dr) * 14 + (c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                  if 0 <= r + dr < 14 and 0 <= c + dc < 14 and (r + dr) * 14 + (c + dc) not in S]
            if nb:
                S.add(nb[int(rng.integers(0, len(nb)))])
        out.append((str(f), label, sorted(S)))
    return out
