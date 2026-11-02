#!/usr/bin/env python
"""Extra calibration / capture pages for a multi-page text plan (e13_text_pipeline_multi.sh): NX pages of G TRAIN texts
with the detector's real private sets, packed as the runners pack them (`--chunk0` per chunk, at most 64 lanes per
chunk, each input CLS + private + SEP <= n_e_cap lanes, 1..MB x 64 plaintext keys). Seeded; one page per output line
(comma-separated train indices). Texts come from SFHE_TEXT_DATA when set (else SST-2 train), at SFHE_BERT_T tokens.

  SFHE_TEXT_DATA=... python selective/e13_pages.py --private P.json --G 24 --chunk0 12 --n-e-cap 15 --mb 1 --n 10
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", required=True)
    ap.add_argument("--G", type=int, required=True)
    ap.add_argument("--chunk0", type=int, required=True)
    ap.add_argument("--n-e-cap", type=int, required=True)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pool", type=int, default=4000, help="train texts to draw from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill", default="small", choices=("small", "large"),
                    help="after each chunk's largest private set, fill with the smallest (default) or the largest sets "
                         "(pages of nearly full lane groups, for plans whose inputs fill them)")
    a = ap.parse_args()
    import sfhe  # noqa: F401  (SFHE_TEXT_DATA)
    from perseus.examples.bert import data as D
    T = int(os.environ.get("SFHE_BERT_T", "128"))
    priv = json.load(open(os.path.expanduser(a.private)))["train"]
    texts = D.sentences(a.pool, "train")
    cand = []
    for i, (s, _lab) in enumerate(texts):
        S = priv.get(str(i)) or []
        if not S:
            continue
        _ids, n = D.encode(s, T)
        S = [p for p in S if 0 < p < n - 1]
        ne, npk = len(S) + 2, n - 2 - len(S)
        if S and ne <= a.n_e_cap and 0 < npk <= a.mb * 64:
            cand.append((i, ne))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(cand)
    sizes = {i: ne for i, ne in cand}
    order = [i for i, _ in cand]
    big = sorted(order, key=lambda i: -sizes[i])[:a.n * 2]     # each page gets one of the largest private sets per chunk
    used, pages = set(), []
    for p in range(a.n):
        page, lanes = [], [0, 0]
        for ch, quota in ((0, a.chunk0), (1, a.G - a.chunk0)):
            k = 0
            fill = sorted(order, key=lambda i: ((sizes[i] if a.fill == "small" else -sizes[i]), rng.random()))
            for i in [big[2 * p + ch]] + fill:
                if k == quota:
                    break
                if i in used or lanes[ch] + sizes[i] > 64 - (quota - k - 1):   # leave a lane per remaining slot
                    continue
                page.append(i); used.add(i); lanes[ch] += sizes[i]; k += 1
        if len(page) == a.G:
            pages.append(page)
    if len(pages) < a.n:
        sys.exit(f"only {len(pages)} pages of {a.G} could be packed from {len(cand)} candidates")
    for page in pages:
        print(",".join(map(str, page)))


if __name__ == "__main__":
    main()
