#!/usr/bin/env python
"""E7: the sparse per-page linear encoder (dense_xpt.plain_linears_sparse) is bit-identical to the dense path
(plain_linears / plain_linears_batch), for 1 and 3 inputs, automatic and fixed weight scales; and its speed."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_harness as H


def same(a, b):
    return (a.n_in, a.n_out, a.w_scale, a.shift, a.name, a.t) == (b.n_in, b.n_out, b.w_scale, b.shift, b.name, b.t) and \
        np.array_equal(np.asarray(a.Wd), np.asarray(b.Wd))


def main():
    from sfhe import dense_xpt as DX
    h = H.setup(layers=1, structure_only=True)
    B = h.model.blocks
    rng = np.random.default_rng(0)
    ok = True
    for n_in in (1, 3):
        for ws in ((None, None), (8192.0, 4096.0)):
            K = [rng.normal(0, 1, (180, 768)) for _ in range(n_in)]
            Vv = [rng.normal(0, 1, (180, 768)) for _ in range(n_in)]
            t0 = time.time()
            if n_in == 1:
                d = DX.plain_linears(K[0], Vv[0], B, 5, ws_qk=ws[0], ws_pv=ws[1], shift_qk=3, shift_pv=2)
            else:
                d = DX.plain_linears_batch(K, Vv, B, 5, ws_qk=ws[0], ws_pv=ws[1], shift_qk=3, shift_pv=2)
            td = time.time() - t0
            t0 = time.time()
            sp = DX.plain_linears_sparse(K, Vv, B, 5, ws_qk=ws[0], ws_pv=ws[1], shift_qk=3, shift_pv=2)
            ts = time.time() - t0
            eq = same(d[0], sp[0]) and same(d[1], sp[1])
            ok &= eq
            print(f"inputs={n_in} ws={ws}: identical={eq}  dense {td:.2f}s  sparse {ts:.2f}s (one layer)")
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
