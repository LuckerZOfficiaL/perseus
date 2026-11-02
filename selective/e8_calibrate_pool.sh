#!/bin/bash
# E8: the per-page PCMM linears' band calibration over a pool of SST-2 TRAIN pages (12 per page, both chunks), the max of
# every linear's recorded magnitude over the pool and the plan's calibration -> one calibration file per weight set.
# (The plans were calibrated on 3 pages only; two validation pages exceeded b9.xpv's band by 1.22x and broke on the device.)
#   bash selective/e8_calibrate_pool.sh NAME PAGES [WEIGHTS_EXPORT]   -> ~/data/selective-fhe/e8/pcmm_calib_pool_NAME.json
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
NAME=${1:?name}; NP=${2:?pages}; W=${3:-}
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe/e8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p $D/calib_pool_$NAME
cd $REPO
WA=""; [ -n "$W" ] && WA="--weights $W"
seq 0 $((NP - 1)) | xargs -P 8 -I{} bash -c "a=\$(({} * 12)); PF_PCMM_CALIB=$D/calib_pool_$NAME/page_{}.json \
    $PY selective/e8_test_bert.py --idx \$(seq -s, \$a \$((a + 11))) --chunk0 6 --layers 12 --split train $WA \
    > $D/calib_pool_$NAME/page_{}.log 2>&1"
$PY - $NAME <<'PY'
import glob, json, os, sys
name = sys.argv[1]
D = os.path.expanduser("~/data/selective-fhe/e8")
base = json.load(open("bootstrap_placements/selective_e8/bert_sel_g12/pcmm_calib.json"))
mx, n = dict(base), 0
for f in glob.glob(f"{D}/calib_pool_{name}/page_*.json"):
    n += 1
    for k, v in json.load(open(f)).items():
        mx[k] = max(mx.get(k, 0.0), v)
json.dump(mx, open(f"{D}/pcmm_calib_pool_{name}.json", "w"), indent=1, sort_keys=True)
up = sorted(((mx[k] / base[k], k) for k in base if base[k] > 0), reverse=True)[:6]
print(f"{n} pool pages; largest increases over the plan's calibration:", [(k, round(r, 2)) for r, k in up])
PY
