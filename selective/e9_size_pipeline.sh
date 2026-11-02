#!/bin/bash
# E9: two-stream ViT (route B, one image per page) for another private-box size: calibrate the per-image PCMM linears'
# band shifts on 3 images, capture the 12-layer circuit with it, plan, count. Same steps as e3_routeb_pipeline.sh.
#   bash selective/e9_size_pipeline.sh BOX     -> bootstrap_placements/selective_e3/vit_sel_box${BOX}_pcmm
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
# CPU-only twin work: a few BLAS threads per process (several pipelines share the machine with the GPU runners)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
BOX=${1:?box side}
NAME=vit_sel_box${BOX}_pcmm
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e9
mkdir -p $D/logs /home/lzhou/data/selective-fhe/e3/logs
cd $REPO
export SEL_ROUTE=pcmm
echo "=== calibrate box $BOX ==="
for i in 0 1 2; do
    PF_PCMM_CALIB=$D/calib_box${BOX}_$i.json $PY $S/e3_test_xpt.py --layers 12 --box $BOX --image $i > $D/logs/calib_box${BOX}_$i.log 2>&1 &
done
wait
grep -hE '\[xpt\] (logits|done)' $D/logs/calib_box${BOX}_*.log
$PY - $BOX <<'PY'
import json, glob, os, sys
box = sys.argv[1]
D = os.path.expanduser("~/data/selective-fhe/e9")
base = json.load(open("bootstrap_placements/vit_b16_224/pcmm_calib.json"))
mx = {}
for f in glob.glob(f"{D}/calib_box{box}_*.json"):
    for k, v in json.load(open(f)).items():
        if k.endswith(".xqk") or k.endswith(".xpv"):
            mx[k] = max(mx.get(k, 0.0), v)
import math
bad = {k: v for k, v in ({**base, **mx}).items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump({**base, **mx}, open(f"{D}/pcmm_calib_box{box}.json", "w"), indent=1, sort_keys=True)
print("merged calibration:", len(base) + len(mx), "linears;", len(mx), "per-image")
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_box$BOX.json
bash $S/e3_pipeline.sh $NAME --box $BOX
cp $D/pcmm_calib_box$BOX.json bootstrap_placements/selective_e3/$NAME/pcmm_calib.json
echo "=== box $BOX pipeline done ==="
