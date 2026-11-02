#!/bin/bash
# E11 faces: one size bucket's plan. Calibrate the per-page PCMM linears' bands on 3 pages of HELD-OUT face photos (the
# end of the bucket; the GPU runs score the others), capture with the first photo grown to the bucket's cap (every own-key
# offset live), plan with the ViT recipe, count. XPT_PAD=1 throughout.
#   bash selective/e11_faces_pipeline.sh NAME K_MIN K_MAX CAP G CHUNK0 HELDOUT
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
NAME=${1:?}; KMIN=${2:?}; KMAX=${3:?}; CAP=${4:?}; G=${5:?}; C0=${6:?}; HO=${7:?}
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e11
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 SEL_ROUTE=pcmm XPT_PAD=${XPT_PAD:-2}
mkdir -p $D/logs $D/graphs
cd $REPO
ARGS="--k-min $KMIN --k-max $KMAX --cap $CAP --chunk0 $C0 --n $G --tail --layers 12"
echo "=== calibrate $NAME (held-out photos: the last $HO of the bucket) ==="
for i in 0 1 2; do
    PF_PCMM_CALIB=$D/calib_${NAME}_$i.json $PY $S/e11_test_faces.py $ARGS --first $((i * G % HO)) > $D/logs/calib_${NAME}_$i.log 2>&1 &
done
wait
grep -hE '\[e11\] (photo|done)' $D/logs/calib_${NAME}_*.log | tail -4
$PY - $NAME <<'PY'
import json, glob, os, sys
name = sys.argv[1]
D = os.path.expanduser("~/data/selective-fhe/e11")
base = json.load(open("bootstrap_placements/vit_b16_224/pcmm_calib.json"))
mx = dict(base)
for f in glob.glob(f"{D}/calib_{name}_*.json"):
    for k, v in json.load(open(f)).items():
        mx[k] = max(mx.get(k, 0.0), v)
import math
bad = {k: v for k, v in (mx).items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump(mx, open(f"{D}/pcmm_calib_{name}.json", "w"), indent=1, sort_keys=True)
print("merged calibration:", len(mx), "linears")
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e3/$NAME
echo "=== capture $NAME ==="
$PY $S/e11_test_faces.py $ARGS --first 0 --grow-first --capture $D/graphs/$NAME > $D/logs/capture_$NAME.log 2>&1 \
    || { echo "capture failed"; tail -5 $D/logs/capture_$NAME.log; exit 1; }
grep -E '\[e11\] (photo|done)' $D/logs/capture_$NAME.log
echo "=== plan $NAME ==="
bash $S/e3_plan.sh $D/graphs/$NAME $NAME | tail -2
cp $D/pcmm_calib_$NAME.json bootstrap_placements/selective_e3/$NAME/pcmm_calib.json
$PY - "bootstrap_placements/selective_e3/$NAME" <<'PY'
import json, sys, glob
p = sys.argv[1]
print("planned bootstraps", sum(json.load(open(f))["summary"]["total_bootstraps"] for f in glob.glob(f"{p}/block_*_placement.json")))
PY
echo "=== done $NAME ==="
