#!/bin/bash
# E7: several inputs on one encrypted page (route B). Calibrate the batched per-image PCMM linears' band shifts on the
# fake (3 image triplets), merge, capture the 12-layer batched circuit with it, plan, count.
#   [EXTRA="--chunk0 3"] bash selective/e7_pipeline.sh [inputs=3] [name] [box=4]
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
# CPU-only twin work: a few BLAS threads per process (several pipelines share the machine with the GPU runners)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
NB=${1:-3}
BOX=${3:-4}
NAME=${2:-vit_sel_box${BOX}_b$NB}
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e7
mkdir -p $D/logs $D/graphs
cd $REPO
export SEL_ROUTE=pcmm
# NB image indices per calibration page (the 4 shipped samples, repeated as needed)
sets=()
for o in 0 1 3; do sets+=("$(seq -s, $o $((o + NB - 1)) | tr ',' '\n' | awk '{printf "%s%d", (NR>1?",":""), $1 % 4}')"); done
echo "=== calibrate ($NB inputs per page) ==="
for i in 0 1 2; do
    PF_PCMM_CALIB=$D/calib_${NAME}_$i.json $PY $S/e7_test_batch.py --inputs $NB --layers 12 --box $BOX --images ${sets[$i]} ${EXTRA:-} \
        > $D/logs/calib_${NAME}_$i.log 2>&1 &
done
wait
grep -hE '\[e7\] (input|done)' $D/logs/calib_${NAME}_*.log
$PY - $NAME <<'PY'
import json, glob, os, sys
name = sys.argv[1]
D = os.path.expanduser("~/data/selective-fhe/e7")
base = json.load(open("bootstrap_placements/vit_b16_224/pcmm_calib.json"))
mx = {}
for f in glob.glob(f"{D}/calib_{name}_*.json"):
    for k, v in json.load(open(f)).items():
        if k.endswith(".xqk") or k.endswith(".xpv"):
            mx[k] = max(mx.get(k, 0.0), v)
merged = {**base, **mx}
import math
bad = {k: v for k, v in (merged).items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump(merged, open(f"{D}/pcmm_calib_{name}.json", "w"), indent=1, sort_keys=True)
print("merged calibration:", len(merged), "linears;", len(mx), "batched")
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e3/$NAME
echo "=== capture $NAME ==="
$PY $S/e7_test_batch.py --inputs $NB --layers 12 --box $BOX --images ${sets[0]} ${EXTRA:-} --capture $D/graphs/$NAME > $D/logs/capture_$NAME.log 2>&1 \
    || { echo "capture failed"; tail -5 $D/logs/capture_$NAME.log; exit 1; }
grep -E '\[e7\] (input|done)' $D/logs/capture_$NAME.log
echo "=== plan $NAME ==="
bash $S/e3_plan.sh $D/graphs/$NAME $NAME | tail -2
cp $D/pcmm_calib_$NAME.json bootstrap_placements/selective_e3/$NAME/pcmm_calib.json
echo "=== counts $NAME ==="
$PY - "$D/graphs/$NAME" "bootstrap_placements/selective_e3/$NAME" <<'PY'
import json, collections, sys, glob
d, p = sys.argv[1], sys.argv[2]
tot = collections.Counter()
for f in glob.glob(f"{d}/block_*/graph.json"):
    for n in json.load(open(f))["nodes"]:
        tot[n["op_type"]] += 1
bts = sum(json.load(open(f))["summary"]["total_bootstraps"] for f in glob.glob(f"{p}/block_*_placement.json"))
print("nodes", sum(tot.values()), "| mult", tot["mult"], "square", tot["square"], "rotate", tot["rotate"],
      "| planned bootstraps", bts)
PY
echo "=== done $NAME ==="
