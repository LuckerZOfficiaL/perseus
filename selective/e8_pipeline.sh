#!/bin/bash
# E8: two-stream BERT, G sentences per encrypted page (route B, page layouts). Calibrate the per-page PCMM linears' band
# shifts on the fake (3 pages of other sentences), merge with the shipped BERT calibration, capture the 12-layer circuit,
# plan it with the shipped BERT recipe, count.
#   [EXTRA="--chunk0 6"] [BERT_CONFIGS=...] bash selective/e8_pipeline.sh G NAME "idx set 1" "idx set 2" "idx set 3"   (set 1 is also the capture page)
# CALIB_ALL=1 (other weights than the shipped SST-2 model's): every linear's band from max(shipped, these pages), not only
# the per-page linears'.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
# CPU-only twin work: a few BLAS threads per process (several pipelines share the machine with the GPU runners)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
G=${1:?inputs per page}; NAME=${2:?plan name}; shift 2
SETS=("$@")
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e8_$(basename $REPO)
mkdir -p $D/logs $D/graphs
cd $REPO
echo "=== calibrate $NAME ($G per page) ==="
for i in 0 1 2; do
    PF_PCMM_CALIB=$D/calib_${NAME}_$i.json $PY $S/e8_test_bert.py --idx ${SETS[$i]} --layers 12 ${EXTRA:-} > $D/logs/calib_${NAME}_$i.log 2>&1 &
done
wait
grep -hE '\[e8\] (sentence|done)' $D/logs/calib_${NAME}_*.log
$PY - $NAME $D ${CALIB_ALL:-0} <<'PY'
import json, glob, os, sys
name, D, every = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
base = json.load(open("bootstrap_placements/bert_sst2_128/pcmm_calib.json"))
mx = {}
for f in glob.glob(f"{D}/calib_{name}_*.json"):
    for k, v in json.load(open(f)).items():
        if every or k.endswith(".xqk") or k.endswith(".xpv"):
            mx[k] = max(mx.get(k, 0.0), base.get(k, 0.0), v)
import math
bad = {k: v for k, v in ({**base, **mx}).items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump({**base, **mx}, open(f"{D}/pcmm_calib_{name}.json", "w"), indent=1, sort_keys=True)
grown = sorted(((v / base[k], k) for k, v in mx.items() if k in base and v > base[k]), reverse=True)
print("merged calibration:", len({**base, **mx}), "linears;", len(mx), "measured;", len(grown), "above the shipped band"
      + (": " + ", ".join(f"{k} x{r:.2f}" for r, k in grown[:4]) if grown else ""))
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e8/$NAME
echo "=== capture $NAME ==="
$PY $S/e8_test_bert.py --idx ${SETS[0]} --layers 12 ${EXTRA:-} --capture $D/graphs/$NAME > $D/logs/capture_$NAME.log 2>&1 \
    || { echo "capture failed"; tail -5 $D/logs/capture_$NAME.log; exit 1; }
grep -E '\[e8\] (sentence|done)' $D/logs/capture_$NAME.log
echo "=== plan $NAME ==="
UP=bootstrap_placements/bert_sst2_128
export PYTHON=$PY SFHE_PATCHES=1 PYTHONPATH="$REPO/selective/sfhe_site:$REPO/selective${PYTHONPATH:+:$PYTHONPATH}"
RECIPE=$($PY -m sfhe.runtime recipe $UP)                 # upstream's current BERT recipe
echo "recipe: $RECIPE"
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh $D/graphs/$NAME selective_e8/$NAME > $D/logs/plan_$NAME.log 2>&1 \
    || { echo "plan failed"; tail -5 $D/logs/plan_$NAME.log; exit 1; }
{ echo "graph=$D/graphs/$NAME"; echo "tool=python/perseus/examples/gpt2/make_plan.sh (blocks), selective BERT (selective/sfhe)";
  echo "runtime=$($PY -m sfhe.runtime runtime $UP)"
  if [ -n "${BERT_CONFIGS:-}" ]; then echo "configs=$BERT_CONFIGS"; else grep -E '^configs=' $UP/PLAN_CMD.txt; fi
  echo "recipe=$RECIPE"; echo "upstream=$(git -C $REPO rev-parse --short HEAD)"; } > bootstrap_placements/selective_e8/$NAME/PLAN_CMD.txt
cp $D/pcmm_calib_$NAME.json bootstrap_placements/selective_e8/$NAME/pcmm_calib.json
echo "=== counts $NAME ==="
$PY - "$D/graphs/$NAME" "bootstrap_placements/selective_e8/$NAME" <<'PY'
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
