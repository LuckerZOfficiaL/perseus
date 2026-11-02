#!/bin/bash
# E14: the long-document baseline, Perseus's own full BERT (every token encrypted; upstream's model, runtime and recipe,
# untouched) at T tokens on another export / dataset: the PCMM calibration of every linear on a few texts at T (max with
# the shipped one), a capture of one text with T real tokens on the fake (upstream run_bert --fake), a plan with
# upstream's current BERT recipe. The approximation config CONFIGS comes from the --full refits (e8_ln_bands.py /
# e12_bert_sm_ranges.py --tokens T --full). sfhe is imported only for its data patches (SFHE_TEXT_DATA, BERT_EXPORT).
#   [RESUME_PLAN=1] bash selective/e14_full_pipeline.sh NAME T EXPORT_DIR DATA.json CONFIGS.json "calibration idx" capture_idx
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -eu
NAME=${1:?name}; T=${2:?tokens}; W=${3:?export dir}; DATA=${4:?texts}; CF=${5:?configs}; CAL=${6:?calibration idx}
CAP=${7:?capture idx}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SFHE_TEXT_DATA=$DATA BERT_EXPORT=$W SFHE_PATCHES=1
export PYTHONPATH="$REPO/selective/sfhe_site:$REPO/selective:$REPO/python"
PY=/home/lzhou/data/venvs/perseus/bin/python
export PYTHON=$PY                                          # make_plan.sh's interpreter
UP=bootstrap_placements/bert_sst2_128
D=/home/lzhou/data/selective-fhe/e14_full
mkdir -p $D/logs $D/graphs
cd $REPO
SETS=$(grep '^runtime=' $UP/PLAN_CMD.txt | cut -d= -f2- | tr ' ' '\n' | grep = | sed 's/^/--set /' | tr '\n' ' ')
RUN="$PY -m perseus.examples.bert.run_bert --fake --text '' --tokens $T --configs $CF $SETS"
if [ "${RESUME_PLAN:-0}" = 1 ]; then export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json; else
echo "=== calibrate $NAME (texts $CAL) ==="
eval PF_PCMM_CALIB=$D/calib_$NAME.json PF_PCMM_CALIB_IN=$UP/pcmm_calib.json $RUN --sst2 $CAL > $D/logs/calib_$NAME.log 2>&1 \
    || { echo "calibration failed"; tail -5 $D/logs/calib_$NAME.log; exit 1; }
grep -E "SUMMARY|AGREE" $D/logs/calib_$NAME.log | cut -c1-200
$PY - $UP/pcmm_calib.json $D/calib_$NAME.json $D/pcmm_calib_$NAME.json <<'PY'
import json, math, sys
base, meas = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = {k: max(base.get(k, 0.0), meas.get(k, 0.0)) for k in set(base) | set(meas)}
bad = {k: v for k, v in out.items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged run?): {bad}")
json.dump(out, open(sys.argv[3], "w"), indent=1, sort_keys=True)
grown = sorted(((meas[k] / base[k], k) for k in meas if k in base and meas[k] > base[k]), reverse=True)
print(f"merged calibration: {len(out)} linears, {len(grown)} above the shipped band"
      + (": " + ", ".join(f"{k} x{r:.2f}" for r, k in grown[:4]) if grown else ""))
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e14/$NAME
echo "=== capture $NAME (text $CAP) ==="
eval $RUN --sst2 $CAP --capture $D/graphs/$NAME > $D/logs/capture_$NAME.log 2>&1 \
    || { echo "capture failed"; tail -5 $D/logs/capture_$NAME.log; exit 1; }
grep -E "SUMMARY" $D/logs/capture_$NAME.log | cut -c1-200
fi                                                         # RESUME_PLAN=1: calibration and capture already there
echo "=== plan $NAME ==="
RECIPE=$($PY -m sfhe.runtime recipe $UP)
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh $D/graphs/$NAME selective_e14/$NAME > $D/logs/plan_$NAME.log 2>&1 \
    || { echo "plan failed"; tail -5 $D/logs/plan_$NAME.log; exit 1; }
P=bootstrap_placements/selective_e14/$NAME
{ echo "graph=$D/graphs/$NAME"; echo "tool=python/perseus/examples/gpt2/make_plan.sh (blocks), upstream full BERT at T=$T"
  grep -E '^runtime=' $UP/PLAN_CMD.txt; echo "configs=$CF"; echo "recipe=$RECIPE"
  echo "upstream=$(git -C $REPO rev-parse --short HEAD)"; echo "text_data=$DATA weights=$W tokens=$T"; } > $P/PLAN_CMD.txt
cp $D/pcmm_calib_$NAME.json $P/pcmm_calib.json
bts=$($PY -c "import json,glob; print(sum(json.load(open(f))['summary']['total_bootstraps'] for f in glob.glob('$P/block_*_placement.json')))")
echo "planned bootstraps $bts"
echo "=== done $NAME ==="
