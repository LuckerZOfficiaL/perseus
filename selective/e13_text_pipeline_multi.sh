#!/bin/bash
# E13 text, robust plan: e8_pipeline.sh with the PCMM bands and the planner's magnitudes taken from MANY pages: the 3
# given train pages plus NX more train pages of the detector's real private sets (e13_pages.py). The captures are
# max-merged per ciphertext name with upstream's scripts/utils/merge_mags.py (a plan sizes every refresh from its
# capture's magnitudes: one page under-sized a refresh for one HateXplain val post on the device, E13), then planned with
# upstream's BERT recipe. Same environment as e8_pipeline.sh (BERT_CONFIGS, SFHE_TEXT_DATA, SFHE_BERT_T, XPT_MB, ...).
#   [CALIB_ALL=1] [NX=10] [PAR=4] EXTRA="--weights W" bash selective/e13_text_pipeline_multi.sh G CHUNK0 N_E_CAP NAME \
#       PRIVATE.json "set 1" "set 2" "set 3"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
G=${1:?}; C0=${2:?}; CAP=${3:?}; NAME=${4:?}; PRIV=${5:?}; shift 5
SETS=("$@")
NX=${NX:-10}; PAR=${PAR:-4}
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e8_$(basename $REPO)
mkdir -p $D/logs $D/graphs
cd $REPO
ARGS="--chunk0 $C0 --n-e-cap $CAP --split train --private $PRIV --layers 12 ${EXTRA:-}"
mapfile -t XP < <($PY $S/e13_pages.py --private $PRIV --G $G --chunk0 $C0 --n-e-cap $CAP --mb ${XPT_MB:-1} --n $NX --seed 1 \
    ${E13_FILL:+--fill $E13_FILL})                      # E13_FILL=large: extra pages of nearly full lane groups
[ ${#XP[@]} -eq $NX ] || { echo "page generation failed"; exit 1; }
ALL=("${SETS[@]}" "${XP[@]}")
echo "=== calibrate $NAME: ${#ALL[@]} train pages ($G per page) ==="
rm -f $D/calib_${NAME}_*.json
for k in "${!ALL[@]}"; do echo "$k ${ALL[$k]}"; done | xargs -P $PAR -L 1 bash -c '
    PF_PCMM_CALIB='$D'/calib_'$NAME'_$0.json '$PY' '$S'/e8_test_bert.py --idx $1 '"$ARGS"' > '$D'/logs/calib_'$NAME'_$0.log 2>&1 \
        || echo "calibration page $0 failed"'
grep -hE '\[e8\] done' $D/logs/calib_${NAME}_*.log | sort -k6 | tail -3
$PY - $NAME $D ${CALIB_ALL:-0} <<'PY'
import json, glob, math, sys
name, D, every = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
base = json.load(open("bootstrap_placements/bert_sst2_128/pcmm_calib.json"))
mx, fs = {}, glob.glob(f"{D}/calib_{name}_*.json")
for f in fs:
    for k, v in json.load(open(f)).items():
        if every or k.endswith(".xqk") or k.endswith(".xpv"):
            mx[k] = max(mx.get(k, 0.0), base.get(k, 0.0), v)
bad = {k: v for k, v in ({**base, **mx}).items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump({**base, **mx}, open(f"{D}/pcmm_calib_{name}.json", "w"), indent=1, sort_keys=True)
print(f"merged calibration over {len(fs)} pages: {len({**base, **mx})} linears")
PY
[ $? -eq 0 ] || exit 1
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
echo "=== capture $NAME: ${#ALL[@]} pages ==="
rm -rf $D/graphs/$NAME $D/graphs/${NAME}_x* $D/graphs/${NAME}_merged bootstrap_placements/selective_e8/$NAME
for k in "${!ALL[@]}"; do echo "$k ${ALL[$k]}"; done | xargs -P $PAR -L 1 bash -c '
    if [ $0 = 0 ]; then G='$D'/graphs/'$NAME'; else G='$D'/graphs/'$NAME'_x$0; fi
    '$PY' '$S'/e8_test_bert.py --idx $1 '"$ARGS"' --capture $G > '$D'/logs/capture_'$NAME'_$0.log 2>&1 || echo "capture $0 failed"'
$PY scripts/utils/merge_mags.py $D/graphs/$NAME $D/graphs/${NAME}_merged $(for k in $(seq 1 $((${#ALL[@]} - 1))); do echo $D/graphs/${NAME}_x$k; done) | tail -2
echo "=== plan $NAME (merged magnitudes) ==="
UP=bootstrap_placements/bert_sst2_128
export PYTHON=$PY SFHE_PATCHES=1 PYTHONPATH="$REPO/selective/sfhe_site:$REPO/selective${PYTHONPATH:+:$PYTHONPATH}"
RECIPE=$($PY -m sfhe.runtime recipe $UP)
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh $D/graphs/${NAME}_merged selective_e8/$NAME > $D/logs/plan_$NAME.log 2>&1 \
    || { echo "plan failed"; tail -5 $D/logs/plan_$NAME.log; exit 1; }
{ echo "graph=$D/graphs/${NAME}_merged (max-merged over ${#ALL[@]} train pages)"
  echo "tool=python/perseus/examples/gpt2/make_plan.sh (blocks), selective BERT (selective/sfhe), e13_text_pipeline_multi.sh"
  echo "runtime=$($PY -m sfhe.runtime runtime $UP)"
  if [ -n "${BERT_CONFIGS:-}" ]; then echo "configs=$BERT_CONFIGS"; else grep -E '^configs=' $UP/PLAN_CMD.txt; fi
  echo "recipe=$RECIPE"; echo "upstream=$(git -C $REPO rev-parse --short HEAD)"
  echo "text_data=${SFHE_TEXT_DATA:-sst2} private=$PRIV extra=${EXTRA:-} SFHE_BERT_T=${SFHE_BERT_T:-128} XPT_MB=${XPT_MB:-1}"
} > bootstrap_placements/selective_e8/$NAME/PLAN_CMD.txt
cp $D/pcmm_calib_$NAME.json bootstrap_placements/selective_e8/$NAME/pcmm_calib.json
bts=$($PY -c "import json,glob; print(sum(json.load(open(f))['summary']['total_bootstraps'] for f in glob.glob('bootstrap_placements/selective_e8/$NAME/block_*_placement.json')))")
echo "planned bootstraps $bts"
echo "=== done $NAME ==="
