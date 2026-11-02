#!/bin/bash
# Plan a captured selective ViT graph with UPSTREAM's current ViT recipe (bootstrap_placements/vit_b16_224 PLAN_CMD
# recipe=) and print its bootstrap counts. The plan's runtime= line is upstream's ViT runtime with sfhe's layout pins
# (sfhe/runtime.py); the planner runs with sfhe's runtime patches (selective/sfhe_site, SFHE_PATCHES=1).
#   bash selective/e3_plan.sh GRAPH_DIR NAME          -> bootstrap_placements/selective_e3/NAME
set -u
G=${1:?graph dir}; NAME=${2:?plan name}
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd $REPO
UP=bootstrap_placements/vit_b16_224
export PYTHON=${PYTHON:-/home/lzhou/data/venvs/perseus/bin/python}
export SFHE_PATCHES=1 PYTHONPATH="$REPO/selective/sfhe_site:$REPO/selective${PYTHONPATH:+:$PYTHONPATH}"
RECIPE=$($PYTHON -m sfhe.runtime recipe $UP)
LOGD=/home/lzhou/data/selective-fhe/logs_$(basename $REPO)
mkdir -p $LOGD
env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh "$G" "selective_e3/$NAME" > $LOGD/plan_$NAME.log 2>&1 \
    || { echo "plan failed: see $LOGD/plan_$NAME.log"; tail -5 $LOGD/plan_$NAME.log; exit 1; }
{ echo "graph=$G"; echo "tool=python/perseus/examples/gpt2/make_plan.sh (blocks), selective ViT (selective/sfhe)";
  echo "runtime=$($PYTHON -m sfhe.runtime runtime $UP)"
  if [ -n "${VIT_CONFIGS:-}" ]; then echo "configs=$VIT_CONFIGS"; else grep -E '^configs=' $UP/PLAN_CMD.txt; fi
  echo "recipe=$RECIPE"; echo "upstream=$(git -C $REPO rev-parse --short HEAD)"; } > bootstrap_placements/selective_e3/$NAME/PLAN_CMD.txt
cp $UP/pcmm_calib.json bootstrap_placements/selective_e3/$NAME/
$PYTHON - bootstrap_placements/selective_e3/$NAME <<'PY'
import json, sys, glob
p = sys.argv[1]
print("planned bootstraps", sum(json.load(open(f))["summary"]["total_bootstraps"] for f in glob.glob(f"{p}/block_*_placement.json")))
PY
