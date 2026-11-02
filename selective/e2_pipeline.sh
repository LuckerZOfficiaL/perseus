#!/bin/bash
# E2: capture the ViT circuit at several token counts / channel widths on the CPU twin, plan each with the
# shipped ViT recipe, so bootstraps and op counts can be compared with the 197-token circuit.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e2
cd $REPO
export CUDA_VISIBLE_DEVICES=""
export PYTHON=$PY
# the shipped plan's recipe (bootstrap_placements/vit_b16_224/PLAN_CMD.txt, "replanned=" line)
RECIPE="PLAN_DELIB_DEG=captured PLAN_SPARSE_BTS_OUT=1:16,512:24 BTS_LEVEL=28 PLAN_RAISE_DROP_MAX=0 PLAN_RAISE_LANDING_MAX=42 PLAN_RAISE_ROUTES=0 PLAN_SLIM_CUBIC=1 PLAN_STC_FIRST_ENTRY=46 PLAN_STC_FIRST_FALLBACK=34:1 PLAN_STC_FIRST_REROUTE=1"
run() {  # run <tokens> <nch>
    local T=$1 NCH=$2 name=vit_T${1}_nch${2} g=$D/graphs/vit_T${1}_nch${2}
    if ! grep -q 'EXIT_CODE=0' $D/logs/capture_T${T}_nch${NCH}.log 2>/dev/null; then
        echo "=== capture $name ==="
        $PY $S/e2_capture_vit.py --tokens $T --nch $NCH --out $g > $D/logs/capture_T${T}_nch${NCH}.log 2>&1
        echo "EXIT_CODE=$?" >> $D/logs/capture_T${T}_nch${NCH}.log
        grep -q 'EXIT_CODE=0' $D/logs/capture_T${T}_nch${NCH}.log || { echo "capture $name failed"; return 1; }
    fi
    echo "=== plan $name ==="
    env $RECIPE bash python/perseus/examples/gpt2/make_plan.sh $g selective_e2/$name > $D/logs/plan_${name}.log 2>&1 \
        || { echo "plan $name failed (see $D/logs/plan_${name}.log)"; return 1; }
    grep -E 'placements:' $D/logs/plan_${name}.log | tail -12
}
# usage: e2_pipeline.sh [--fresh] T:NCH [T:NCH ...]   (--fresh removes the config's old capture first)
FRESH=0; [ "${1:-}" = "--fresh" ] && { FRESH=1; shift; }
for cfg in "$@"; do
    T=${cfg%%:*}; NCH=${cfg##*:}
    if [ $FRESH = 1 ]; then rm -rf $D/graphs/vit_T${T}_nch${NCH} $D/logs/capture_T${T}_nch${NCH}.log bootstrap_placements/selective_e2/vit_T${T}_nch${NCH}; fi
    run $T $NCH
done
echo "=== e2 pipeline done ==="
