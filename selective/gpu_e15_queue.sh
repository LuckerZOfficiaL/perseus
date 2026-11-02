#!/bin/bash
# E15 (tasks whose answer depends on the private part), encrypted runs: a card's job list, each job waiting for its plan.
# Same conventions as gpu_v10_queue.sh (gpu_guard.sh; pools serialised host-wide; XPT_PAD=2).
#   tmux new-session -d -s e15_q5 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> JOBS='drugs_orig ages_s_ft' AFTER=<tmux> \
#       bash selective/gpu_e15_queue.sh"
# Jobs: drugs_{orig,ft} (drug reviews, 12 per page, 2 key blocks; private = medications / conditions; orig = the ordinary
# fine-tune, ft = its one-way fine-tune), ages_{s,m,l}_{orig,ft} (IMDB-WIKI age photos, private = the face; size buckets
# as the E12 faces; the held-out calibration tail is train photos, so --heldout 24 everywhere).
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e15
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 SEL_ROUTE=pcmm
source selective/gpu_guard.sh
while tmux has-session -t "${AFTER:-none}" 2>/dev/null; do sleep 20; done
P3=bootstrap_placements/selective_e3
P8=bootstrap_placements/selective_e8
waitplan() { while [ ! -f "$1/pcmm_calib.json" ]; do sleep 30; done; }
drugs() {   # weights
    local w=$1
    local P=$P8/bert_drugs_g12m; [ $w = ft ] && P=$P8/bert_drugsft_g12m; waitplan $P
    local WA="--weights $D/exports/drugs_bert/classic"; [ $w = ft ] && WA="--weights $D/exports/drugs_oneway/classic"
    run drugs_$w env XPT_PAD=2 XPT_MB=2 SFHE_PROBES=8 SFHE_TEXT_DATA=$O/drugs_t128.json \
        BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_drugsb/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 \
        --n-e-cap 15 --idx 0:1000 --workers 4 --private $O/drugs_t128_private.json $WA --out $O/drugs_$w.json
}
ages() {    # bucket weights
    local b=$1 w=$2 args
    case $b in
        s) args="--k-min 1 --k-max 20 --cap 21 --inputs 8 --chunk0 4" ;;
        m) args="--k-min 21 --k-max 31 --cap 32 --inputs 4 --chunk0 2" ;;
        l) args="--k-min 32 --k-max 63 --cap 64 --inputs 2 --chunk0 1" ;;
    esac
    local P=$P3/vit_ages_$b; [ $w = ft ] && P=$P3/vit_agesft_$b; waitplan $P
    local WA="--weights $O/vit_age"; [ $w = ft ] && WA="--weights $O/ages_oneway"
    run ages_${b}_$w env XPT_PAD=2 SFHE_FACES=ages VIT_CONFIGS=$REPO/selective/sfhe/configs/vit_base_224_ages2/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e11_run_faces_gpu.py --plan $P $args --heldout 24 \
        $WA --out $O/ages_${b}_$w.json
}
for job in $JOBS; do
    case $job in
        drugs_*) drugs ${job#drugs_} ;;
        ages_*) set -- ${job//_/ }; ages $2 $3 ;;
        *) echo "unknown job $job" ;;
    esac
done
echo "=== all done $(date +%H:%M:%S) ==="
