#!/bin/bash
# E15 drug reviews with E4's fixed detector (condition names private too), encrypted runs; a copy of gpu_e15_queue.sh
# (that one was running when this was written) with the drugsc_{orig,ft} jobs.
# Same conventions as gpu_v10_queue.sh (gpu_guard.sh; pools serialised host-wide; XPT_PAD=2).
#   tmux new-session -d -s e15c_q5 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> JOBS='drugsc_ft drugsc_orig' AFTER=<tmux> \
#       bash selective/gpu_e15c_queue.sh"
# Jobs: drugsc_{orig,ft} (drug reviews, 12 per page, 2 key blocks; private = medications, diseases and condition names;
# orig = the ordinary fine-tune, ft = its one-way fine-tune).
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e15
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 SEL_ROUTE=pcmm
source selective/gpu_guard.sh
while tmux has-session -t "${AFTER:-none}" 2>/dev/null; do sleep 20; done
P8=bootstrap_placements/selective_e8
waitplan() { while [ ! -f "$1/pcmm_calib.json" ]; do sleep 30; done; }
drugsc() {  # weights
    local w=$1
    local P=$P8/bert_drugsc_g12m; [ $w = ft ] && P=$P8/bert_drugscft_g12m; waitplan $P
    local WA="--weights $D/exports/drugs_bert/classic"; [ $w = ft ] && WA="--weights $D/exports/drugs_oneway/classic"
    run drugsc_$w env XPT_PAD=2 XPT_MB=2 SFHE_PROBES=8 SFHE_TEXT_DATA=$O/drugs_t128.json \
        BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_drugsb/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 \
        --n-e-cap 15 --idx 0:1000 --workers 4 --private $O/drugs_t128c_private.json $WA --out $O/drugsc_$w.json
}
for job in $JOBS; do
    case $job in
        drugsc_*) drugsc ${job#drugsc_} ;;
        *) echo "unknown job $job" ;;
    esac
done
echo "=== all done $(date +%H:%M:%S) ==="
