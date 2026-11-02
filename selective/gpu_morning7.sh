#!/bin/bash
# 2026-10-29: the padding-lane fix (XPT_PAD=1: free lanes query the last input's plaintext keys and stay in range) on the
# device, with the existing plans (same ops; the plans were cut on the larger, drifting padding magnitudes).
#   WHAT=vit  : two-stream 3 per page, E1b weights, the 300-image set (before: page 50 broken, 297/300)
#   WHAT=bert : BERT 12 per page, E1b weights, pages 18 / 19 / 26 of the 872-sentence run with the per-block check
#   tmux new-session -d -s m7_X "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> WHAT=vit|bert bash selective/gpu_morning7.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 XPT_PAD=1
if [ "$WHAT" = vit ]; then
    O=$D/night/vit
    source selective/gpu_guard.sh
    run b3_ft_tiny300_pad env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$D/e7/pcmm_calib_b3.json $PY selective/e7_run_gpu.py \
        --plan bootstrap_placements/selective_e3/vit_sel_box4_b3 --inputs 3 --n 300 --images $D/images_tiny300 \
        --weights $D/e1b/vit_oneway --out $O/b3_ft_tiny300_pad.json
else
    O=$D/night/bert
    source selective/gpu_guard.sh
    run diag_p19_26_ft_pad $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g12 --inputs 12 \
        --chunk0 6 --idx 0:872 --pages 18,19,26 --workers 3 --check --weights $D/exports/bert_oneway/classic \
        --out $O/diag_p19_26_ft_pad.json
fi
echo "=== all done $(date +%H:%M:%S) ==="
