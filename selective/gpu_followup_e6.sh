#!/bin/bash
# E6 follow-up on the granted card, after gpu_queue_e6.sh: accuracy on 32 Tiny-ImageNet validation images, the same
# 4x4 private box (16 patches at (3,3)) for every method, each with its fine-tuned weights:
#   two-stream + E1b weights, late-inject-replace k=2 and k=1 + E5b weights.
# (two-stream with the original weights on the same images: ~/data/selective-fhe/e3/gpu_results_pcmm_tiny32.json)
#   tmux new-session -d -s e6fu "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_followup_e6.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e6/gpu_tiny32
IM=$D/e3/images_tiny32
source selective/gpu_guard.sh
run two_stream_ft env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$D/e3/pcmm_calib_sel.json \
    $PY selective/e3_run_gpu.py --plan bootstrap_placements/selective_e3/vit_sel_box4_pcmm --n 32 --box 4 --images $IM \
    --weights $D/e1b/vit_oneway --out $O/two_stream_ft.json
for K in 2 1; do
    run late_k$K $PY selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k${K}_box4 --k $K --n 32 \
        --images $IM --weights $D/e5b/vit_late --out $O/late_k$K.json
done
echo "=== all done $(date +%H:%M:%S) ==="
