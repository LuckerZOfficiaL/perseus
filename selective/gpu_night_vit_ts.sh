#!/bin/bash
# Night of 2026-10-29, on a granted card: two-stream, 3 images per encrypted page (E7), E1b fine-tuned weights, on the
# 300-image Tiny-ImageNet set (disjoint from images_tiny32), the same 4x4 private box at (3,3) for every image.
#   tmux new-session -d -s n_ts "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_vit_ts.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/vit
IM=$D/images_tiny300
source selective/gpu_guard.sh
export SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$D/e7/pcmm_calib_b3.json
run b3_ft_tiny300 $PY selective/e7_run_gpu.py --plan bootstrap_placements/selective_e3/vit_sel_box4_b3 --inputs 3 --n 300 \
    --images $IM --weights $D/e1b/vit_oneway --out $O/b3_ft_tiny300.json
echo "=== all done $(date +%H:%M:%S) ==="
