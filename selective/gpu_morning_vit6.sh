#!/bin/bash
# 2026-10-29 morning, on a granted card: E10, two-stream ViT with 6 images per page (3 per chunk, both chunks),
# 4x4 private box: 30 Tiny-ImageNet images (original weights), then the 300-image set with the E1b weights (next to the
# 3-per-page run b3_ft_tiny300).
#   tmux new-session -d -s m_vit6 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning_vit6.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/vit
source selective/gpu_guard.sh
export SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=bootstrap_placements/selective_e3/vit_sel_box4_b6c/pcmm_calib.json
P=bootstrap_placements/selective_e3/vit_sel_box4_b6c
run b6c_tiny30 $PY selective/e7_run_gpu.py --plan $P --inputs 6 --chunk0 3 --n 30 --images $D/e3/images_tiny32 --out $O/b6c_tiny30.json
run b6c_ft_tiny300 $PY selective/e7_run_gpu.py --plan $P --inputs 6 --chunk0 3 --n 300 --images $D/images_tiny300 \
    --weights $D/e1b/vit_oneway --out $O/b6c_ft_tiny300.json
echo "=== all done $(date +%H:%M:%S) ==="
