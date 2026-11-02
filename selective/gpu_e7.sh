#!/bin/bash
# E7 on the granted card: two-stream route B with 3 inputs per encrypted page.
#   1. EuroSAT, 3 images = 1 page (timing next to the single-input 18.37 s/image of gpu_queue_e6.sh)
#   2. Tiny-ImageNet, 30 images = 10 pages, original weights (next to e3/gpu_results_pcmm_tiny32.json)
#   tmux new-session -d -s e7gpu "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_e7.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e7/gpu
source selective/gpu_guard.sh
export SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$D/e7/pcmm_calib_b3.json
P=bootstrap_placements/selective_e3/vit_sel_box4_b3
run b3_eurosat $PY selective/e7_run_gpu.py --plan $P --inputs 3 --n 3 --out $O/b3_eurosat.json
run b3_tiny30 $PY selective/e7_run_gpu.py --plan $P --inputs 3 --n 30 --images $D/e3/images_tiny32 --out $O/b3_tiny30.json
echo "=== all done $(date +%H:%M:%S) ==="
