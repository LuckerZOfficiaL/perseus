#!/bin/bash
# 2026-10-29 morning, GPU 6 after its BERT E1b run (gpu_morning3.sh stopped by stop_m3_after_ft.sh): E10 ViT 2x2 box,
# 21 per page (12 + 9; 24 per page exceeds the device PCMM's exact int32 bound: 144 input blocks x 512 rows > 65535),
# 300 Tiny-ImageNet images, original weights.
#   tmux new-session -d -s m5 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning5.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/e9
P=bootstrap_placements/selective_e3/vit_sel_box2_b21c
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150
until grep -q "m3 stopped" $D/night/queue_m3.log; do sleep 10; done
while [ ! -f $P/pcmm_calib.json ]; do sleep 30; done
source selective/gpu_guard.sh
run box2_b21c env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e7_run_gpu.py --plan $P --inputs 21 \
    --chunk0 12 --n 300 --box 2 --images $D/images_tiny300 --out $O/box2_b21c_tiny300.json
echo "=== all done $(date +%H:%M:%S) ==="
