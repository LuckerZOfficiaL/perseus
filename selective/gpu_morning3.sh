#!/bin/bash
# 2026-10-29 08:00 relaunch after the out-of-memory stop (two runs x 16 page-recording workers), one card, in order:
#   1. E8/E10 BERT 12 per page, 872 SST-2 validation sentences, E1b weights
#   2. E10 ViT both chunks: 2x2 box x 24 per page, 3x3 box x 12 per page (300 images, original weights)
#   3. E10 ViT 4x4 box x 6 per page, 300 images, E1b weights
#   4. E8/E10 BERT 12 per page, 872 sentences, original weights
# 4 page-recording workers, host-memory guard (gpu_guard.sh MEM_FLOOR_GB), Hub access off.
#   tmux new-session -d -s m3 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning3.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150
O=$D/night/bert
source selective/gpu_guard.sh
PB=bootstrap_placements/selective_e8/bert_sel_g12
PV=bootstrap_placements/selective_e3
run sel_g12_ft_872 $PY selective/e8_run_gpu.py --plan $PB --inputs 12 --chunk0 6 --idx 0:872 --workers 4 \
    --weights $D/exports/bert_oneway/classic --out $O/sel_g12_ft_872.json
O=$D/night/e9
for spec in "2 24 12" "3 12 6"; do
    set -- $spec; B=$1; NB=$2; C0=$3
    run box${B}_b${NB}c env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$PV/vit_sel_box${B}_b${NB}c/pcmm_calib.json $PY selective/e7_run_gpu.py \
        --plan $PV/vit_sel_box${B}_b${NB}c --inputs $NB --chunk0 $C0 --n 300 --box $B --images $D/images_tiny300 \
        --out $O/box${B}_b${NB}c_tiny300.json
done
O=$D/night/vit
run b6c_ft_tiny300 env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$PV/vit_sel_box4_b6c/pcmm_calib.json $PY selective/e7_run_gpu.py \
    --plan $PV/vit_sel_box4_b6c --inputs 6 --chunk0 3 --n 300 --images $D/images_tiny300 --weights $D/e1b/vit_oneway \
    --out $O/b6c_ft_tiny300.json
O=$D/night/bert
run sel_g12_orig_872 $PY selective/e8_run_gpu.py --plan $PB --inputs 12 --chunk0 6 --idx 0:872 --workers 4 \
    --out $O/sel_g12_orig_872.json
echo "=== all done $(date +%H:%M:%S) ==="
