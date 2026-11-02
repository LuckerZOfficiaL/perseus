#!/bin/bash
# Night of 2026-10-29, on a granted card: E9, two-stream ViT cost versus the private box (route B), after the
# two-stream tiny300 run (tmux n_ts). Single input per page: boxes 2, 3, 4, 5, 6, 7 (4 to 49 private patches) on 6
# Tiny-ImageNet images each; batched: box 2 x 12 per page (24 images), box 3 x 6 per page (30 images). Original weights.
#   tmux new-session -d -s n_e9 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_e9.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/e9
IM=$D/e3/images_tiny32
P=bootstrap_placements/selective_e3
source selective/gpu_guard.sh
export SEL_ROUTE=pcmm
while tmux has-session -t n_ts 2>/dev/null; do sleep 30; done
for B in 2 3 4 5 6 7; do
    while [ ! -f $P/vit_sel_box${B}_pcmm/pcmm_calib.json ]; do sleep 60; done
    run box${B}_single env PF_PCMM_CALIB_IN=$P/vit_sel_box${B}_pcmm/pcmm_calib.json $PY selective/e3_run_gpu.py \
        --plan $P/vit_sel_box${B}_pcmm --n 6 --box $B --images $IM --out $O/box${B}_single.json
done
for spec in "2 12" "3 6"; do
    set -- $spec; B=$1; NB=$2
    while [ ! -f $P/vit_sel_box${B}_b$NB/pcmm_calib.json ]; do sleep 60; done
    run box${B}_b$NB env PF_PCMM_CALIB_IN=$P/vit_sel_box${B}_b$NB/pcmm_calib.json $PY selective/e7_run_gpu.py \
        --plan $P/vit_sel_box${B}_b$NB --inputs $NB --n 30 --box $B --images $IM --out $O/box${B}_b$NB.json
done
echo "=== all done $(date +%H:%M:%S) ==="
