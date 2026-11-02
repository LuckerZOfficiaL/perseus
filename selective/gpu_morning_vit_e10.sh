#!/bin/bash
# 2026-10-29 morning, on a granted card: E10 batched ViT on both chunks for small private boxes: 2x2 box, 24 images per
# page; 3x3 box, 12 per page (original weights, Tiny-ImageNet). Starts when the card's BERT E1b run (tmux m_bert12,
# sel_g12_ft_872) has ended; that queue's original-weights run is dropped (it runs on another card).
#   tmux new-session -d -s m_vite10 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning_vit_e10.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/e9
P=bootstrap_placements/selective_e3
# the queue's script (QPID: bash selective/gpu_morning_bert12.sh); its next run is the dropped one: kill only that
# script's own child (the same command line also runs on another card)
QPID=$(pgrep -u "$USER" -f "^bash selective/gpu_morning_bert12.sh")
until grep -q "sel_g12_ft_872 end" $D/night/bert/queue_bert12.log; do sleep 5; done
for i in $(seq 30); do
    C=$(pgrep -P "$QPID" -f "e8_run_gpu.py" 2>/dev/null)
    [ -n "$C" ] && { kill $C; break; }
    sleep 1
done
tmux kill-session -t m_bert12 2>/dev/null
sleep 5
source selective/gpu_guard.sh
export SEL_ROUTE=pcmm
for spec in "2 24 12" "3 12 6"; do
    set -- $spec; B=$1; NB=$2; C0=$3
    while [ ! -f $P/vit_sel_box${B}_b${NB}c/pcmm_calib.json ]; do sleep 60; done
    run box${B}_b${NB}c env PF_PCMM_CALIB_IN=$P/vit_sel_box${B}_b${NB}c/pcmm_calib.json $PY selective/e7_run_gpu.py \
        --plan $P/vit_sel_box${B}_b${NB}c --inputs $NB --chunk0 $C0 --n 300 --box $B --images $D/images_tiny300 \
        --out $O/box${B}_b${NB}c_tiny300.json
done
echo "=== all done $(date +%H:%M:%S) ==="
