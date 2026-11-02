#!/bin/bash
# 2026-10-29 morning, on a granted card: E10, two-stream BERT 12 sentences per page, all 872 SST-2 validation sentences,
# original weights (the E1b run is on another card). Starts once the 6-per-page device check (tmux n_bsel, sel_g6_12)
# is done; that queue's own 872-sentence 6-per-page run is dropped (the 12-per-page runs cover it).
#   tmux new-session -d -s m_bert12o "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning_bert12_orig.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
until grep -q "sel_g6_12 end" $O/queue_sel.log; do sleep 10; done
tmux kill-session -t n_bsel 2>/dev/null
pkill -u "$USER" -f "e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g6 --inputs 6 --idx 0:872" 2>/dev/null
sleep 5
source selective/gpu_guard.sh
run sel_g12_orig_872 $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g12 --inputs 12 --chunk0 6 \
    --idx 0:872 --out $O/sel_g12_orig_872.json
echo "=== all done $(date +%H:%M:%S) ==="
