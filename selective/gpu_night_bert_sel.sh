#!/bin/bash
# Night of 2026-10-29, on a granted card: E8, two-stream BERT encrypted on Perseus (route B, page layouts).
#   1. one sentence per page, 3 sentences (timing next to the full model's 23.9 s)
#   2. six sentences per page, 12 sentences (2 pages: the pipeline on the device)
#   3. six per page, all 872 SST-2 validation sentences, E1b fine-tuned weights (the model two-stream is meant to run)
#   (the original-weights 872 run is on another card: gpu_night_bert_sel_orig.sh)
# Waits for the full-BERT baseline (tmux n_bfull) to finish first.
#   tmux new-session -d -s n_bsel "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_bert_sel.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
FT=$D/exports/bert_oneway/classic
source selective/gpu_guard.sh
while tmux has-session -t n_bfull 2>/dev/null; do sleep 30; done
P1=bootstrap_placements/selective_e8/bert_sel_g1
P6=bootstrap_placements/selective_e8/bert_sel_g6
run sel_g1_3 $PY selective/e8_run_gpu.py --plan $P1 --inputs 1 --idx 0,1,2 --out $O/sel_g1_3.json
while [ ! -f $P6/pcmm_calib.json ]; do sleep 60; done
run sel_g6_12 $PY selective/e8_run_gpu.py --plan $P6 --inputs 6 --idx 0:12 --out $O/sel_g6_12.json
run sel_g6_ft_872 $PY selective/e8_run_gpu.py --plan $P6 --inputs 6 --idx 0:872 --weights $FT --workers 16 --out $O/sel_g6_ft_872.json
echo "=== all done $(date +%H:%M:%S) ==="
