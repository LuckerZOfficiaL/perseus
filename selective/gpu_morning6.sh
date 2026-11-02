#!/bin/bash
# 2026-10-29: device-only BERT page failures (sel_g12_ft_872 pages 19 and 26 garbage on the GPU, clean on the twin):
# rerun those two pages (and the clean page 18) exactly as in the full run (same pages, same weight scales over all 73
# pages) with the E1b weights and a per-block check, twice. Runs after the
# card's current queue.
#   tmux new-session -d -s m6 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> AFTER=<tmux session> bash selective/gpu_morning6.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150
while tmux has-session -t "$AFTER" 2>/dev/null; do sleep 15; done
source selective/gpu_guard.sh
for rep in 1 2; do
    run diag_p19_26_ft_rep$rep $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g12 --inputs 12 \
        --chunk0 6 --idx 0:872 --pages 18,19,26 --workers 3 --check --weights $D/exports/bert_oneway/classic --out $O/diag_p19_26_ft_rep$rep.json
done
echo "=== all done $(date +%H:%M:%S) ==="
