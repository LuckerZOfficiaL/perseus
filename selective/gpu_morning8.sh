#!/bin/bash
# 2026-10-29: is the BERT block-9 device failure a PCMM band overflow? Pages 18 / 19 / 26 (E1b) with the PCMM margin
# doubled (PF_PCMM_MARGIN=3: every band shift one power of two higher). Fixed -> band; still broken -> a refresh's range.
#   tmux new-session -d -s m8 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> AFTER=<tmux session> bash selective/gpu_morning8.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 PF_PCMM_MARGIN=3
while tmux has-session -t "$AFTER" 2>/dev/null; do sleep 10; done
source selective/gpu_guard.sh
run diag_p19_26_ft_margin3 $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g12 --inputs 12 \
    --chunk0 6 --idx 0:872 --pages 18,19,26 --workers 3 --check --weights $D/exports/bert_oneway/classic \
    --out $O/diag_p19_26_ft_margin3.json
echo "=== all done $(date +%H:%M:%S) ==="
