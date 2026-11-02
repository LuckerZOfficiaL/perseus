#!/bin/bash
# 2026-10-29 morning, on a granted card: E10, two-stream BERT with 12 sentences per page (6 per chunk, both chunks):
# 12 sentences (original weights) as the device check, then all 872 SST-2 validation sentences with the E1b weights,
# then with the original weights.
#   tmux new-session -d -s m_bert12 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_morning_bert12.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
FT=$D/exports/bert_oneway/classic
source selective/gpu_guard.sh
P=bootstrap_placements/selective_e8/bert_sel_g12
run sel_g12_12 $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --idx 0:12 --out $O/sel_g12_12.json
grep -q "EXIT_CODE=0" $O/sel_g12_12.log || { echo "device check failed"; exit 1; }
run sel_g12_ft_872 $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --idx 0:872 --weights $FT --out $O/sel_g12_ft_872.json
run sel_g12_orig_872 $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --idx 0:872 --out $O/sel_g12_orig_872.json
echo "=== all done $(date +%H:%M:%S) ==="
