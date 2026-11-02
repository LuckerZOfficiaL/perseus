#!/bin/bash
# Night of 2026-10-29, on a granted card: E8 with the original SST-2 weights on all 872 validation sentences, six per page,
# after the late-injection runs (tmux n_late) and once the 2-page device check (sel_g6_12) has passed.
#   tmux new-session -d -s n_bsel_orig "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_bert_sel_orig.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
source selective/gpu_guard.sh
while tmux has-session -t n_late 2>/dev/null; do sleep 30; done
until grep -q "EXIT_CODE=0" $O/sel_g6_12.log 2>/dev/null; do
    grep -q "EXIT_CODE=[1-9]" $O/sel_g6_12.log 2>/dev/null && { echo "sel_g6_12 failed; not starting"; exit 1; }
    sleep 60
done
run sel_g6_orig_872 $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g6 --inputs 6 --idx 0:872 \
    --workers 16 --out $O/sel_g6_orig_872.json
echo "=== all done $(date +%H:%M:%S) ==="
