#!/bin/bash
# Rerun of gpu_queue_e6.sh's two late-inject-replace timing runs (the first attempt declared blocks 0..k's linears
# instead of the last k blocks'; fixed in LatePlannedPrefill.linears). Same images, boxes and weights as the queue.
#   tmux new-session -d -s e6re "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_rerun_late_e6.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e6/gpu_queue
source selective/gpu_guard.sh
for K in 2 1; do
    run late_k$K $PY selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k${K}_box4 --k $K --n 3 \
        --weights $D/e5b/vit_late --at2 10,9 --out $O/late_k$K.json
done
echo "=== all done $(date +%H:%M:%S) ==="
