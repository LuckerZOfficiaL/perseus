#!/bin/bash
# Night of 2026-10-29, on a granted card: late-inject-replace k=2 then k=1, E5b fine-tuned weights, direct hand-over,
# on the 300-image Tiny-ImageNet set, the same single 4x4 private box at (3,3) as the two-stream run.
#   tmux new-session -d -s n_late "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_vit_late.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/vit
IM=$D/images_tiny300
source selective/gpu_guard.sh
for K in 2 1; do
    run late_k${K}_tiny300 $PY selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k${K}_box4 --k $K \
        --n 300 --images $IM --weights $D/e5b/vit_late --direct-handover --out $O/late_k${K}_tiny300.json
done
echo "=== all done $(date +%H:%M:%S) ==="
