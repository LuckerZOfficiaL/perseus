#!/bin/bash
# E6: late-inject-replace k=2 with the per-image hand-over plaintexts built directly (--direct-handover, no twin pass);
# same 3 EuroSAT images, boxes and weights as gpu_queue_e6.sh, so results must equal late_k2.json's.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e6/gpu_queue
source selective/gpu_guard.sh
run late_k2_direct $PY selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k2_box4 --k 2 --n 3 \
    --weights $D/e5b/vit_late --at2 10,9 --direct-handover --out $O/late_k2_direct.json
echo "=== all done $(date +%H:%M:%S) ==="
