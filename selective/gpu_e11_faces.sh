#!/bin/bash
# E11 on a granted card: one face-photo size bucket (e11_run_faces_gpu.py), original weights then E1b (WHICH="orig ft"),
# XPT_PAD=1.
#   tmux new-session -d -s g11_X "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> NAME=vit_faces_s ARGS='--k-min 1 --k-max 20 \
#       --cap 21 --inputs 8 --chunk0 4 --heldout 24' bash selective/gpu_e11_faces.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e11/gpu
P=bootstrap_placements/selective_e3/$NAME
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 XPT_PAD=1 SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$P/pcmm_calib.json
source selective/gpu_guard.sh
for W in ${WHICH:-orig ft}; do
    WA=""; [ $W = ft ] && WA="--weights $D/e1b/vit_oneway"
    run ${NAME}_$W $PY selective/e11_run_faces_gpu.py --plan $P $ARGS $WA --out $O/${NAME}_$W.json
done
echo "=== all done $(date +%H:%M:%S) ==="
