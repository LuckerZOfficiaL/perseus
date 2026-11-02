#!/bin/bash
# 2026-10-29 08:05, the morning queue split over three cards (GPU 6 keeps the BERT E1b 872 run of gpu_morning3.sh; its
# later items move here and gpu_morning3.sh is stopped once that run ends).
#   WHAT=vit  : E10 ViT both chunks, 300 Tiny-ImageNet images: 2x2 x 24 / 3x3 x 12 per page (original weights), then
#               4x4 x 6 per page with the E1b weights
#   WHAT=bert : E10 BERT 12 per page, 872 SST-2 validation sentences, original weights; starts after the E1b run's page
#               recording (one 4-worker pool at a time on the host)
#   tmux new-session -d -s m4_X "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> WHAT=vit|bert bash selective/gpu_morning4.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150
PV=bootstrap_placements/selective_e3
if [ "$WHAT" = vit ]; then
    O=$D/night/e9
    source selective/gpu_guard.sh
    for spec in "2 24 12" "3 12 6"; do
        set -- $spec; B=$1; NB=$2; C0=$3
        run box${B}_b${NB}c env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$PV/vit_sel_box${B}_b${NB}c/pcmm_calib.json $PY selective/e7_run_gpu.py \
            --plan $PV/vit_sel_box${B}_b${NB}c --inputs $NB --chunk0 $C0 --n 300 --box $B --images $D/images_tiny300 \
            --out $O/box${B}_b${NB}c_tiny300.json
    done
    O=$D/night/vit
    run b6c_ft_tiny300 env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$PV/vit_sel_box4_b6c/pcmm_calib.json $PY selective/e7_run_gpu.py \
        --plan $PV/vit_sel_box4_b6c --inputs 6 --chunk0 3 --n 300 --images $D/images_tiny300 --weights $D/e1b/vit_oneway \
        --out $O/b6c_ft_tiny300.json
else
    O=$D/night/bert
    until grep -q "planned setup\|Traceback\|GUARD" $O/sel_g12_ft_872.log 2>/dev/null; do sleep 10; done
    source selective/gpu_guard.sh
    run sel_g12_orig_872 $PY selective/e8_run_gpu.py --plan bootstrap_placements/selective_e8/bert_sel_g12 --inputs 12 \
        --chunk0 6 --idx 0:872 --workers 4 --out $O/sel_g12_orig_872.json
fi
echo "=== all done $(date +%H:%M:%S) ==="
