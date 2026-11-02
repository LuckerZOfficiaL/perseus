#!/bin/bash
# 2026-10-29 morning (relaunch, Hub access off: the per-sentence tokenizer loads hit HF rate limits), on a granted card:
#   WHAT=ft   : two-stream BERT, 12 per page, all 872 SST-2 validation sentences, E1b weights; then E10 ViT runs on both
#               chunks (2x2 box x 24 per page, 3x3 box x 12 per page, 300 Tiny-ImageNet images, original weights)
#   WHAT=orig : the same BERT run with the original weights
#   tmux new-session -d -s m2_X "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> WHAT=ft|orig bash selective/gpu_morning2_bert_vit.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
source selective/gpu_guard.sh
P=bootstrap_placements/selective_e8/bert_sel_g12
if [ "$WHAT" = ft ]; then
    run sel_g12_ft_872 $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --idx 0:872 \
        --weights $D/exports/bert_oneway/classic --out $O/sel_g12_ft_872.json
    O=$D/night/e9
    PV=bootstrap_placements/selective_e3
    export SEL_ROUTE=pcmm
    for spec in "2 24 12" "3 12 6"; do
        set -- $spec; B=$1; NB=$2; C0=$3
        while [ ! -f $PV/vit_sel_box${B}_b${NB}c/pcmm_calib.json ]; do sleep 60; done
        run box${B}_b${NB}c env PF_PCMM_CALIB_IN=$PV/vit_sel_box${B}_b${NB}c/pcmm_calib.json $PY selective/e7_run_gpu.py \
            --plan $PV/vit_sel_box${B}_b${NB}c --inputs $NB --chunk0 $C0 --n 300 --box $B --images $D/images_tiny300 \
            --out $O/box${B}_b${NB}c_tiny300.json
    done
else
    run sel_g12_orig_872 $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --idx 0:872 --out $O/sel_g12_orig_872.json
fi
echo "=== all done $(date +%H:%M:%S) ==="
