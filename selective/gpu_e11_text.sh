#!/bin/bash
# E8 rerun + E11 text on a granted card, with the refitted plan bert_sel_g12c15_ln (12 per page, up to 15 encrypted tokens,
# SST-2-refitted LayerNorm bands; PCMM bands: the plan's (train pages) merged with the 34-page train pools).
#   WHAT=ner  : the 363 validation sentences with detected entities (private = the entities), original then E1b weights
#   WHAT=span : all 872 validation sentences, 30% span (the E8 protocol: are pages 19 / 26 fixed?), original then E1b
#   tmux new-session -d -s g11t_X "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> WHAT=ner|span AFTER=<tmux> bash selective/gpu_e11_text.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e11/gpu
P=bootstrap_placements/selective_e8/bert_sel_g12c15_ln
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 XPT_PAD=1 \
    BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_sst2/configs.json PF_PCMM_CALIB_IN=$D/e8/pcmm_calib_g12c15_ln_pool.json
while tmux has-session -t "${AFTER:-none}" 2>/dev/null; do sleep 20; done
while [ ! -f $P/pcmm_calib.json ]; do sleep 30; done
if [ ! -f $PF_PCMM_CALIB_IN ]; then
    $PY - $P/pcmm_calib.json $D/e8/pcmm_calib_pool_ft.json $D/e8/pcmm_calib_pool_orig.json $PF_PCMM_CALIB_IN <<'PY'
import json, sys
mx = {}
for f in sys.argv[1:-1]:
    for k, v in json.load(open(f)).items():
        mx[k] = max(mx.get(k, 0.0), v)
json.dump(mx, open(sys.argv[-1], "w"), indent=1, sort_keys=True)
print("merged PCMM calibration", len(mx))
PY
fi
source selective/gpu_guard.sh
COMMON="--plan $P --inputs 12 --chunk0 6 --n-e-cap 15 --idx 0:872 --workers 4"
if [ "$WHAT" = ner ]; then
    run text_ner_orig $PY selective/e8_run_gpu.py $COMMON --private $D/e11/ner_private.json --out $O/text_ner_orig.json
    run text_ner_ft $PY selective/e8_run_gpu.py $COMMON --private $D/e11/ner_private.json \
        --weights $D/exports/bert_oneway/classic --out $O/text_ner_ft.json
else
    run text_span_orig $PY selective/e8_run_gpu.py $COMMON --out $O/text_span_orig.json
    run text_span_ft $PY selective/e8_run_gpu.py $COMMON --weights $D/exports/bert_oneway/classic --out $O/text_span_ft.json
fi
echo "=== all done $(date +%H:%M:%S) ==="
