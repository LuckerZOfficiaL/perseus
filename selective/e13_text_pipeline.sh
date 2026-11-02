#!/bin/bash
# E13: the two-stream BERT plan for another text task (a BERT-base export sharing the uncased vocabulary, a dataset in the
# SFHE_TEXT_DATA format, its detector file): LayerNorm bands refitted on its train texts (e8_ln_bands.py), softmax round-0
# bands on top (e12_bert_sm_ranges.py --refit), then calibrate / capture / plan with e8_pipeline.sh on three train pages
# of the detector's private sets (12 per page, 6 + 6, up to 15 encrypted tokens).
#   bash selective/e13_text_pipeline.sh NAME EXPORT_DIR DATA.json PRIVATE.json "set 1" "set 2" "set 3"
# (train indices; each text at most 13 private tokens: CLS and SEP are encrypted too, 15 per input)
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -eu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SEL_ROUTE=pcmm XPT_PAD=2
NAME=${1:?name}; W=${2:?export dir}; DATA=${3:?texts}; PRIV=${4:?detector file}; shift 4
export SFHE_TEXT_DATA=$DATA
PY=/home/lzhou/data/venvs/perseus/bin/python
C=$REPO/selective/sfhe/configs
cd $REPO
echo "=== LayerNorm bands $NAME ==="
$PY selective/e8_ln_bands.py --train 3000 --val 1924 --weights $W --out $C/bert_base_n32_t128_${NAME}/configs.json
echo "=== softmax ranges $NAME ==="
$PY selective/e12_bert_sm_ranges.py --train 3000 --weights $W --refit --margin 4 \
    --base $C/bert_base_n32_t128_${NAME}/configs.json --out $C/bert_base_n32_t128_${NAME}b/configs.json
echo "=== plan $NAME ==="
BERT_CONFIGS=$C/bert_base_n32_t128_${NAME}b/configs.json \
    EXTRA="--chunk0 6 --n-e-cap 15 --split train --private $PRIV --weights $W" CALIB_ALL=1 \
    bash selective/e8_pipeline.sh 12 bert_${NAME}_g12c15 "$@"
echo "text_data=$DATA weights=$W private=$PRIV" >> bootstrap_placements/selective_e8/bert_${NAME}_g12c15/PLAN_CMD.txt
