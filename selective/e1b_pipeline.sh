#!/bin/bash
# E1b: fine-tune under the one-way rule, then re-run the E1 evaluation on the fine-tuned weights.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/sat/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe
cd $REPO
echo "=== bert train ===";  $PY $S/e1b_finetune.py --model bert --steps 1500 --batch 32 --out $D/e1b/bert_oneway || exit 1
echo "=== bert eval ===";   $PY $S/e1_oneway_bert.py --model $D/e1b/bert_oneway --out $D/e1b/bert_sst2_ft.json || exit 1
echo "=== vit train ===";   $PY $S/e1b_finetune.py --model vit --steps 2000 --batch 32 --train-n 16000 --out $D/e1b/vit_oneway || exit 1
echo "=== vit eval ===";    $PY $S/e1_oneway_vit.py --model $D/e1b/vit_oneway --n 2000 --batch 100 --out $D/e1b/vit_tinyimagenet_ft.json || exit 1
echo "=== pipeline done ==="
