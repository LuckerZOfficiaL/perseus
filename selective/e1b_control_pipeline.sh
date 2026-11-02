#!/bin/bash
# E1b control: the same fine-tune recipe with nothing sensitive (plain fine-tune), then the E1 sweep on it.
# Separates the effect of the one-way training signal from the effect of fine-tuning on the eval domain.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/sat/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe
cd $REPO
until grep -q 'EXIT_CODE=' $D/e1b/pipeline.log; do sleep 20; done     # GPU share: after the main E1b pipeline
echo "=== bert plain-control train ==="; $PY $S/e1b_finetune.py --model bert --steps 1500 --batch 32 --fracs 0 --out $D/e1b/bert_plain || exit 1
echo "=== bert plain-control eval ===";  $PY $S/e1_oneway_bert.py --model $D/e1b/bert_plain --out $D/e1b/bert_sst2_plainft.json || exit 1
echo "=== vit plain-control train ===";  $PY $S/e1b_finetune.py --model vit --steps 2000 --batch 32 --train-n 16000 --ks 0 --out $D/e1b/vit_plain || exit 1
echo "=== vit plain-control eval ===";   $PY $S/e1_oneway_vit.py --model $D/e1b/vit_plain --n 2000 --batch 100 --out $D/e1b/vit_tinyimagenet_plainft.json || exit 1
echo "=== control pipeline done ==="
