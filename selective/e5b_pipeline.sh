#!/bin/bash
# E5b: fine-tune under late injection (k_late in {1,2,3,4}), then re-run the E5 sweep on the fine-tuned weights.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/sat/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe
cd $REPO
echo "=== bert late train ==="; $PY $S/e5b_finetune.py --model bert --steps 1500 --out $D/e5b/bert_late || exit 1
echo "=== bert late eval ===";  $PY $S/e5_late_inject_bert.py --model $D/e5b/bert_late --klate 0,1,2,3,4,12 --out $D/e5b/bert_sst2_ft.json || exit 1
echo "=== vit late train ===";  $PY $S/e5b_finetune.py --model vit --steps 2000 --train-n 16000 --out $D/e5b/vit_late || exit 1
echo "=== vit late eval ===";   $PY $S/e5_late_inject_vit.py --model $D/e5b/vit_late --n 2000 --klate 0,1,2,3,4,12 --placeholders drop --out $D/e5b/vit_tinyimagenet_ft.json || exit 1
echo "=== e5b done ==="
