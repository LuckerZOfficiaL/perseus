#!/bin/bash
# E5c: fine-tune under late-inject-additive and late-inject-attention (k_late in {1,2,3,4}), then re-run the sweep
# on each fine-tuned model at k_late in {1,2,3,4}; after the E5b (replace) pipeline so the GPU is not oversubscribed.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/sat/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe
cd $REPO
until grep -q 'EXIT_CODE=' $D/e5b/pipeline.log 2>/dev/null; do sleep 30; done
mkdir -p $D/e5c
for inj in add attention; do
    F=""; [ $inj = attention ] && F="--fusion $D/e5c/bert_late_$inj/fusion.pt"
    echo "=== bert $inj train ==="; $PY $S/e5b_finetune.py --model bert --inject $inj --steps 1500 --out $D/e5c/bert_late_$inj || exit 1
    echo "=== bert $inj eval ===";  $PY $S/e5_late_inject_bert.py --model $D/e5c/bert_late_$inj --inject $inj $F --klate 0,1,2,3,4,12 --out $D/e5c/bert_sst2_${inj}_ft.json || exit 1
done
for inj in add attention; do
    F=""; [ $inj = attention ] && F="--fusion $D/e5c/vit_late_$inj/fusion.pt"
    echo "=== vit $inj train ===";  $PY $S/e5b_finetune.py --model vit --inject $inj --steps 2000 --train-n 16000 --out $D/e5c/vit_late_$inj || exit 1
    echo "=== vit $inj eval ===";   $PY $S/e5_late_inject_vit.py --model $D/e5c/vit_late_$inj --inject $inj $F --n 2000 --klate 0,1,2,3,4,12 --placeholders drop --out $D/e5c/vit_tinyimagenet_${inj}_ft.json || exit 1
done
echo "=== e5c done ==="
