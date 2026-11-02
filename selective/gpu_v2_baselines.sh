#!/bin/bash
# E12 baselines on THIS upstream version: the full encrypted ViT-B/16 and BERT-base exactly as upstream's
# scripts/repro/{vit,bert}_base.sh run them (CUDA graphs, --stc-cache 32 for ViT), then eager (as our runs), same card.
#   tmux new-session -d -s v2_base "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_v2_baselines.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
O=/home/lzhou/data/selective-fhe/e12/baselines
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=$REPO/python MEM_FLOOR_GB=150
source selective/gpu_guard.sh
echo "upstream $(git log --oneline -1)"
run vit_full_graphs $PY -m perseus.examples.vit.run_prefill --images scripts/repro/data/vit_eurosat --n 4 --stc-cache 32 --graphs
run bert_full_graphs $PY -m perseus.examples.bert.run_bert --text 128 --graphs --repeat 3
run vit_full_eager $PY -m perseus.examples.vit.run_prefill --images scripts/repro/data/vit_eurosat --n 4 --stc-cache 32
run bert_full_eager $PY -m perseus.examples.bert.run_bert --text 128 --repeat 3
run bert_full_sst2_eager $PY -m perseus.examples.bert.run_bert --text "" --sst2 0:6
echo "=== all done $(date +%H:%M:%S) ==="
