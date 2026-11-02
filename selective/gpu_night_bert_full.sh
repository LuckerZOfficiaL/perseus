#!/bin/bash
# Night of 2026-10-29, on a granted card: the shipped encrypted BERT-base (full model, one 128-token page) as the
# baseline for E8: the 128-token text and the first 12 SST-2 validation sentences (one compile per sentence length).
#   tmux new-session -d -s n_bfull "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> bash selective/gpu_night_bert_full.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/night/bert
source selective/gpu_guard.sh
export PYTHONPATH=python
run bert_full_text128 $PY -m perseus.examples.bert.run_bert --text 128 --repeat 3
run bert_full_sst2_0_12 $PY -m perseus.examples.bert.run_bert --text "" --sst2 0:12
echo "=== all done $(date +%H:%M:%S) ==="
