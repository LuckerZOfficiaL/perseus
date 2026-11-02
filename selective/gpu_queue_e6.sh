#!/bin/bash
# E6 GPU queue: four encrypted runs back to back on the same 3 EuroSAT images, so their timings share one GPU state.
#   1. full ViT-B/16 (shipped plan)          2. two-stream, route B (E3)
#   3. late-inject-replace, k=2 (E6)          4. late-inject-replace, k=1 (E6)
# The late runs use the E5b fine-tuned weights and two 4x4 private boxes (one per 128-token page), as planned.
# GPU_PHYS = the physical card the launcher made visible (state logging + guards). Each run waits while another user's
# process is on that card, and is killed at once if it shows up on any other card.
#   tmux new-session -d -s e6gpu "CUDA_VISIBLE_DEVICES=<granted card> GPU_PHYS=<same> bash selective/gpu_queue_e6.sh"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e6/gpu_queue
GPU_PHYS=${GPU_PHYS:?physical GPU index}
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_PHYS '$1==g {print $2}')
[ -n "$UUID" ] || { echo "no GPU $GPU_PHYS"; exit 1; }
mkdir -p $O
state() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' -v g=$GPU_PHYS '$1==g'; }
apps() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null; }
run() {
    local name=$1; shift
    while apps | grep -q "$UUID"; do echo "[$(date +%H:%M:%S)] GPU $GPU_PHYS busy (another process), waiting"; sleep 60; done
    echo "=== $name start $(date +%H:%M:%S) | GPU $GPU_PHYS $(state) ==="
    "$@" > $O/$name.log 2>&1 &
    local P=$! seen=0
    while kill -0 $P 2>/dev/null; do
        local mine=$(apps | awk -F', ' -v p=$P '$1==p {print $2}')
        if [ -n "$mine" ] && [ "$mine" != "$UUID" ]; then
            kill $P; echo "GUARD: process $P appeared on $mine, not GPU $GPU_PHYS; killed" | tee -a $O/$name.log
        fi
        [ -n "$mine" ] && seen=1
        local others=$(apps | grep "$UUID" | awk -F', ' -v p=$P '$1!=p' | wc -l)
        [ "$others" -gt 0 ] && echo "[$(date +%H:%M:%S)] note: another process is on GPU $GPU_PHYS during $name (timing contended)" | tee -a $O/$name.log
        sleep 5
    done
    wait $P; local rc=$?
    echo "EXIT_CODE=$rc" >> $O/$name.log
    echo "=== $name end $(date +%H:%M:%S) rc=$rc on-target=$seen | GPU $GPU_PHYS $(state) ==="
    grep -E 'SUMMARY|median|s/image' $O/$name.log | tail -2
}
run full env PF_PCMM_CALIB_IN=bootstrap_placements/vit_b16_224/pcmm_calib.json \
    $PY -m perseus.examples.vit.run_prefill --plan bootstrap_placements/vit_b16_224 --images scripts/repro/data/vit_eurosat \
    --n 3 --force --device 0 --stc-cache 8
run two_stream env SEL_ROUTE=pcmm PF_PCMM_CALIB_IN=$D/e3/pcmm_calib_sel.json \
    $PY selective/e3_run_gpu.py --plan bootstrap_placements/selective_e3/vit_sel_box4_pcmm --n 3 --box 4 --out $O/two_stream.json
for K in 2 1; do
    run late_k$K $PY selective/e6_run_gpu.py --plan bootstrap_placements/selective_e3/vit_late_k${K}_box4 --k $K --n 3 \
        --weights $D/e5b/vit_late --at2 10,9 --out $O/late_k$K.json
done
echo "=== all done $(date +%H:%M:%S) ==="
