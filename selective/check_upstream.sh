#!/bin/bash
# After pulling upstream Perseus into this checkout: rebuild perseus._pc, then run sfhe's circuits on the CPU twin
# (2 layers each) and fail if any per-block error leaves its bound. No GPU. Then, as printed at the end, re-derive the
# artifacts that depend on upstream (the BERT LayerNorm refit, our plans) before any GPU run.
#   git pull upstream main && bash selective/check_upstream.sh
set -u
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO"
PY=${PY:-/home/lzhou/data/venvs/perseus/bin/python}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 XPT_PAD=1
LOG=/home/lzhou/data/selective-fhe/logs_$(basename "$REPO")/check_$(git rev-parse --short HEAD)
mkdir -p "$LOG"
echo "=== upstream $(git log --oneline -1 upstream/main 2>/dev/null) | checkout $(git log --oneline -1) ==="
if [ "${SKIP_BUILD:-0}" != 1 ]; then
    bash selective/build_perseus.sh > "$LOG/build.log" 2>&1 || { echo "BUILD FAILED: $LOG/build.log"; exit 1; }
    echo "build ok"
fi
fail=0
check() {   # name bound cmd...: run, then the largest "maxerr" number in its output must stay under bound
    local name=$1 bound=$2; shift 2
    "$@" > "$LOG/$name.log" 2>&1
    local rc=$?
    local worst=$(grep -oE "worst block maxerr [0-9.e+-]+" "$LOG/$name.log" | awk '{print $4}' | tail -1)
    if [ $rc -ne 0 ] || [ -z "$worst" ] || ! awk -v w="$worst" -v b="$bound" 'BEGIN {exit !(w + 0 < b + 0)}'; then
        echo "FAIL $name (rc=$rc, worst block error ${worst:-none}, bound $bound): $LOG/$name.log"; fail=1
    else
        echo "ok   $name: worst block error $worst (bound $bound)"
    fi
}
check vit_single 0.05 $PY selective/e3_test_xpt.py --layers 2
check vit_3_per_page 0.05 $PY selective/e7_test_batch.py --inputs 3 --layers 2
check vit_6_both_chunks 0.05 $PY selective/e7_test_batch.py --inputs 6 --chunk0 3 --layers 2 --images 0,1,2,3,0,1
check bert_12_both_chunks 0.01 $PY selective/e8_test_bert.py --idx 0,1,2,3,4,5,6,7,8,9,10,11 --chunk0 6 --layers 2
check vit_faces_layout 0.05 $PY selective/e11_test_faces.py --n 6 --chunk0 3 --layers 2
$PY selective/e6_test_late.py --k 1 > "$LOG/vit_late_k1.log" 2>&1 && grep -q "logits:" "$LOG/vit_late_k1.log" \
    && echo "ok   vit_late_k1: $(grep -o 'KL(oracle || encrypted) [^;]*' $LOG/vit_late_k1.log)" \
    || { echo "FAIL vit_late_k1: $LOG/vit_late_k1.log"; fail=1; }
if [ $fail -ne 0 ]; then echo "=== CHECKS FAILED ==="; exit 1; fi
cat <<'TXT'
=== CPU checks passed. Before GPU runs on this upstream version, re-derive what depends on it:
  1. BERT LayerNorm bands from upstream's current config (writes sfhe/configs/bert_base_n32_t128_sst2):
       python selective/e8_ln_bands.py --train 3000 --val 872 --weights '' --weights <E1b export> \
           --out selective/sfhe/configs/bert_base_n32_t128_sst2/configs.json
  2. our plans (capture + plan with upstream's current recipe): selective/e3_routeb_pipeline.sh, e7_pipeline.sh,
     e8_pipeline.sh, e11_faces_pipeline.sh, e6_pipeline.sh
  3. GPU: the full-model baselines on THIS version (upstream's own runners), then ours, same card, back to back.
TXT
