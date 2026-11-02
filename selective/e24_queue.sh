# E24 queue on one granted card (GPU_PHYS): LLM LoRA regimes on drugsc / mts, in priority order (1 full-attention
# reference, 2 hiding-aware, 3 private-only + late fusion, 4 one-way seeds), each phase over every model, then the summary.
# Re-runnable: finished adapters and scored regimes are skipped.
cd /home/lzhou/perseus-up/selective; O=/home/lzhou/data/selective-fhe/e24; source gpu_guard.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/lzhou/data/venvs/sat/bin/python
MODELS="meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct"
until grep -q DONE $O/cache.log 2>/dev/null; do sleep 20; done
phase() {   # phase REGIME:SEED ...
  for task in drugsc mts; do for m in $MODELS; do n=$(basename $m)
    for rs in "$@"; do r=${rs%:*}; s=${rs#*:}
      [ -f $O/lora/${task}_${n}_${r}_s$s/adapter_config.json ] || run tr_${task}_${n}_${r}_s$s $PY e24_llm.py --train $r --seed $s --task $task --model $m
    done
    run ev_${task}_${n}_$1 $PY e24_llm.py --eval --task $task --model $m
  done; done
}
phase full:0
phase delete:0 mask:0 typed:0
phase private:0
phase oneway:1 oneway:2
$PY e24_llm.py --summary > $O/summary.txt 2>&1
echo "=== all done $(date +%T) ==="
