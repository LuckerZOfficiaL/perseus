# E27 queue on one granted card (GPU_PHYS): PrivacyRestore baseline per (task, model); the server model is E24's
# full-attention LoRA, retrained here with E24's recipe (E24 deleted its adapters). Re-runnable: finished results skip.
cd /home/lzhou/perseus-up/selective; O=/home/lzhou/data/selective-fhe/e27; source gpu_guard.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/lzhou/data/venvs/sat/bin/python; L=/home/lzhou/data/selective-fhe/e24/lora
for m in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct; do
  n=$(basename $m)
  for task in ${TASKS:-drugsc mts}; do
    [ -f $O/${task}_$n.json ] && continue
    [ -f $L/${task}_${n}_full_s0/adapter_config.json ] || run lora_${task}_$n $PY e24_llm.py --train full --seed 0 --task $task --model $m
    run pr_${task}_$n $PY e27_privacyrestore.py --task $task --model $m
  done
done
echo "=== all done $(date +%T) ==="
