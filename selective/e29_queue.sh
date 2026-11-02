# E29 queue for one granted card (GPU_PHYS): distilled one-way LoRA (1000 steps, CE + KL to E24's full-attention
# LoRA). First this card's own seed-0 models (MODELS), then a pool shared by every card: seeds 1 and 2 of all models,
# each item claimed once (mkdir in $O/claims). Teacher adapters: e24/lora/*_full_s0 (trained here if missing).
cd /home/lzhou/perseus-up/selective; O=/home/lzhou/data/selective-fhe/e29; source gpu_guard.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/lzhou/data/venvs/sat/bin/python; L=/home/lzhou/data/selective-fhe/e24/lora
ALL="meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct"
mkdir -p $O/claims
while [ -n "${AFTER:-}" ] && tmux has-session -t "=$AFTER" 2>/dev/null; do sleep 30; done
item() {   # item MODEL TASK SEED
  local m=$1 task=$2 s=$3 n=$(basename $1)
  [ -f $O/${task}_${n}_s$s.json ] && return
  mkdir $O/claims/${task}_${n}_s$s 2>/dev/null || return
  if [ ! -f $L/${task}_${n}_full_s0/adapter_config.json ]; then
    if mkdir $O/claims/teacher_${task}_$n 2>/dev/null; then
      run tlora_${task}_$n $PY e24_llm.py --train full --seed 0 --task $task --model $m
    fi
    until [ -f $L/${task}_${n}_full_s0/adapter_config.json ]; do sleep 60; done
  fi
  [ -f $O/lora/${task}_${n}_s$s/adapter_config.json ] || run tr_${task}_${n}_s$s $PY e29_distill.py --train --task $task --model $m --seed $s
  run ev_${task}_${n}_s$s $PY e29_distill.py --eval --task $task --model $m --seed $s
}
for m in $MODELS; do for task in drugsc mts; do item $m $task 0; done; done
for s in 1 2; do for m in $ALL; do for task in drugsc mts; do item $m $task $s; done; done; done
$PY e29_distill.py --summary > $O/summary_gpu$GPU_PHYS.txt 2>&1
echo "=== all done $(date +%T) ==="
