# E31: the language models' adapters trained for 1000 steps instead of 300 (E30: the one-way adapter's gap to the
# full-attention one came from too few steps, not from a missing distillation loss). Every regime that the paper
# compares gets the same budget: oneway (OWTE), full (the full-attention reference) and delete (the best trained
# hiding baseline). Seed 0 of every regime first (largest models first, so they do not straggle), then OWTE seeds 1, 2.
# Items are claimed once (mkdir) by any worker on any card; Qwen2.5-32B only on cards without a second worker of ours.
cd /home/lzhou/perseus-up/selective; O=/home/lzhou/data/selective-fhe/e31; source gpu_guard_shared.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 E24_OUT=$O
PY=/home/lzhou/data/venvs/sat/bin/python
BIG_OK=${BIG_OK:-1}
mkdir -p $O/claims
item() {   # item MODEL TASK REGIME SEED
  local m=$1 task=$2 r=$3 s=$4 n=$(basename $1)
  [ "$n" = Qwen2.5-32B-Instruct ] && [ "$BIG_OK" = 0 ] && return
  [ -f $O/${task}_${n}_${r}_s$s.json ] && return
  mkdir $O/claims/${task}_${n}_${r}_s$s 2>/dev/null || return
  [ -f $O/lora/${task}_${n}_${r}_s$s/adapter_config.json ] || \
    run tr_${task}_${n}_${r}_s$s $PY e24_llm.py --train $r --steps 1000 --seed $s --task $task --model $m
  run ev_${task}_${n}_${r}_s$s $PY e24_llm.py --eval --task $task --model $m
}
ORDER="Qwen/Qwen2.5-32B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B-Instruct"
for m in $ORDER; do for task in drugsc mts; do for r in oneway full delete; do item $m $task $r 0; done; done; done
for s in 1 2; do for m in $ORDER; do for task in drugsc mts; do item $m $task oneway $s; done; done; done
echo "=== all done $(date +%T) ==="
