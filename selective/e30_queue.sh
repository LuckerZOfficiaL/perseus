# E30: control for E29 -- the same 1000-step one-way LoRA with cross-entropy only (KL weight 0), to separate the effect
# of distillation from that of training longer (E24's one-way LoRA: 300 steps). Small Llamas, both tasks, seed 0.
# Shares the card with our own queues (gpu_guard_shared.sh).
cd /home/lzhou/perseus-up/selective; O=/home/lzhou/data/selective-fhe/e30; source gpu_guard_shared.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 E29_OUT=$O
PY=/home/lzhou/data/venvs/sat/bin/python
for m in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct; do for task in drugsc mts; do
  n=$(basename $m)
  [ -f $O/${task}_${n}_s0.json ] && continue
  [ -f $O/lora/${task}_${n}_s0/adapter_config.json ] || run tr_${task}_${n}_s0 $PY e29_distill.py --train --kl 0 --task $task --model $m --seed 0
  run ev_${task}_${n}_s0 $PY e29_distill.py --eval --task $task --model $m --seed 0
done; done
echo "=== all done $(date +%T) ==="
