#!/bin/bash
# Stop gpu_morning3.sh (tmux m3) once its first run (BERT E1b 872) has ended: its later items run on other cards.
L=/home/lzhou/data/selective-fhe/night/queue_m3.log
until grep -q "sel_g12_ft_872 end" $L; do sleep 2; done
QPID=$(pgrep -u "$USER" -f "^bash selective/gpu_morning3.sh")
C=$(pgrep -P "$QPID" 2>/dev/null)
tmux kill-session -t m3 2>/dev/null
[ -n "$C" ] && kill $C 2>/dev/null
echo "m3 stopped $(date +%H:%M:%S) (killed child: ${C:-none})" >> $L
