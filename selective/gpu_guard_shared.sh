# gpu_guard.sh for a second queue sharing a card with our own jobs: waits only while ANOTHER USER's process is on the
# card (our own runs may share it; their timings are contended). Otherwise identical to gpu_guard.sh.
# Sourced helper: guarded GPU runs on one granted card. Needs GPU_PHYS (physical index the launcher made visible) and
# O (output dir). run NAME CMD...: waits while another user's process is on the card, runs CMD to $O/NAME.log, kills it
# at once if it shows up on any other card or if the host's available memory drops under MEM_FLOOR_GB (default 150; the
# machine is shared), notes contention, appends EXIT_CODE.
GPU_PHYS=${GPU_PHYS:?physical GPU index}
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_PHYS '$1==g {print $2}')
[ -n "$UUID" ] || { echo "no GPU $GPU_PHYS"; exit 1; }
mkdir -p "$O"
state() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' -v g=$GPU_PHYS '$1==g'; }
apps() { nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null; }
LOCKDIR=/home/lzhou/data/selective-fhe/locks
mkdir -p $LOCKDIR
HOLD_PY=/home/lzhou/data/venvs/perseus/bin/python
run() {
    local name=$1; shift
    # one start at a time per card (flock), and a small CUDA holder for the whole run: a job in its CPU setup shows no
    # GPU process, so without the holder another guarded job could start on the card (seen 2026-10-30)
    exec 9>$LOCKDIR/gpu$GPU_PHYS.lock
    flock 9
    while apps | grep "$UUID" | cut -d, -f1 | while read q; do u=$(ps -o user= -p $q 2>/dev/null); [ -n "$u" ] && [ "$u" != "$USER" ] && echo $q; done | grep -q .; do echo "[$(date +%H:%M:%S)] GPU $GPU_PHYS busy (another user's process), waiting"; sleep 60; done
    $HOLD_PY -c "import os, time, torch; p = os.getppid(); torch.zeros(1, device='cuda')
while os.getppid() == p: time.sleep(5)" > /dev/null 2>&1 &         # exits when the queue's shell does
    local H=$!
    while kill -0 $H 2>/dev/null && ! apps | awk -F', ' -v h=$H '$1==h' | grep -q .; do sleep 1; done
    flock -u 9
    exec 9>&-
    echo "=== $name start $(date +%H:%M:%S) | GPU $GPU_PHYS $(state) ==="
    "$@" > $O/$name.log 2>&1 &
    local P=$! seen=0
    while kill -0 $P 2>/dev/null; do
        local mine=$(apps | awk -F', ' -v p=$P '$1==p {print $2}')
        if [ -n "$mine" ] && [ "$mine" != "$UUID" ]; then
            kill $P; echo "GUARD: process $P appeared on $mine, not GPU $GPU_PHYS; killed" | tee -a $O/$name.log
        fi
        [ -n "$mine" ] && seen=1
        local avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
        if [ "$avail" -lt "${MEM_FLOOR_GB:-150}" ]; then
            pkill -P $P; kill $P
            echo "GUARD: host memory available ${avail} GB < ${MEM_FLOOR_GB:-150} GB; killed $name and its workers" | tee -a $O/$name.log
        fi
        local others=$(apps | grep "$UUID" | awk -F', ' -v p=$P -v h=$H '$1!=p && $1!=h' | wc -l)
        [ "$others" -gt 0 ] && echo "[$(date +%H:%M:%S)] note: another process is on GPU $GPU_PHYS during $name (timing contended)" | tee -a $O/$name.log
        sleep 5
    done
    wait $P; local rc=$?
    kill $H 2>/dev/null; wait $H 2>/dev/null
    echo "EXIT_CODE=$rc" >> $O/$name.log
    echo "=== $name end $(date +%H:%M:%S) rc=$rc on-target=$seen | GPU $GPU_PHYS $(state) ==="
    grep -E 'SUMMARY' $O/$name.log | tail -1
}
