#!/bin/bash
# Hand a card from a running queue to a new job list: wait until MARKER appears in LOG (the old queue's current job ended),
# kill the old queue's next child (its tmux session and any python it just started), then run the new queue here.
#   tmux new-session -d -s NEW "bash selective/queue_switch.sh OLD_SESSION 'faces_s_orig end' LOG -- <env and command>"
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
OLD=$1; MARKER=$2; LOG=$3; shift 4
QPID=$(tmux list-panes -t "$OLD" -F '#{pane_pid}' 2>/dev/null)
until grep -q "$MARKER" "$LOG" 2>/dev/null; do sleep 3; done
kids() { local p; for p in $(pgrep -P "$1"); do echo $p; kids $p; done; }
K=$(kids "$QPID")
tmux kill-session -t "$OLD" 2>/dev/null
[ -n "$K" ] && kill $K 2>/dev/null
sleep 5
echo "[switch] $(date +%H:%M:%S) $OLD stopped after '$MARKER' (killed: ${K:-none})"
cd "$REPO"
exec env "$@"
