#!/bin/bash
# E6: capture the late-inject-replace circuit (hand-over + last k blocks) on the twin, plan it, count it.
#   bash selective/e6_pipeline.sh <k> [extra e6_test_late.py args]
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
K=${1:?k}; shift
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e6
NAME=${NAME:-vit_late_k${K}_box4}
mkdir -p $D/graphs $D/logs
cd $REPO
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e3/$NAME
echo "=== capture $NAME ==="
$PY $S/e6_test_late.py --k $K --box 4 --capture $D/graphs/$NAME "$@" > $D/logs/capture_$NAME.log 2>&1 || { echo "capture failed"; tail -3 $D/logs/capture_$NAME.log; exit 1; }
grep -E '\[late\] (logits|done)' $D/logs/capture_$NAME.log
echo "=== plan $NAME ==="
bash $S/e3_plan.sh $D/graphs/$NAME $NAME | tail -2
echo "=== counts $NAME ==="
$PY - "$D/graphs/$NAME" <<'PY'
import json, collections, sys, glob, os
d = sys.argv[1]
tot, steps = collections.Counter(), collections.Counter()
for f in sorted(glob.glob(os.path.join(d, "block_*/graph.json"))):
    for n in json.load(open(f))["nodes"]:
        tot[n["op_type"]] += 1; steps[n["step"].split(".")[-1]] += 1
print("stages", len(glob.glob(os.path.join(d, "block_*/graph.json"))), "| nodes", sum(tot.values()), "| mult", tot["mult"], "square", tot["square"],
      "rotate", tot["rotate"], "| bts recorded", tot["auto_bootstrap"] + tot["deliberate_bootstrap"])
print("steps:", {k: steps[k] for k in ["embed", "plain", "qkt", "softmax", "softmax_v", "gelu", "ln_1", "ln_2", "qkv", "up", "down", "out_proj"]})
PY
echo "=== done $NAME ==="
