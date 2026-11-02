#!/bin/bash
# E3: capture the selective ViT circuit (12 layers, fake session, oracle check), plan it, count its ops.
#   bash selective/e3_pipeline.sh <name> [extra e3_test_xpt.py args]
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
NAME=${1:?name}; shift
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe/e3
cd $REPO
rm -rf $D/graphs/$NAME bootstrap_placements/selective_e3/$NAME
echo "=== capture $NAME ==="
$PY selective/e3_test_xpt.py --layers 12 --capture $D/graphs/$NAME "$@" > $D/logs/xpt_${NAME}_capture.log 2>&1 || { echo "capture failed"; tail -5 $D/logs/xpt_${NAME}_capture.log; exit 1; }
grep -E '\[xpt\] (logits|done)' $D/logs/xpt_${NAME}_capture.log
echo "=== plan $NAME ==="
bash selective/e3_plan.sh $D/graphs/$NAME $NAME | tail -2
echo "=== counts $NAME ==="
$PY - "$D/graphs/$NAME" <<'PY'
import json, collections, sys
d = sys.argv[1]
tot, steps = collections.Counter(), collections.Counter()
for b in range(12):
    for n in json.load(open(f"{d}/block_{b}/graph.json"))["nodes"]:
        tot[n["op_type"]] += 1; steps[n["step"].split(".")[-1]] += 1
print("nodes", sum(tot.values()), "| mult", tot["mult"], "square", tot["square"], "rotate", tot["rotate"],
      "| bts recorded", tot["auto_bootstrap"] + tot["deliberate_bootstrap"])
print("steps:", {k: steps[k] for k in ["qkt", "softmax", "softmax_v", "gelu", "ln_1", "ln_2", "qkv", "up", "down", "out_proj", "embed", "pkv"]})
PY
echo "=== done $NAME ==="
