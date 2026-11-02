#!/bin/bash
# Route B: calibrate the per-image PCMM linears' band shifts on the fake (3 images), merge into a calibration file,
# capture the 12-layer circuit with it, plan, count.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e3
NAME=${1:-vit_sel_box4_pcmm}
cd $REPO
export SEL_ROUTE=pcmm
echo "=== calibrate ==="
for i in 0 1 2; do
    PF_PCMM_CALIB=$D/calib_xpt_$i.json $PY $S/e3_test_xpt.py --layers 12 --box 4 --image $i > $D/logs/calib_$i.log 2>&1 &
done
wait
grep -E '\[xpt\] (logits|done)' $D/logs/calib_*.log
$PY - <<'PY'
import json, glob, os
D = os.path.expanduser("~/data/selective-fhe/e3")
base = json.load(open("bootstrap_placements/vit_b16_224/pcmm_calib.json"))
mx = {}
for f in glob.glob(f"{D}/calib_xpt_*.json"):
    for k, v in json.load(open(f)).items():
        if k.endswith(".xqk") or k.endswith(".xpv"):
            mx[k] = max(mx.get(k, 0.0), v)
merged = {**base, **mx}
json.dump(merged, open(f"{D}/pcmm_calib_sel.json", "w"), indent=1, sort_keys=True)
print("merged calibration:", len(merged), "linears;", {k: round(v, 3) for k, v in sorted(mx.items())[:6]}, "...")
PY
export PF_PCMM_CALIB_IN=$D/pcmm_calib_sel.json
echo "=== capture + plan + count $NAME ==="
bash $S/e3_pipeline.sh $NAME --box 4
cp $D/pcmm_calib_sel.json bootstrap_placements/selective_e3/$NAME/pcmm_calib.json
echo "=== route B pipeline done ==="
