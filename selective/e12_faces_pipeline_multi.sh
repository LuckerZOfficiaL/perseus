#!/bin/bash
# E12 faces, robust plan: like e11_faces_pipeline.sh, but the PCMM bands and the planner's magnitudes come from MANY pages:
# the 3 held-out face pages plus NX pages of photos WITHOUT a face carrying random face-like private blobs of the bucket's
# sizes (e11_faces_common.blob_sets; never evaluated). The captures are max-merged per ciphertext name with upstream's
# scripts/utils/merge_mags.py (a plan sizes every refresh from its capture's magnitudes; one page under-sized refreshes
# for extreme photos, E12 page 22), then planned. XPT_PAD=2; VIT_CONFIGS from the environment.
#   VIT_CONFIGS=... NX=10 bash selective/e12_faces_pipeline_multi.sh NAME K_MIN K_MAX CAP G CHUNK0 HELDOUT
# WEIGHTS=DIR: another HF ViT checkpoint (default: the shipped model); SFHE_FACES=ages: E15's age photos (e11_faces_common).
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
NAME=${1:?}; KMIN=${2:?}; KMAX=${3:?}; CAP=${4:?}; G=${5:?}; C0=${6:?}; HO=${7:?}
NX=${NX:-10}; PAR=${PAR:-4}
PY=/home/lzhou/data/venvs/perseus/bin/python
S=$REPO/selective
D=/home/lzhou/data/selective-fhe/e11
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 SEL_ROUTE=pcmm XPT_PAD=${XPT_PAD:-2}
mkdir -p $D/logs $D/graphs
cd $REPO
ARGS="--k-min $KMIN --k-max $KMAX --cap $CAP --chunk0 $C0 --n $G --layers 12${WEIGHTS:+ --weights $WEIGHTS}"
echo "=== calibrate $NAME: 3 held-out face pages + $NX blob pages ==="
rm -f $D/calib_${NAME}_*.json
{ for i in 0 1 2; do echo "h $i"; done; for x in $(seq 1 $NX); do echo "b $x"; done; } | xargs -P $PAR -L 1 bash -c '
    if [ "$0" = h ]; then A="--tail --first $(($1 * '$G' % '$HO'))"; else A="--blob-seed $1"; fi
    PF_PCMM_CALIB='$D'/calib_'$NAME'_$0$1.json '$PY' '$S'/e11_test_faces.py '"$ARGS"' $A > '$D'/logs/calib_'$NAME'_$0$1.log 2>&1'
$PY - $NAME $D <<'PY'
import json, glob, math, sys
name, D = sys.argv[1], sys.argv[2]
mx = dict(json.load(open("bootstrap_placements/vit_b16_224/pcmm_calib.json")))
fs = glob.glob(f"{D}/calib_{name}_*.json")
for f in fs:
    for k, v in json.load(open(f)).items():
        mx[k] = max(mx.get(k, 0.0), v)
bad = {k: v for k, v in mx.items() if not math.isfinite(v) or v > 1e4}
if bad:
    sys.exit(f"calibration magnitude out of range (a diverged calibration run?): {bad}")
json.dump(mx, open(f"{D}/pcmm_calib_{name}.json", "w"), indent=1, sort_keys=True)
print(f"merged calibration over {len(fs)} pages:", len(mx), "linears")
PY
[ $? -eq 0 ] || exit 1
export PF_PCMM_CALIB_IN=$D/pcmm_calib_$NAME.json
echo "=== capture $NAME: base page + $NX blob pages ==="
rm -rf $D/graphs/$NAME $D/graphs/${NAME}_x* $D/graphs/${NAME}_merged bootstrap_placements/selective_e3/$NAME
{ echo base; for x in $(seq 1 $NX); do echo $x; done; } | xargs -P $PAR -I{} bash -c '
    if [ {} = base ]; then A="--tail --first 0 --grow-first"; G='$D'/graphs/'$NAME'; else A="--blob-seed {} --grow-first"; G='$D'/graphs/'$NAME'_x{}; fi
    '$PY' '$S'/e11_test_faces.py '"$ARGS"' $A --capture $G > '$D'/logs/capture_'$NAME'_{}.log 2>&1 || echo "capture {} failed"'
$PY scripts/utils/merge_mags.py $D/graphs/$NAME $D/graphs/${NAME}_merged $(for x in $(seq 1 $NX); do echo $D/graphs/${NAME}_x$x; done) \
    | tail -2
echo "=== plan $NAME (merged magnitudes) ==="
bash $S/e3_plan.sh $D/graphs/${NAME}_merged $NAME | tail -2
cp $D/pcmm_calib_$NAME.json bootstrap_placements/selective_e3/$NAME/pcmm_calib.json
echo "weights=${WEIGHTS:-shipped} faces=${SFHE_FACES:-imagenette} multi=$NX blob pages" >> bootstrap_placements/selective_e3/$NAME/PLAN_CMD.txt
echo "=== done $NAME ==="
