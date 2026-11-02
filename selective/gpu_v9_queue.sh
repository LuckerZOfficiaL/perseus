#!/bin/bash
# E12 / E13 (paper runs on THIS upstream version; v3 adds hatex_*, hatex24_*, text24_ner_*; v4: the ft HateXplain plans, enron_*; v5: enron at 4 per page; v6: each Enron model with its own range config; v7: Enron 3 per page, cap 29; v8: hatexval24_* = HateXplain's val split, a second held-out set; v9: *m = the multi-page plans (e13_text_pipeline_multi.sh)): a card's job list, each job waiting for its plan. Pools are serialised
# host-wide (pool.lock) and runs are killed under MEM_FLOOR_GB (gpu_guard.sh). Two-stream runs use XPT_PAD=2 (padding lanes: their own group + the last input's plaintext keys).
#   tmux new-session -d -s v2_q6 "CUDA_VISIBLE_DEVICES=<card> GPU_PHYS=<card> JOBS='faces_s_orig faces_s_ft' AFTER=<tmux> \
#       bash selective/gpu_v2_queue.sh"
# Jobs: faces_{s,m,l}_{orig,ft} (two-stream, private = detected faces; shipped ViT config), faces3_* (the same with the
# ViT LayerNorm bands refitted on photos, e12_vit_ln_bands.py), faces4_* (+ softmax refit), faces5_* (+ multi-page
# magnitudes, e12_faces_pipeline_multi.sh), late{1,2}_faces (late-inject-replace, E5b weights,
# the small-face photos), text_{ner,span}_{orig,ft} (two-stream BERT, 12 per page).
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
set -u
cd $REPO
PY=/home/lzhou/data/venvs/perseus/bin/python
D=/home/lzhou/data/selective-fhe
O=$D/e12
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MEM_FLOOR_GB=150 SEL_ROUTE=pcmm
source selective/gpu_guard.sh
while tmux has-session -t "${AFTER:-none}" 2>/dev/null; do sleep 20; done
P3=bootstrap_placements/selective_e3
P8=bootstrap_placements/selective_e8
waitplan() { while [ ! -f "$1/pcmm_calib.json" ]; do sleep 30; done; }
faces() {   # bucket weights [version: v2 = shipped ViT config, v3 = LayerNorm bands refitted on photos]
    local b=$1 w=$2 ver=${3:-v2} args
    case $b in
        s) args="--k-min 1 --k-max 20 --cap 21 --inputs 8 --chunk0 4 --heldout 24" ;;
        m) args="--k-min 21 --k-max 31 --cap 32 --inputs 4 --chunk0 2 --heldout 12" ;;
        l) args="--k-min 32 --k-max 63 --cap 64 --inputs 2 --chunk0 1 --heldout 6" ;;
    esac
    local P=$P3/vit_faces_${b}_$ver; waitplan $P
    local WA=""; [ $w = ft ] && WA="--weights $D/e1b/vit_oneway"
    local VC=""; [ $ver = v3 ] && VC="VIT_CONFIGS=$REPO/selective/sfhe/configs/vit_base_224_photos/configs.json"
    [ $ver = v4 ] || [ $ver = v5 ] && VC="VIT_CONFIGS=$REPO/selective/sfhe/configs/vit_base_224_photos2/configs.json"
    local tag=faces; [ $ver = v3 ] && tag=faces3; [ $ver = v4 ] && tag=faces4; [ $ver = v5 ] && tag=faces5
    run ${tag}_${b}_$w env XPT_PAD=2 $VC PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e11_run_faces_gpu.py --plan $P $args \
        $WA --out $O/${tag}_${b}_$w.json
}
late() {    # k [version: v2 = shipped ViT config, v4 = LayerNorm + softmax refitted on photos]
    local ver=${2:-v2}
    local P=$P3/vit_late_k$1_$ver; waitplan $P
    local VC=""; [ $ver = v4 ] && VC="VIT_CONFIGS=$REPO/selective/sfhe/configs/vit_base_224_photos2/configs.json"
    local tag=late; [ $ver = v4 ] && tag=late4
    run ${tag}$1_faces env $VC PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e6_run_gpu.py --plan $P --k $1 --faces 1,20 \
        --heldout 24 --n 1000 --direct-handover --weights $D/e5b/vit_late --out $O/${tag}$1_faces.json
}
text() {    # ner|span weights [version: v2 = LayerNorm refit, v3 = LayerNorm + softmax refits]
    local ver=${3:-v2}
    local P=$P8/bert_sel_g12c15_$ver; waitplan $P
    local CF=$REPO/selective/sfhe/configs/bert_base_n32_t128_sst2/configs.json
    [ $ver = v3 ] && CF=$REPO/selective/sfhe/configs/bert_base_n32_t128_sst2b/configs.json
    local tag=text; [ $ver = v3 ] && tag=text3
    local WA=""; [ $2 = ft ] && WA="--weights $D/exports/bert_oneway/classic"
    local PR=""; [ $1 = ner ] && PR="--private $D/e11/ner_private.json"
    run ${tag}_$1_$2 env XPT_PAD=2 BERT_CONFIGS=$CF \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 12 --chunk0 6 --n-e-cap 15 \
        --idx 0:872 --workers 4 $PR $WA --out $O/${tag}_$1_$2.json
}
hatex() {   # weights [G: inputs per page, 12 or 24] [split: "" = test, _val = val] -- E13: HateXplain posts, private = identity-group mentions
    local w=$1 G=${2:-12} V=${3:-} E=$D/e13
    local P=$P8/bert_hatex_g${G}c15; [ $w = ft ] && P=$P8/bert_hatexft_g${G}c15; waitplan $P
    local WA="--weights $D/exports/hatexplain/classic"; [ $w = ft ] && WA="--weights $D/exports/hatexplain_oneway/classic"
    local tag=hatex; [ $G = 24 ] && tag=hatex24; [ -n "$V" ] && tag=hatexval$G
    run ${tag}_$w env XPT_PAD=2 SFHE_TEXT_DATA=$E/hatexplain$V.json \
        BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_hatexb/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs $G --chunk0 $((G / 2)) \
        --n-e-cap 15 --idx 0:1924 --workers 4 --private $E/hatexplain${V}_private.json $WA --out $O/${tag}_$w.json
}
ner24() {   # weights -- SST-2 NER, 24 sentences per page
    local P=$P8/bert_sel_g24c15_v3; waitplan $P
    local WA=""; [ $1 = ft ] && WA="--weights $D/exports/bert_oneway/classic"
    run text24_ner_$1 env XPT_PAD=2 BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_sst2b/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 24 --chunk0 12 --n-e-cap 15 \
        --idx 0:872 --workers 4 --private $D/e11/ner_private.json $WA --out $O/text24_ner_$1.json
}
enron() {   # weights -- E14: Enron e-mails at 256 tokens, 4 per page (8 per page needs 104.6 GB of device memory), private = personal data (names, places, organisations, contacts)
    local w=$1 E=$D/e14
    local P=$P8/bert_enron_t256_g3c29; [ $w = ft ] && P=$P8/bert_enronft_t256_g3c29; waitplan $P
    local WA="--weights $D/exports/enron_bert/classic"; [ $w = ft ] && WA="--weights $D/exports/enron_oneway/classic"
    local CF=enronb; [ $w = ft ] && CF=enronftb                      # the config the plan was built with
    run enron_$w env XPT_PAD=2 XPT_MB=4 SFHE_BERT_T=256 SFHE_TEXT_DATA=$E/enron_t256.json \
        BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t256_$CF/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 3 --chunk0 2 \
        --n-e-cap 29 --idx 0:1000 --workers 4 --private $E/enron_t256_private.json $WA --out $O/enron_$w.json
}
enron_full() {   # E14 baseline: upstream's full BERT at 256 tokens on 6 test e-mails with 256 real tokens (eager, as ours)
    local P=bootstrap_placements/selective_e14/enron_full_t256; waitplan $P
    run enron_full_t256 env SFHE_PATCHES=1 PYTHONPATH="$REPO/selective/sfhe_site:$REPO/selective:$REPO/python" \
        SFHE_TEXT_DATA=$D/e14/enron_t256.json BERT_EXPORT=$D/exports/enron_bert/classic \
        $PY -m perseus.examples.bert.run_bert --plan $P --tokens 256 --text "" --sst2 ${ENRON_FULL_IDX:-2,6,14,15,16,19}
}
multi() {   # task weights [split] -- 24 per page on the multi-page plans: task ner (SST-2 names) | hatex (HateXplain)
    local task=$1 w=$2 V=${3:-} P CF DATA PR WA tag
    if [ $task = ner ]; then
        P=$P8/bert_sel_g24m_v3; [ $w = ft ] && P=$P8/bert_selft_g24m_v3
        CF=sst2b; DATA=""; PR=$D/e11/ner_private.json; IDX=0:872; tag=text24m_ner_$w
        WA=""; [ $w = ft ] && WA="--weights $D/exports/bert_oneway/classic"
    else
        P=$P8/bert_hatex_g24m; [ $w = ft ] && P=$P8/bert_hatexft_g24m
        CF=hatexb; DATA="SFHE_TEXT_DATA=$D/e13/hatexplain$V.json"; PR=$D/e13/hatexplain${V}_private.json; IDX=0:1924
        tag=hatex24m_$w; [ -n "$V" ] && tag=hatexval24m_$w
        WA="--weights $D/exports/hatexplain/classic"; [ $w = ft ] && WA="--weights $D/exports/hatexplain_oneway/classic"
    fi
    waitplan $P
    run $tag env XPT_PAD=2 $DATA BERT_CONFIGS=$REPO/selective/sfhe/configs/bert_base_n32_t128_$CF/configs.json \
        PF_PCMM_CALIB_IN=$P/pcmm_calib.json $PY selective/e8_run_gpu.py --plan $P --inputs 24 --chunk0 12 \
        --n-e-cap 15 --idx $IDX --workers 4 --private $PR $WA --out $O/$tag.json
}
for job in $JOBS; do
    case $job in
        faces_*) set -- ${job//_/ }; faces $2 $3 v2 ;;
        faces3_*) set -- ${job//_/ }; faces $2 $3 v3 ;;
        faces4_*) set -- ${job//_/ }; faces $2 $3 v4 ;;
        faces5_*) set -- ${job//_/ }; faces $2 $3 v5 ;;
        late4*_faces) k=${job#late4}; late ${k%_faces} v4 ;;
        late*_faces) k=${job#late}; late ${k%_faces} ;;
        text3_*) set -- ${job//_/ }; text $2 $3 v3 ;;
        hatex_*) hatex ${job#hatex_} 12 ;;
        hatex24_*) hatex ${job#hatex24_} 24 ;;
        hatexval24_*) hatex ${job#hatexval24_} 24 _val ;;
        text24m_ner_*) multi ner ${job#text24m_ner_} ;;
        hatex24m_*) multi hatex ${job#hatex24m_} ;;
        hatexval24m_*) multi hatex ${job#hatexval24m_} _val ;;
        text24_ner_*) ner24 ${job#text24_ner_} ;;
        enron_full) enron_full ;;
        enron_*) enron ${job#enron_} ;;
        text_*) set -- ${job//_/ }; text $2 $3 ;;
        *) echo "unknown job $job" ;;
    esac
done
echo "=== all done $(date +%H:%M:%S) ==="
