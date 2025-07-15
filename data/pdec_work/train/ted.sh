#!/bin/bash

# binary dataset path
DATA_BIN="."
ROOT_PATH="."
num_gpus=8
FAIR_PATH=${ROOT_PATH}/fairseq
WORK_PATH=${ROOT_PATH}/pdec_work
cd ${FAIR_PATH}

METHOD=ted_pdec
ID=1

ENC=6
DEC=6
BIAS=1
ADAPTION='True'
DROP=0.1
INNER=2048
CONTRASTIVE='True'
POSITION=6
TYPE='enc'
T=1.0
DIM=512
MODE=1

SEED=0

mkdir ${WORK_PATH}/checkpoints
mkdir ${WORK_PATH}/checkpoints/${METHOD}
mkdir ${WORK_PATH}/checkpoints/${METHOD}/${ID}

mkdir ${WORK_PATH}/logs
mkdir ${WORK_PATH}/logs/${METHOD}

mkdir ${WORK_PATH}/results
mkdir ${WORK_PATH}/results/${METHOD}
mkdir ${WORK_PATH}/results/${METHOD}/${ID}



# =============================
# ✅ 注释掉训练部分
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ${DATA_BIN} \
# --user-dir models/PhasedDecoder/ \
# --seed $SEED --fp16 --ddp-backend=no_c10d --arch transformer_pdec_${ENC}_e_${DEC}_d --task translation_multi_simple_epoch \
# --sampling-method "temperature" --sampling-temperature 5 \
# --attention-position-bias $BIAS \
# --adaption-flag $ADAPTION \
# --adaption-inner-size $INNER \
# --adaption-dropout $DROP \
# --contrastive-flag $CONTRASTIVE \
# --contrastive-type $TYPE \
# --dim $DIM \
# --mode $MODE \
# --cl-position $POSITION \
# --temperature $T \
# --langs "en,de,es,it" \
# --lang-pairs "de-en,en-de,es-en,en-es,it-en,en-it" \
# --encoder-langtok tgt \
# --criterion label_smoothed_cross_entropy_instruction --label-smoothing 0.1 \
# --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
# --warmup-updates 4000 --max-epoch 30 --max-tokens 4000 \
# --share-all-embeddings --weight-decay 0.0001 \
# --no-epoch-checkpoints --no-progress-bar \
# --keep-best-checkpoints 5 --log-interval 1000 --log-format simple \
# --save-dir ${WORK_PATH}/checkpoints/${METHOD}/${ID} > ${WORK_PATH}/logs/${METHOD}/${ID}.log


# checkpoints=$(ls ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint.best_loss_* | tr '\n' ' ')
# python3 scripts/average_checkpoints.py \
# --inputs $checkpoints \
# --output ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt
# =============================


# 切换回工作目录
cd ${WORK_PATH}

# 推理阶段（只包含 en, de, es, it）
bash ted_evaluation/batch_inference.sh ${METHOD} ${ID} ${ROOT_PATH} ${num_gpus} ${DATA_BIN} en de es it

# BERTScore 评估
bash ted_evaluation/batch_bertscore.sh ${METHOD} ${ID} ${WORK_PATH} ${num_gpus} en de es it

# COMET 评估
bash ted_evaluation/batch_comet.sh ${METHOD} ${ID} ${WORK_PATH} ${num_gpus} en de es it

# 表格生成（如果支持多语言参数的话）
mkdir -p ${WORK_PATH}/excel
python ted_evaluation/make_table.py ${METHOD} ${ID} ${ROOT_PATH} en de es it