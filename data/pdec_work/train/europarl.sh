 #!/bin/bash

# Europarl dataset training script
# Languages: en, de, es, it
# Language pairs: en-de, de-en, en-es, es-en, en-it, it-en

# binary dataset path
DATA_BIN="C:/Users/33491/PycharmProjects/machine/fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
ROOT_PATH="."
num_gpus=8
FAIR_PATH=${ROOT_PATH}/fairseq
WORK_PATH=${ROOT_PATH}/pdec_work
cd ${FAIR_PATH}

METHOD=europarl_pdec
ID=1

# Model parameters
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

# Create necessary directories
mkdir -p ${WORK_PATH}/checkpoints
mkdir -p ${WORK_PATH}/checkpoints/${METHOD}
mkdir -p ${WORK_PATH}/checkpoints/${METHOD}/${ID}

mkdir -p ${WORK_PATH}/logs
mkdir -p ${WORK_PATH}/logs/${METHOD}

mkdir -p ${WORK_PATH}/results
mkdir -p ${WORK_PATH}/results/${METHOD}
mkdir -p ${WORK_PATH}/results/${METHOD}/${ID}

# Training command
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train ${DATA_BIN} \
--user-dir models/PhasedDecoder/ \
--seed $SEED --fp16 --ddp-backend=no_c10d --arch transformer_pdec_${ENC}_e_${DEC}_d --task translation_multi_simple_epoch \
--sampling-method "temperature" --sampling-temperature 5 \
--attention-position-bias $BIAS \
--adaption-flag $ADAPTION \
--adaption-inner-size $INNER \
--adaption-dropout $DROP \
--contrastive-flag $CONTRASTIVE \
--contrastive-type $TYPE \
--dim $DIM \
--mode $MODE \
--cl-position $POSITION \
--temperature $T \
--langs "en,de,es,it" \
--lang-pairs "en-de,de-en,en-es,es-en,en-it,it-en" \
--encoder-langtok tgt \
--criterion label_smoothed_cross_entropy_instruction --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --max-epoch 30 --max-tokens 4000 \
--share-all-embeddings --weight-decay 0.0001 \
--no-epoch-checkpoints --no-progress-bar \
--keep-best-checkpoints 5 --log-interval 1000 --log-format simple \
--save-dir ${WORK_PATH}/checkpoints/${METHOD}/${ID} > ${WORK_PATH}/logs/${METHOD}/${ID}.log

# Average checkpoints
checkpoints=$(ls ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint.best_loss_* | tr '\n' ' ')
python3 scripts/average_checkpoints.py \
--inputs $checkpoints \
--output ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt

# Switch back to work directory
cd ${WORK_PATH}

# Create europarl evaluation scripts if they don't exist
mkdir -p ${WORK_PATH}/europarl_evaluation

# Run inference
echo "Starting inference..."
bash europarl_evaluation/batch_inference.sh ${METHOD} ${ID} ${ROOT_PATH} ${num_gpus} ${DATA_BIN} en de es it

# Generate results table
echo "Generating results table..."
mkdir -p ${WORK_PATH}/excel
python europarl_evaluation/make_table.py ${METHOD} ${ID} ${ROOT_PATH} en de es it

echo "Training and evaluation completed!"
echo "Results saved in ${WORK_PATH}/checkpoints/${METHOD}/${ID}/"
echo "Logs saved in ${WORK_PATH}/logs/${METHOD}/${ID}.log"
echo "Inference results in ${WORK_PATH}/results/${METHOD}/${ID}/"