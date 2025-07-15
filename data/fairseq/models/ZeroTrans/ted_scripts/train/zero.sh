#!/bin/bash

ROOT_PATH="../"
cd $ROOT_PATH/fairseq

ID=1
METHOD="zero"
SAVE_PATH=$ROOT_PATH/ted_scripts/checkpoints/$METHOD/
RESULTS_PATH=$ROOT_PATH/ted_scripts/results/$METHOD/
LOG_PATH=$ROOT_PATH/ted_scripts/logs/$METHOD/

mkdir $SAVE_PATH
mkdir $LOG_PATH
mkdir $RESULTS_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train $ROOT_PATH/ted_19-bin/ \
--user-dir models/ZeroTrans \
--seed 0 --fp16 --ddp-backend=no_c10d --arch transformer_zero --task translation_multi_simple_epoch_zero \
--sampling-method "temperature" --sampling-temperature 5 \
--langs "en,ar,he,ru,ko,it,ja,zh,es,nl,vi,tr,fr,pl,ro,fa,hr,cs,de" \
--lang-pairs "ar-en,en-ar,he-en,en-he,ru-en,en-ru,ko-en,en-ko,it-en,en-it,ja-en,en-ja,zh-en,en-zh,es-en,en-es,nl-en,en-nl,vi-en,en-vi,tr-en,en-tr,fr-en,en-fr,pl-en,en-pl,ro-en,en-ro,fa-en,en-fa,hr-en,en-hr,cs-en,en-cs,de-en,en-de" \
--encoder-langtok tgt \
--criterion label_smoothed_cross_entropy_zero --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --max-epoch 30 --max-tokens 4000 \
--share-all-embeddings --weight-decay 0.0001 \
--no-epoch-checkpoints --no-progress-bar \
--keep-best-checkpoints 5 --log-interval 1000 --log-format simple \
--language-num 19 \
--lse True --lse-dim 128 --lse-position 5 \
--contrastive-learning True --negative-sampling-number 30 --dec-dim 64 --contrastive-position 2 \
--save-dir ${SAVE_PATH}/${ID} > ${LOG_PATH}/${ID}.log

cd $ROOT_PATH/ted_scripts

bash $ROOT_PATH/ted_scripts/evaluation/ted_inference.sh $METHOD $ID