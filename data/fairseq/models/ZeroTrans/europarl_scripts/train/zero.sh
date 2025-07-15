#!/bin/bash

ROOT_PATH="../"
cd $ROOT_PATH/fairseq

ID=1
METHOD="zero"
SAVE_PATH=$ROOT_PATH/europarl_scripts/checkpoints/$METHOD/
RESULTS_PATH=$ROOT_PATH/europarl_scripts/results/$METHOD/
LOG_PATH=$ROOT_PATH/europarl_scripts/logs/$METHOD/

mkdir $SAVE_PATH
mkdir $LOG_PATH
mkdir $RESULTS_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $ROOT_PATH/europarl_15-bin/ \
--user-dir models/ZeroTrans \
--seed 0 --fp16 --ddp-backend=no_c10d --arch transformer_zero --task translation_multi_simple_epoch_zero \
--sampling-method "temperature" --sampling-temperature 5 \
--langs "en,de,nl,da,es,pt,ro,it,sl,pl,cs,bg,fi,hu,et" \
--lang-pairs "de-en,en-de,nl-en,en-nl,da-en,en-da,es-en,en-es,pt-en,en-pt,ro-en,en-ro,it-en,en-it,sl-en,en-sl,pl-en,en-pl,cs-en,en-cs,bg-en,en-bg,fi-en,en-fi,hu-en,en-hu,et-en,en-et" \
--encoder-langtok tgt \
--criterion label_smoothed_cross_entropy_zero --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000 --max-epoch 60 --max-tokens 8000 \
--share-all-embeddings --weight-decay 0.0001 \
--no-epoch-checkpoints --no-progress-bar \
--keep-best-checkpoints 5 --log-interval 1000 --log-format simple \
--language-num 15 \
--lse True --lse-dim 128 --lse-position 5 \
--contrastive-learning True --negative-sampling-number 30 --dec-dim 64 --contrastive-position 2 \
--save-dir ${SAVE_PATH}/${ID} > ${LOG_PATH}/${ID}.log

cd $ROOT_PATH/europarl_scripts

bash $ROOT_PATH/europarl_scripts/evaluation/europarl_inference.sh $METHOD $ID