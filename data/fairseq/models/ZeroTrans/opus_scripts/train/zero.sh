#!/bin/bash

ROOT_PATH="../"
cd $ROOT_PATH/fairseq

ID=1
METHOD="zero"
SAVE_PATH=$ROOT_PATH/opus_scripts/checkpoints/$METHOD/
RESULTS_PATH=$ROOT_PATH/opus_scripts/results/$METHOD/
LOG_PATH=$ROOT_PATH/opus_scripts/logs/$METHOD/

mkdir $SAVE_PATH
mkdir $LOG_PATH
mkdir $RESULTS_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3, fairseq-train $ROOT_PATH/opus_100-bin/ \
--user-dir models/ZeroTrans \
--seed 0 --fp16 --ddp-backend=no_c10d --arch transformer_zero_opus --task translation_multi_simple_epoch_zero \
--sampling-method "temperature" --sampling-temperature 5 \
--langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
--lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
--encoder-langtok tgt \
--criterion label_smoothed_cross_entropy_zero --label-smoothing 0.1 \
--optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
--warmup-updates 4000  --max-update 400000 --max-tokens 8000 --dropout 0.1 \
--share-all-embeddings --weight-decay 0.0001 \
--no-epoch-checkpoints --no-progress-bar \
--log-interval 1000 --log-format simple  \
--max-source-positions 256 --max-target-positions 256 \
--skip-invalid-size-inputs-valid-test \
--language-num 95
--lse True --lse-dim 128 --lse-position 5 \
--contrastive-learning True --negative-sampling-number 30 --dec-dim 64 --contrastive-position 1 \
--save-dir ${SAVE_PATH}/${ID} > ${LOG_PATH}/${ID}.log

cd $ROOT_PATH/opus_scripts

bash $ROOT_PATH/opus_scripts/evaluation/opus_inference.sh $METHOD $ID