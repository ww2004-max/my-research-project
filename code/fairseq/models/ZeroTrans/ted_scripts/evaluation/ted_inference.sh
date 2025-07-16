#!/usr/bin/env bash

# input param
# vanilla;lang_pred;sc;dec
METHOD=$1
ID=$2
cuda=$3

ROOT_PATH="../"
cd ${ROOT_PATH}/fairseq

# average top-5 checkpoints
checkpoints=$(ls ${ROOT_PATH}/ted_scripts/checkpoints/${METHOD}/$ID/checkpoint.best_loss_* | tr '\n' ' ')
python3 ${ROOT_PATH}/fairseq/scripts/average_checkpoints.py \
--inputs $checkpoints \
--output ${ROOT_PATH}/ted_scripts/checkpoints/${METHOD}/${METHOD}/$ID/checkpoint_averaged.pt

task='translation_multi_simple_epoch'
USER_DIR=""
if [ $METHOD == 'zero' ];then
  task=$task"_"$METHOD
  USER_DIR="--user-dir models/ZeroTrans "
fi
echo $USER_DIR

mkdir $ROOT_PATH/ted_scripts/results/$METHOD/${ID}
for src in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
  for tgt in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
    if [[ $src != $tgt ]];then
      tgt_file=$src"-"$tgt".raw.txt"
      CUDA_VISIBLE_DEVICES=$cuda fairseq-generate ${ROOT_PATH}/ted_19-bin/ --gen-subset test \
      $user_dir_arg\
      -s $src -t $tgt \
      --langs "en,ar,he,ru,ko,it,ja,zh,es,nl,vi,tr,fr,pl,ro,fa,hr,cs,de" \
      --lang-pairs "ar-en,en-ar,he-en,en-he,ru-en,en-ru,ko-en,en-ko,it-en,en-it,ja-en,en-ja,zh-en,en-zh,es-en,en-es,nl-en,en-nl,vi-en,en-vi,tr-en,en-tr,fr-en,en-fr,pl-en,en-pl,ro-en,en-ro,fa-en,en-fa,hr-en,en-hr,cs-en,en-cs,de-en,en-de" \
      --path ${ROOT_PATH}/ted_scripts/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt \
      --remove-bpe sentencepiece \
      --task $task \
      --encoder-langtok "tgt" \
      --beam 4 > ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$tgt_file

      cat ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".h"
      # reference
      cat ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".r"
      rm ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$tgt_file

      cat ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".h" | perl $DETOKEN -threads 32 -l $tgt >> ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.h"
      cat ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".r" | perl $DETOKEN -threads 32 -l $tgt >> ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.r"
      rm ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".h"
      rm ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".r"

      echo $src"-"$tgt >> ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$ID".sacrebleu"
      TOK='13a'
      if [ $tgt == 'zh' ];then
         TOK='zh'
      fi
      if [ $tgt == 'ja' ];then
         TOK='ja-mecab'
      fi
      if [ $tgt == 'ko' ];then
         TOK='ko-mecab'
      fi
      sacrebleu ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.h" -w 4 -tok $TOK < ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.r" >> ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$ID".sacrebleu"
      # rm ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.h"
      # rm ${ROOT_PATH}/ted_scripts/results/${METHOD}/${ID}/$src"-"$tgt".detok.r"
    fi
  done
done

python ted_bertscore.py $METHOD $ID $cuda
