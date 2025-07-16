#!/usr/bin/env bash

ROOT_PATH="../"

ID=$1
METHOD=$2
cuda=$3

task='translation_multi_simple_epoch'
USER_DIR=""
if [ $METHOD == 'zero' ];then
  task=$task"_"$METHOD
  USER_DIR="--user-dir models/ZeroTrans "
fi
echo $USER_DIR

mkdir $ROOT_PATH/opus_scripts/results/$METHOD/${ID}
for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
  TOK="13a"
  if [ $lang == 'zh' ];then
    TOK="zh"
  fi
  if [ $lang == 'ja' ];then
    TOK="ja-mecab"
  fi
  if [ $lang == 'ko' ];then
    TOK="ko-mecab"
  fi

  # many-to-en
  tgt_file=$lang"-en.raw.txt"
  CUDA_VISIBLE_DEVICES=$cuda fairseq-generate $ROOT_PATH/opus_100-bin/ --gen-subset test \
  $USER_DIR\
  -s $lang -t en \
  --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
  --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
  --path $ROOT_PATH/opus_scripts/checkpoints/$ID/checkpoint_best.pt \
  --remove-bpe sentencepiece \
  --task $task \
  --max-source-positions 256 --max-target-positions 256 \
  --skip-invalid-size-inputs-valid-test \
  --encoder-langtok tgt \
  --beam 4 > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file
  # hypothesis
  cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$lang"-en.h"
  # reference
  cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$lang"-en.r"
  rm $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file
  echo $lang"-en" >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"
  sacrebleu $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$lang"-en.h" -w 4 -tok 13a < $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$lang"-en.r" >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"

  # en-to-many
  tgt_file="en-"$lang".raw.txt"
  CUDA_VISIBLE_DEVICES=$cuda fairseq-generate $ROOT_PATH/opus_100-bin/ --gen-subset test \
  $USER_DIR\
  -s en -t $lang \
  --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
  --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
  --path $ROOT_PATH/opus_scripts/checkpoints/$ID/checkpoint_best.pt \
  --remove-bpe sentencepiece \
  --task $task \
  --max-source-positions 256 --max-target-positions 256 \
  --skip-invalid-size-inputs-valid-test \
  --encoder-langtok tgt \
  --beam 4 > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file

  # hypothesis
  cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/"en-"$lang".h"
  # reference
  cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/"en-"$lang".r"
  rm $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file
  echo "en-"$lang >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"
  sacrebleu $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/"en-"$lang".h" -w 4 -tok $TOK < $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/"en-"$lang".r" >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"
done


for src in de nl ar fr zh ru; do
  for tgt in de nl ar fr zh ru; do
    if [[ $src != $tgt ]];then
      TOK="13a"
      if [ $tgt == 'zh' ];then
        TOK="zh"
      fi
      tgt_file=$src"-"$tgt".raw.txt"
      CUDA_VISIBLE_DEVICES=$cuda fairseq-generate $ROOT_PATH/opus_100-bin/ --gen-subset test \
      $USER_DIR\
      -s $src -t $tgt \
      --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
      --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
      --path $ROOT_PATH/opus_scripts/checkpoints/$ID/checkpoint_best.pt \
      --remove-bpe sentencepiece \
      --task $task \
      --max-source-positions 256 --max-target-positions 256 \
      --skip-invalid-size-inputs-valid-test \
      --encoder-langtok tgt \
      --beam 4 > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file
      # hypothesis
      cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$src"-"$tgt".h"
      # reference
      cat $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$src"-"$tgt".r"
      rm $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$tgt_file
      echo $src"-"$tgt >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"
      sacrebleu $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$src"-"$tgt".h" -w 4 -tok $TOK < $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$src"-"$tgt".r" >> $ROOT_PATH/opus_scripts/results/$METHOD/${ID}/$ID".sacrebleu"
    fi
  done
done

python $ROOT_PATH/opus_scripts/evaluation/opus_bertscore.py $METHOD $ID $cuda
