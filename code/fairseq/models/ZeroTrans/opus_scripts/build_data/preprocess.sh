#!/bin/bash

ROOT_PATH="../"

mkdir $ROOT_PATH/opus_scripts/logs
mkdir $ROOT_PATH/opus_scripts/checkpoints
mkdir $ROOT_PATH/opus_scripts/results

curl -O https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz
mkdir $ROOT_PATH/opus_scripts/build_data/opus_100_data
tar -xzvf opus-100-corpus-v1.0.tar.gz -C $ROOT_PATH/opus_scripts/build_data/opus_100_data
rm opus-100-corpus-v1.0.tar.gz
python $ROOT_PATH/opus_scripts/build_data/de_duplicate.py $ROOT_PATH/opus_scripts/build_data/opus_100_data

mkdir $ROOT_PATH/opus_scripts/build_data/raw

for lang in af am ar as az be bg bn br bs ca cs cy da de el; do
    echo sort files
    mkdir $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/valid.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/valid.en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/test.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/test.en
done

for lang in eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    echo sort files
    mkdir $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/valid.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/valid.en
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.${lang} $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/test.${lang}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.en $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/test.en
done

for lpair in de-nl nl-zh ar-nl ru-zh fr-nl de-fr fr-zh ar-ru ar-zh ar-fr de-zh fr-ru de-ru nl-ru ar-de; do
    TMP=(${lpair//-/ })
    SRC=${TMP[0]}
    TGT=${TMP[1]}
    mkdir $ROOT_PATH/opus_scripts/build_data/raw/${SRC}_${TGT}
    echo sort zero-shot files
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-test.${SRC} $ROOT_PATH/opus_scripts/build_data/raw/${SRC}_${TGT}/test.${SRC}
    cp $ROOT_PATH/opus_scripts/build_data/opus_100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-test.${TGT} $ROOT_PATH/opus_scripts/build_data/raw/${SRC}_${TGT}/test.${TGT}
done

BPE_FILES=""
for lang in ar zh nl fr de ru; do
    BPE_FILES=${BPE_FILES}","$ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.en
done

for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    BPE_FILES=${BPE_FILES}","$ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/train.${lang}
done
python $SPM_TRAIN --input=$BPE_FILES --model_prefix=$ROOT_PATH/opus_scripts/build_data/spm_64k --vocab_size=64000 --character_coverage=1.0 --input_sentence_size=10000000

BPE_PATH=$ROOT_PATH/opus_scripts/build_data/bpe
mkdir $BPE_PATH

BINARY_PATH=$ROOT_PATH/opus_100-bin
mkdir $BINARY_PATH
cut -f 1 $ROOT_PATH/opus_scripts/build_data/spm_64k.vocab | tail -n +4 | sed "s/$/ 1/g" > ${BINARY_PATH}/dict.txt


for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    mkdir $BPE_PATH/${lang}_en
    for split in train valid test; do
        python $SPM_ENCODE --model $ROOT_PATH/opus_scripts/build_data/spm_64k.model --output_format=piece \
        --inputs $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/${split}.${lang} \
        --outputs ${BPE_PATH}/${lang}_en/${split}.${lang}
        python $SPM_ENCODE --model $ROOT_PATH/opus_scripts/build_data/spm_64k.model --output_format=piece \
        --inputs $ROOT_PATH/opus_scripts/build_data/raw/${lang}_en/${split}.en \
        --outputs ${BPE_PATH}/${lang}_en/${split}.en
    done
    
    fairseq-preprocess --task "translation" --source-lang en --target-lang ${lang} \
      --trainpref ${BPE_PATH}/${lang}_en/train \
      --validpref ${BPE_PATH}/${lang}_en/valid \
      --testpref ${BPE_PATH}/${lang}_en/test \
      --destdir ${BINARY_PATH} --padding-factor 1 --workers 16 \
      --srcdict ${BINARY_PATH}/dict.txt \
      --tgtdict ${BINARY_PATH}/dict.txt
done

for lpair in de-nl nl-zh ar-nl ru-zh fr-nl de-fr fr-zh ar-ru ar-zh ar-fr de-zh fr-ru de-ru nl-ru ar-de; do
    TMP=(${lpair//-/ })
    SRC=${TMP[0]}
    TGT=${TMP[1]}
    mkdir ${BPE_PATH}/${SRC}_${TGT}
    python $SPM_ENCODE --model $ROOT_PATH/opus_scripts/build_data/spm_64k.model --output_format=piece \
        --inputs $ROOT_PATH/opus_scripts/build_data/raw/${SRC}_${TGT}/test.${SRC} \
        --outputs ${BPE_PATH}/${SRC}_${TGT}/test.${SRC}
    python $SPM_ENCODE --model $ROOT_PATH/opus_scripts/build_data/spm_64k.model --output_format=piece \
        --inputs $ROOT_PATH/opus_scripts/build_data/raw/${SRC}_${TGT}/test.${TGT} \
        --outputs ${BPE_PATH}/${SRC}_${TGT}/test.${TGT}

    fairseq-preprocess --task "translation" --source-lang ${SRC} --target-lang ${TGT} \
      --testpref $BPE_PATH/${SRC}_${TGT}/test \
      --destdir ${BINARY_PATH} --padding-factor 1 --workers 16 \
      --srcdict ${BINARY_PATH}/dict.txt \
      --tgtdict ${BINARY_PATH}/dict.txt
    fairseq-preprocess --task "translation" --source-lang ${TGT} --target-lang ${SRC} \
      --testpref ${BPE_PATH}/${SRC}_${TGT}/test \
      --destdir ${BINARY_PATH} --padding-factor 1 --workers 16 \
      --srcdict ${BINARY_PATH}/dict.txt \
      --tgtdict ${BINARY_PATH}/dict.txt
done