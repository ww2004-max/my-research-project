#!/bin/bash

ROOT_PATH="../"

mkdir $ROOT_PATH/ted_scripts/logs
mkdir $ROOT_PATH/ted_scripts/checkpoints
mkdir $ROOT_PATH/ted_scripts/results

# download dataset
curl -o ted_talks.tar.gz http://phontron.com/data/ted_talks.tar.gz
tar -xzvf ted_talks.tar.gz -C ${ROOT_PATH}/ted_scripts/ted_talks

RAW_PATH=${ROOT_PATH}/ted_scripts/build_data/raw
TURN_PATH=${ROOT_PATH}/ted_scripts/build_data/turncated
mkdir $RAW_PATH
mkdir $TURN_PATH
python ${ROOT_PATH}/ted_scripts/ted_reader.py
python ${ROOT_PATH}/ted_scripts/turncation.py
for lang in ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
   scp ${RAW_PATH}"/en_"${lang}"/train."$lang ${TURN_PATH}"/en_"${lang}"/train."$lang
   scp ${RAW_PATH}"/en_"${lang}"/train.en" ${TURN_PATH}"/en_"${lang}"/train.en"
   scp ${RAW_PATH}"/"${lang}"_en/train."$lang ${TURN_PATH}"/"${lang}"_en/train."$lang
   scp ${RAW_PATH}"/"${lang}"_en/train.en" ${TURN_PATH}"/"${lang}"_en/train.en"
done

FAIR_PATH=${ROOT_PATH}/fairseq
SCRIPTS=${FAIR_PATH}/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py
SPM_DECODE=$SCRIPTS/spm_decode.py
BPESIZE=50000
TRAIN_FILES=${ROOT_PATH}/ted_scripts/build_data/bpe.input-output

# integrate training data for bpe (sentencepiece)
for lang in ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
  filename=${TURN_PATH}"/train."$lang
  echo $filename
  cat $filename >> $TRAIN_FILES
done
filename=${TURN_PATH}"/ar_en/train.en"
cat $filename >> $TRAIN_FILES

echo "learning joint BPE over ${TRAIN_FILES}..."
python $SPM_TRAIN \
    --input=$TRAIN_FILES \
    --model_prefix=${ROOT_PATH}/ted_scripts/build_data/ted.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

BPE_PATH=${ROOT_PATH}/ted_scripts/build_data/bpe
mkdir $BPE_PATH
# encode row data via bpe
for src in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
     for tgt in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
         if [ $src == $tgt ]; then
             continue
         fi
         mkdir ${BPE_PATH}/${src}_${tgt}
         for split in train test valid; do
             if [[ $src != "en" && $tgt != "en" && $split != "test" ]]; then
                 continue
             fi
             python $SPM_ENCODE --model ${ROOT_PATH}/ted_scripts/build_data/ted.bpe.model --output_format=piece \
             --inputs ${TURN_PATH}/${src}_${tgt}/${split}.${src} \
             --outputs ${BPE_PATH}/${src}_${tgt}/${split}.${src}

             python $SPM_ENCODE --model ${ROOT_PATH}/ted_scripts/build_data/ted.bpe.model --output_format=piece \
             --inputs ${TURN_PATH}/${src}_${tgt}/${split}.${tgt} \
             --outputs ${BPE_PATH}/${src}_${tgt}/${split}.${tgt}
         done
     done
done

BINARY_PATH=${ROOT_PATH}/ted_19-bin
mkdir $BINARY_PATH
cut -f 1 ${ROOT_PATH}/ted_scripts/build_data/ted.bpe.vocab | tail -n +4 | sed "s/$/ 1/g" > ${BINARY_PATH}/dict.txt
for src in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
  for tgt in en ar he ru ko it ja zh es nl vi tr fr pl ro fa hr cs de; do
    if [ $src == $tgt ]; then
      continue
    fi
    if [ $src == 'en' ] || [ $tgt == 'en' ]; then
        fairseq-preprocess --task "translation" --source-lang $src --target-lang $tgt \
        --trainpref ${BPE_PATH}/${src}"_"${tgt}/train \
        --validpref ${BPE_PATH}/${src}"_"${tgt}/valid \
        --testpref ${BPE_PATH}/${src}"_"${tgt}/test \
        --destdir ${BINARY_PATH} --padding-factor 1 --workers 128 \
        --srcdict ${BINARY_PATH}/dict.txt --tgtdict ${BINARY_PATH}/dict.txt
    fi
    if [ $src != 'en' ] && [ $tgt != 'en' ]; then
        fairseq-preprocess --task "translation" --source-lang $src --target-lang $tgt \
        --testpref ${BPE_PATH}/${src}"_"${tgt}/test \
        --destdir ${BINARY_PATH} --padding-factor 1 --workers 128 \
        --srcdict ${BINARY_PATH}/dict.txt --tgtdict ${BINARY_PATH}/dict.txt
    fi
  done
done

# align for analysis
MONO_PATH=${ROOT_PATH}/ted_scripts/build_data/mono
mkdir $MONO_PATH
python ${ROOT_PATH}/ted_scripts/build_data/align.py