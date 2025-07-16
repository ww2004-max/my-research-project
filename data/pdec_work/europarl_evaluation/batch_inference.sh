#!/bin/bash

# Europarl batch inference script
# Usage: bash batch_inference.sh METHOD ID ROOT_PATH num_gpus DATA_BIN [languages...]

METHOD=$1
ID=$2
ROOT_PATH=$3
num_gpus=$4
DATA_BIN=$5

# Default languages if not provided
if [ $# -gt 5 ]; then
    shift 5
    LANGS=("$@")
else
    LANGS=("en" "de" "es" "it")
fi

FAIR_PATH=${ROOT_PATH}/fairseq
WORK_PATH=${ROOT_PATH}/pdec_work

cd ${FAIR_PATH}

# Language pairs for the specified languages
PAIRS=()
for src in "${LANGS[@]}"; do
    for tgt in "${LANGS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            # Only include the specified language pairs
            if [[ ("$src" == "en" && "$tgt" == "de") || ("$src" == "de" && "$tgt" == "en") || \
                  ("$src" == "en" && "$tgt" == "es") || ("$src" == "es" && "$tgt" == "en") || \
                  ("$src" == "en" && "$tgt" == "it") || ("$src" == "it" && "$tgt" == "en") ]]; then
                PAIRS+=("${src}-${tgt}")
            fi
        fi
    done
done

echo "Processing language pairs: ${PAIRS[@]}"

# Run inference for each language pair
for pair in "${PAIRS[@]}"; do
    IFS='-' read -r src tgt <<< "$pair"
    echo "Processing $src -> $tgt"
    
    fairseq-generate ${DATA_BIN} \
        --user-dir models/PhasedDecoder/ \
        --task translation_multi_simple_epoch \
        --source-lang $src --target-lang $tgt \
        --path ${WORK_PATH}/checkpoints/${METHOD}/${ID}/checkpoint_averaged.pt \
        --dataset-impl mmap \
        --beam 5 --lenpen 1.0 \
        --gen-subset test \
        --remove-bpe \
        --sacrebleu > ${WORK_PATH}/results/${METHOD}/${ID}/europarl_${src}_${tgt}.txt
        
    echo "Completed $src -> $tgt"
done

echo "All inference completed!"
cd ${WORK_PATH}