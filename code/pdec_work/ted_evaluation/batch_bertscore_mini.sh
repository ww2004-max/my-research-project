#!/bin/bash

METHOD=${1}
ID=${2}
ROOT_PATH=${3}
num_gpus=${4}

WORK_PATH=${ROOT_PATH}/pdec_work
SAVE_PATH=${WORK_PATH}/results
cd ${WORK_PATH}

pids=()
declare -A lang_pairs

# 只初始化我们需要的4种语言对
for src in en de es it; do
    for tgt in en de es it; do
        if [[ $src != $tgt ]];then
            lang_pairs["$src,$tgt"]=1
        fi
    done
done

# GPU monitor
for gpu_id in $(seq 0 $((num_gpus - 1))); do
    if [[ ${#lang_pairs[@]} -gt 0 ]]; then
        IFS=',' read -r src tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
        unset lang_pairs["$src,$tgt"]
        CUDA_VISIBLE_DEVICES=$gpu_id python ted_evaluation/ted_bertscore.py $METHOD $ID $src $tgt $WORK_PATH &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            # if gpu is free, start the next one
            IFS=',' read -r src tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$src,$tgt"]
            CUDA_VISIBLE_DEVICES=$gpu_id python ted_evaluation/ted_bertscore.py $METHOD $ID $src $tgt $WORK_PATH &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break 
    fi
    sleep 5
done

wait
echo "All BERTScore evaluations completed."
