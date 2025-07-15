#!/bin/bash

# 设置路径
DATA_BIN="/d/europarl_15-bin"  # D盘上的数据路径
ROOT_PATH="/c/Users/33491/PycharmProjects/machine"
num_gpus=1
FAIR_PATH=${ROOT_PATH}/fairseq  # 修正为fairseq而不是fairseqirseq
WORK_PATH=${ROOT_PATH}/pdec_work

# 设置模型参数
METHOD=ted_pdec_mini
ID=1

# 确保结果目录存在
mkdir -p ${WORK_PATH}/results/${METHOD}/${ID}

echo "🔍 开始推理..."

# 创建推理脚本
cat > ${WORK_PATH}/ted_evaluation/batch_inference_mini.sh << 'EOF'
#!/bin/bash

METHOD=${1}
ID=${2}
ROOT_PATH=${3}
num_gpus=${4}
DATA_BIN=${5}

WORK_PATH=${ROOT_PATH}/pdec_work
SAVE_PATH=${WORK_PATH}/results
cd ${WORK_PATH}

pids=()
declare -A lang_pairs

mkdir -p ${SAVE_PATH}/${METHOD}/${ID}
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
        bash ted_evaluation/single_inference_mini.sh $METHOD $ID $src $tgt $gpu_id $ROOT_PATH $DATA_BIN &
        pids[$gpu_id]=$!
    fi
done

while :; do
    for gpu_id in $(seq 0 $((num_gpus - 1))); do
        if ! kill -0 ${pids[$gpu_id]} 2> /dev/null && [[ ${#lang_pairs[@]} -gt 0 ]]; then
            # if gpu is free, start the next one
            IFS=',' read -r src tgt <<< $(echo "${!lang_pairs[@]}" | cut -d' ' -f1)
            unset lang_pairs["$src,$tgt"]
            bash ted_evaluation/single_inference_mini.sh $METHOD $ID $src $tgt $gpu_id $ROOT_PATH $DATA_BIN &
            pids[$gpu_id]=$!
        fi
    done
    if [[ ${#lang_pairs[@]} -eq 0 ]]; then
        break 
    fi
    sleep 5
done

wait
echo "All translations completed."
EOF

# 创建单语言对推理脚本
cat > ${WORK_PATH}/ted_evaluation/single_inference_mini.sh << 'EOF'
#!/bin/bash

METHOD=${1}
ID=${2}
src=${3}
tgt=${4}
gpu_id=${5}
ROOT_PATH=${6}
DATA_BIN=${7}
WORK_PATH=${ROOT_PATH}/pdec_work
SAVE_PATH=${WORK_PATH}/results
DETOKENIZER=${ROOT_PATH}/moses/scripts/tokenizer/detokenizer.perl

cd ${ROOT_PATH}/fairseq  # 修正为fairseq而不是fairseqirseq

DIR=""
task='translation_multi_simple_epoch'
if [ $METHOD == 'ted_pdec_mini' ];then
  DIR='--user-dir models/PhasedDecoder/ '
fi

tgt_file=$src"-"$tgt".raw.txt"
echo "正在处理语言对 $src-$tgt..."

# 使用特定的Python环境
PYTHON_CMD="/d/conda_envs/work/python.exe"  # 使用您的conda环境中的Python

# 检查检查点文件是否存在
CHECKPOINT_FILE=$WORK_PATH/checkpoints/$METHOD/$ID/checkpoint_averaged.pt
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_FILE"
    exit 1
fi

# 执行推理命令
echo "执行推理命令..."
$PYTHON_CMD -m fairseq_cli.generate $DATA_BIN \
$DIR \
-s $src -t $tgt \
--langs "en,de,es,it" \
--lang-pairs "en-de,de-en,en-es,es-en,en-it,it-en" \
--path $CHECKPOINT_FILE \
--remove-bpe sentencepiece \
--required-batch-size-multiple 1 \
--task $task \
--encoder-langtok tgt \
--beam 4 > $SAVE_PATH/$METHOD/$ID/$tgt_file

# 检查推理结果是否生成
if [ ! -f "$SAVE_PATH/$METHOD/$ID/$tgt_file" ]; then
    echo "错误: 推理结果文件未生成"
    exit 1
fi

echo "提取假设、参考和源文本..."
# hypothesis
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^H" | sort -t '-' -k2n | cut -f 3- > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h"
# reference
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^T" | sort -t '-' -k2n | cut -f 2- > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r"
# source
cat $SAVE_PATH/$METHOD/$ID/$tgt_file | grep -P "^S" | sort -t '-' -k2n | cut -f 2- | sed 's/__[a-zA-Z_]*__ //' > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s"
rm $SAVE_PATH/$METHOD/$ID/$tgt_file

echo "进行detokenize处理..."
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h" | perl ${DETOKENIZER} -threads 32 -l $tgt >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r" | perl ${DETOKENIZER} -threads 32 -l $tgt >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s" | perl ${DETOKENIZER} -threads 32 -l $src >> $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.s"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".h"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".r"
rm $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".s"

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

echo "计算评估指标..."
# 使用特定的Python环境运行sacrebleu
$PYTHON_CMD -m sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h" -w 4 -tok $TOK < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r" > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".bleu
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".bleu" >> $SAVE_PATH/$METHOD/$ID/$ID".sacrebleu"

$PYTHON_CMD -m sacrebleu $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.h" -w 4 -m chrf --chrf-word-order 2 < $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".detok.r" > $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".chrf
echo $src"-"$tgt >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"
cat $SAVE_PATH/$METHOD/$ID/$src"-"$tgt".chrf" >> $SAVE_PATH/$METHOD/$ID/$ID".chrf"

echo "语言对 $src-$tgt 处理完成"
EOF

# 给脚本添加执行权限
chmod +x ${WORK_PATH}/ted_evaluation/batch_inference_mini.sh
chmod +x ${WORK_PATH}/ted_evaluation/single_inference_mini.sh

# 执行推理
cd ${WORK_PATH}
bash ted_evaluation/batch_inference_mini.sh ${METHOD} ${ID} ${ROOT_PATH} ${num_gpus} ${DATA_BIN}

echo "✅ 推理完成!"