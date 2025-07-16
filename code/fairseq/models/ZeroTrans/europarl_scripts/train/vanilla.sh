#!/bin/bash

# Step 0: 设置项目根目录和路径
FAIRSEQ_ROOT="/c/Users/33491/PycharmProjects/machine/fairseq"
PROJECT_ROOT="$FAIRSEQ_ROOT/models/ZeroTrans"
DATA_PATH="$PROJECT_ROOT/europarl_scripts/build_data/europarl_15-bin"
SCRIPTS_PATH="$PROJECT_ROOT/europarl_scripts"

ID=1
METHOD="vanilla"

SAVE_PATH="$SCRIPTS_PATH/checkpoints/$METHOD/"
RESULTS_PATH="$SCRIPTS_PATH/results/$METHOD/"
LOG_PATH="$SCRIPTS_PATH/logs/$METHOD/"

BINARY_PATH="$DATA_PATH"  # 确保数据路径正确
INFER_SCRIPT="$SCRIPTS_PATH/evaluation/europarl_inference.sh"

# Step 1: 创建所需目录
if [ "$1" == "step1" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 1: Creating directories..."
    mkdir -p "$SAVE_PATH"
    mkdir -p "$LOG_PATH"
    mkdir -p "$RESULTS_PATH"
fi

# Step 2: 打印路径信息用于调试
if [ "$1" == "step2" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 2: Printing path info..."
    echo "FAIRSEQ_ROOT = $FAIRSEQ_ROOT"
    echo "DATA_PATH = $DATA_PATH"
    echo "SAVE_PATH = $SAVE_PATH"
    echo "RESULTS_PATH = $RESULTS_PATH"
    echo "LOG_PATH = $LOG_PATH"
fi

# Step 3: 检查 omegaconf 是否已安装
if [ "$1" == "step3" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 3: Checking/installing omegaconf..."
    python -c "import omegaconf" 2>/dev/null || {
        echo "omegaconf 未安装，正在安装..."
        pip install omegaconf
    }
fi

# Step 4: 进入 Fairseq 根目录
if [ "$1" == "step4" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 4: Entering Fairseq root directory..."
    cd "$FAIRSEQ_ROOT" || { echo "无法进入 Fairseq 目录: $FAIRSEQ_ROOT"; exit 1; }
fi

# Step 5: 开始训练模型
if [ "$1" == "step5" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 5: Starting training..."

    CUDA_VISIBLE_DEVICES=0 \
    python train.py "$DATA_PATH" \
    --seed 0 --fp16 --ddp-backend=no_c10d --arch transformer --task translation_multi_simple_epoch \
    --sampling-method "temperature" --sampling-temperature 5 \
    --langs "en,de" \
    --lang-pairs "en-de,de-en" \
    --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --max-epoch 60 --max-tokens 2000 \
    --share-all-embeddings --weight-decay 0.0001 \
    --no-epoch-checkpoints --no-progress-bar \
    --keep-best-checkpoints 5 --log-interval 10 \
    --save-dir "${SAVE_PATH}/${ID}" \
    --num-workers 0 \
    > "${LOG_PATH}/${ID}.log"
fi

# Step 6: 回到脚本目录并执行推理
if [ "$1" == "step6" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 6: Running inference script..."
    cd "$SCRIPTS_PATH" || { echo "无法进入脚本目录: $SCRIPTS_PATH"; exit 1; }

    sleep 5  # 延迟几秒确保文件写入完成

    bash "$INFER_SCRIPT" "$METHOD" "$ID"
fi