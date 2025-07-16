#!/bin/bash

# 设置你的项目根目录（请根据实际情况修改）
PROJECT_ROOT="/c/Users/33491/PycharmProjects/machine"
PDEC_WORK="$PROJECT_ROOT/pdec_work"
FAIRSEQ="$PROJECT_ROOT/fairseq"

# 使用现有的数据目录（已移至D盘）
DATA_BIN="/d/europarl_15-bin"
num_gpus=1

METHOD=ted_pdec_mini
ID=1

ENC=6
DEC=6
BIAS=1
ADAPTION='True'
DROP=0.1
INNER=2048
CONTRASTIVE='True'
POSITION=6
TYPE='enc'
T=1.0
DIM=512
MODE=1
SEED=0

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误：未提供参数"
    echo "请使用以下参数之一:"
    echo "  1 = 训练"
    echo "  2 = 合并检查点"
    echo "  3 = 推理"
    echo "  4 = BERTScore 评估"
    echo "  5 = COMET 评估"
    echo "  6 = 生成表格"
    echo "  7 = 创建GPU内存清理脚本"
    echo "  8 = 使用GPU内存清理脚本重新训练"
    exit 1
fi

# 使用if-elif结构替代case
if [ "$1" = "1" ]; then
    echo "🚀 步骤 1: 开始训练(精简版)..."
    mkdir -p "$PDEC_WORK/checkpoints/$METHOD/$ID"
    mkdir -p "$PDEC_WORK/logs/$METHOD"

    echo "当前数据目录：$DATA_BIN"

    cd "$FAIRSEQ" || { echo "❌ 错误：无法进入 fairseq 目录"; exit 1; }

    # 添加父目录到 PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

    # 训练命令
    CUDA_VISIBLE_DEVICES=0 /d/conda_envs/YOLOV11/python.exe -m fairseq_cli.train "$DATA_BIN" \
    --user-dir "$PDEC_WORK/models/PhasedDecoder" \
    --seed $SEED --fp16 --ddp-backend=no_c10d \
    --arch transformer \
    --encoder-layers $ENC \
    --decoder-layers $DEC \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" --sampling-temperature 5 \
    --langs "en,de,es,it" \
    --lang-pairs "en-de,de-en,en-es,es-en,en-it,it-en" \
    --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 --max-epoch 5 --max-tokens 4000 \
    --share-all-embeddings --weight-decay 0.0001 \
    --no-epoch-checkpoints --no-progress-bar \
    --keep-best-checkpoints 3 --log-interval 1000 --log-format simple \
    --max-sentences 16 \
    --update-freq 2 \
    --save-dir "$PDEC_WORK/checkpoints/$METHOD/$ID" > "$PDEC_WORK/logs/$METHOD/$ID.log"

elif [ "$1" = "2" ]; then
    echo "🔄 步骤 2: 合并最优 checkpoint..."
    python -c "
import os, glob, sys
sys.path.append('$FAIRSEQ')
checkpoint_dir = '$PDEC_WORK/checkpoints/$METHOD/$ID'
output_file = '$PDEC_WORK/checkpoints/$METHOD/$ID/checkpoint_averaged.pt'
checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint.best_loss_*'))
if checkpoints:
    from fairseq.scripts.average_checkpoints import main
    sys.argv = ['average_checkpoints.py', '--inputs'] + checkpoints + ['--output', output_file]
    print(f'找到 {len(checkpoints)} 个检查点，开始合并...')
    main()
    print(f'检查点已合并到 {output_file}')
else:
    print('⚠️ 警告：未找到 best loss 检查点。跳过 average_checkpoints.py')
"

elif [ "$1" = "3" ]; then
    echo "🔍 步骤 3: 开始推理..."
    cd "$PDEC_WORK" || { echo "❌ 错误：无法进入工作目录"; exit 1; }
    bash ted_evaluation/batch_inference_mini.sh "$METHOD" "$ID" "$PROJECT_ROOT" "$num_gpus" "$DATA_BIN"

elif [ "$1" = "4" ]; then
    echo "📊 步骤 4: 运行 BERTScore 评估..."
    cd "$PDEC_WORK" || { echo "❌ 错误：无法进入工作目录"; exit 1; }
    for tgt in en de es it; do
        CUDA_VISIBLE_DEVICES=0 python ted_evaluation/ted_bertscore_mini.py "$METHOD" "$ID" "$tgt" 0 "$PROJECT_ROOT"
    done

elif [ "$1" = "5" ]; then
    echo "🎯 步骤 5: 运行 COMET 评估..."
    cd "$PDEC_WORK" || { echo "❌ 错误：无法进入工作目录"; exit 1; }
    for tgt in en de es it; do
        python ted_evaluation/ted_comet_mini.py "$METHOD" "$ID" "$tgt" "$PROJECT_ROOT"
    done

elif [ "$1" = "6" ]; then
    echo "📈 步骤 6: 生成 Excel 表格..."
    cd "$PDEC_WORK" || { echo "❌ 错误：无法进入工作目录"; exit 1; }
    mkdir -p "$PDEC_WORK/excel"
    python ted_evaluation/make_table_mini.py "$METHOD" "$ID" "$PROJECT_ROOT"

elif [ "$1" = "7" ]; then
    echo "🧹 步骤 7: 创建GPU内存清理脚本..."
    mkdir -p "$PDEC_WORK/scripts"
    cat > "$PDEC_WORK/scripts/gpu_memory_cleaner.py" << 'EOF'
#!/usr/bin/env python3
"""
GPU内存清理工具 - 在训练过程中定期清理GPU内存
使用方法:
python gpu_memory_cleaner.py --interval 3600 --pid <训练进程PID>
"""

import argparse
import time
import os
import sys
import signal
import logging
import subprocess
import torch
import gc
import psutil
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('gpu_memory_cleaner')

def get_gpu_memory_info():
    """获取GPU内存使用情况"""
    if not torch.cuda.is_available():
        return "GPU不可用", 0, 0

    try:
        # 获取当前GPU设备
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        name = gpu_properties.name

        # 获取内存信息
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - reserved_memory

        # 转换为MB
        total_memory_mb = total_memory / 1024 / 1024
        reserved_memory_mb = reserved_memory / 1024 / 1024
        allocated_memory_mb = allocated_memory / 1024 / 1024
        free_memory_mb = free_memory / 1024 / 1024

        return name, total_memory_mb, free_memory_mb, allocated_memory_mb, reserved_memory_mb
    except Exception as e:
        logger.error(f"获取GPU信息时出错: {e}")
        return "未知", 0, 0, 0, 0

def clean_gpu_memory():
    """清理GPU内存"""
    try:
        # 强制Python垃圾回收
        gc.collect()

        # 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 尝试释放一些不必要的PyTorch缓存
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.device.type == 'cuda':
                        obj.detach_()
                        del obj
            except Exception:
                pass

        # 再次进行垃圾回收
        gc.collect()

        return True
    except Exception as e:
        logger.error(f"清理GPU内存时出错: {e}")
        return False

def is_process_running(pid):
    """检查进程是否在运行"""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def main():
    parser = argparse.ArgumentParser(description='GPU内存清理工具')
    parser.add_argument('--interval', type=int, default=3600, help='清理间隔(秒)')
    parser.add_argument('--pid', type=int, help='要监视的训练进程PID')
    parser.add_argument('--log-file', type=str, default='gpu_cleaner.log', help='日志文件路径')

    args = parser.parse_args()

    # 添加文件日志
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"GPU内存清理工具已启动，清理间隔: {args.interval}秒")

    if args.pid:
        logger.info(f"监视训练进程PID: {args.pid}")

    try:
        while True:
            # 检查训练进程是否还在运行
            if args.pid and not is_process_running(args.pid):
                logger.info(f"训练进程 {args.pid} 已结束，清理工具退出")
                break

            # 获取清理前的GPU内存信息
            gpu_info_before = get_gpu_memory_info()
            logger.info(f"清理前 - GPU: {gpu_info_before[0]}, 总内存: {gpu_info_before[1]:.2f}MB, "
                       f"可用: {gpu_info_before[2]:.2f}MB, 已分配: {gpu_info_before[3]:.2f}MB, 已预留: {gpu_info_before[4]:.2f}MB")

            # 清理GPU内存
            success = clean_gpu_memory()

            # 获取清理后的GPU内存信息
            gpu_info_after = get_gpu_memory_info()
            logger.info(f"清理后 - GPU: {gpu_info_after[0]}, 总内存: {gpu_info_after[1]:.2f}MB, "
                       f"可用: {gpu_info_after[2]:.2f}MB, 已分配: {gpu_info_after[3]:.2f}MB, 已预留: {gpu_info_after[4]:.2f}MB")

            freed_memory = gpu_info_after[2] - gpu_info_before[2]
            logger.info(f"{'成功' if success else '失败'}清理GPU内存，释放了 {freed_memory:.2f}MB")

            # 等待下一次清理
            logger.info(f"等待 {args.interval} 秒后进行下一次清理...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("用户中断，清理工具退出")
    except Exception as e:
        logger.error(f"发生错误: {e}")

    logger.info("GPU内存清理工具已退出")

if __name__ == "__main__":
    main()
EOF

elif [ "$1" = "8" ]; then
    echo "🚀 步骤 8: 使用GPU内存清理脚本重新训练..."
    mkdir -p "$PDEC_WORK/checkpoints/$METHOD/$ID"
    mkdir -p "$PDEC_WORK/logs/$METHOD"
    mkdir -p "$PDEC_WORK/scripts"

    echo "当前数据目录：$DATA_BIN"

    cd "$FAIRSEQ" || { echo "❌ 错误：无法进入 fairseq 目录"; exit 1; }

    # 添加父目录到 PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

    # 启动训练
    CUDA_VISIBLE_DEVICES=0 /d/conda_envs/YOLOV11/python.exe -m fairseq_cli.train "$DATA_BIN" \
    --user-dir "$PDEC_WORK/models/PhasedDecoder" \
    --seed $SEED --fp16 --ddp-backend=no_c10d \
    --arch transformer \
    --encoder-layers $ENC \
    --decoder-layers $DEC \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" --sampling-temperature 5 \
    --langs "en,de,es,it" \
    --lang-pairs "en-de,de-en,en-es,es-en,en-it,it-en" \
    --encoder-langtok tgt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 --max-epoch 5 --max-tokens 4000 \
    --share-all-embeddings --weight-decay 0.0001 \
    --no-epoch-checkpoints --no-progress-bar \
    --keep-best-checkpoints 3 --log-interval 1000 --log-format simple \
    --max-sentences 16 \
    --update-freq 2 \
    --save-dir "$PDEC_WORK/checkpoints/$METHOD/$ID" > "$PDEC_WORK/logs/$METHOD/$ID.log" &

    # 获取训练进程PID
    TRAIN_PID=$!
    echo "训练进程PID: $TRAIN_PID"

    # 启动GPU内存清理脚本
    echo "启动GPU内存清理脚本..."
    python "$PDEC_WORK/scripts/gpu_memory_cleaner.py" --interval 1800 --pid $TRAIN_PID --log-file "$PDEC_WORK/logs/gpu_cleaner.log" &

    echo "训练和GPU内存清理已启动"
    echo "可以使用以下命令查看训练日志:"
    echo "tail -f $PDEC_WORK/logs/$METHOD/$ID.log"
    echo "可以使用以下命令查看GPU清理日志:"
    echo "tail -f $PDEC_WORK/logs/gpu_cleaner.log"

else
    echo "❌ 错误：未知命令: $1"
    echo "请使用以下参数之一:"
    echo "  1 = 训练"
    echo "  2 = 合并检查点"
    echo "  3 = 推理"
    echo "  4 = BERTScore 评估"
    echo "  5 = COMET 评估"
    echo "  6 = 生成表格"
    echo "  7 = 创建GPU内存清理脚本"
    echo "  8 = 使用GPU内存清理脚本重新训练"
    exit 1
fi