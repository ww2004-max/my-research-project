#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Europarl dataset training script for PhasedDecoder
使用GPU训练的Europarl多语言翻译模型
"""

import os
import sys
import subprocess
import time

def main():
    print("开始GPU加速的Europarl训练流程...")
    
    # 设置路径和参数
    ROOT_PATH = "C:/Users/33491/PycharmProjects/machine"
    DATA_BIN = "C:/Users/33491/PycharmProjects/machine/fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
    
    # 训练参数
    METHOD = "europarl_pdec"
    ID = "1"
    
    # 模型参数 (基于TED配置)
    ENC = 6
    DEC = 6
    BIAS = 1
    ADAPTION = 'True'
    DROP = 0.1
    INNER = 2048
    CONTRASTIVE = 'True'
    POSITION = 6
    TYPE = 'enc'
    T = 1.0
    DIM = 512
    MODE = 1
    SEED = 0
    
    # 创建目录
    checkpoint_dir = os.path.join(WORK_PATH, "checkpoints", METHOD, ID)
    logs_dir = os.path.join(WORK_PATH, "logs", METHOD)
    results_dir = os.path.join(WORK_PATH, "results", METHOD, ID)
    
    for directory in [checkpoint_dir, logs_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 检查CUDA
    print("检查CUDA状态...")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 切换到fairseq目录
    original_dir = os.getcwd()
    os.chdir(FAIRSEQ)
    print(f"切换到目录: {os.getcwd()}")
    
    # 构建训练命令
    train_cmd = [
        "python", "-m", "fairseq_cli.train", DATA_BIN,
        "--user-dir", "models/PhasedDecoder",  # 修正后的路径
        "--seed", str(SEED),
        "--fp16",
        "--ddp-backend=no_c10d",
        "--arch", f"transformer_pdec_{ENC}_e_{DEC}_d",
        "--task", "translation_multi_simple_epoch",
        "--sampling-method", "temperature",
        "--sampling-temperature", "5",
        "--attention-position-bias", str(BIAS),
        "--adaption-flag", ADAPTION,
        "--adaption-inner-size", str(INNER),
        "--adaption-dropout", str(DROP),
        "--contrastive-flag", CONTRASTIVE,
        "--contrastive-type", TYPE,
        "--dim", str(DIM),
        "--mode", str(MODE),
        "--cl-position", str(POSITION),
        "--temperature", str(T),
        "--langs", "en,de,es,it",
        "--lang-pairs", "en-de,de-en,en-es,es-en,en-it,it-en",
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy_instruction",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0005",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--max-epoch", "30",
        "--max-tokens", "4000",
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "5",
        "--log-interval", "1000",
        "--log-format", "simple",
        "--save-dir", checkpoint_dir
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 使用单GPU避免配置复杂性
    
    # 记录训练命令
    log_file = os.path.join(logs_dir, f"{ID}.log")
    print(f"训练命令: {' '.join(train_cmd)}")
    print(f"日志文件: {log_file}")
    
    # 开始训练
    print("🚀 开始GPU模型训练...")
    start_time = time.time()
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(
                train_cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=FAIRSEQ
            )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 训练成功完成！耗时: {training_time/3600:.2f}小时")
            print(f"检查点保存在: {checkpoint_dir}")
            print(f"训练日志: {log_file}")
        else:
            print(f"❌ 训练失败，返回码: {result.returncode}")
            print(f"请检查日志文件: {log_file}")
            
            # 显示最后20行日志
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print("\n最后20行日志:")
                    for line in lines[-20:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"无法读取日志文件: {e}")
    
    except Exception as e:
        print(f"❌ 训练过程中出现异常: {e}")
    
    finally:
        # 恢复原始目录
        os.chdir(original_dir)
        print(f"恢复目录: {os.getcwd()}")

if __name__ == "__main__":
    main() 