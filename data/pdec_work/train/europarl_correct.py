#!/usr/bin/env python3
"""
基于原始vanilla.sh的正确Europarl训练脚本
"""

import os
import sys
import subprocess
import time

def main():
    # 设置路径（基于原始脚本）
    FAIRSEQ_ROOT = "C:/Users/33491/PycharmProjects/machine/fairseq"
    PROJECT_ROOT = os.path.join(FAIRSEQ_ROOT, "models/ZeroTrans")
    DATA_PATH = os.path.join(PROJECT_ROOT, "europarl_scripts/build_data/europarl_15-bin")
    
    # 设置输出路径
    ROOT_PATH = "C:/Users/33491/PycharmProjects/machine"
    WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
    
    METHOD = "europarl_vanilla"
    ID = "1"
    
    SAVE_PATH = os.path.join(WORK_PATH, "checkpoints", METHOD)
    LOG_PATH = os.path.join(WORK_PATH, "logs", METHOD)
    
    print("开始正确的Europarl训练流程...")
    
    # 创建必要目录
    dirs_to_create = [SAVE_PATH, LOG_PATH, os.path.join(SAVE_PATH, ID)]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 切换到fairseq根目录
    original_dir = os.getcwd()
    os.chdir(FAIRSEQ_ROOT)
    print(f"切换到目录: {FAIRSEQ_ROOT}")
    
    # 构建训练命令（基于原始脚本）
    train_cmd = [
        "python", "train.py", DATA_PATH,
        "--seed", "0",
        "--fp16",
        "--ddp-backend=no_c10d",
        "--arch", "transformer",
        "--task", "translation_multi_simple_epoch",
        "--sampling-method", "temperature",
        "--sampling-temperature", "5",
        "--langs", "en,de,es,it",
        "--lang-pairs", "en-de,de-en,en-es,es-en,en-it,it-en",
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0005",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--max-epoch", "10",  # 减少epoch数用于测试
        "--max-tokens", "2000",
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "5",
        "--log-interval", "10",
        "--save-dir", os.path.join(SAVE_PATH, ID),
        "--num-workers", "0"
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # 使用一个GPU
    
    # 开始训练
    print("开始模型训练...")
    log_file = os.path.join(LOG_PATH, f"{ID}.log")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.run(train_cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        
        if process.returncode == 0:
            print("训练完成!")
        else:
            print(f"训练失败，返回码: {process.returncode}")
            print(f"请检查日志文件: {log_file}")
            
            # 显示日志文件的最后几行
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print("\n最后20行日志:")
                    for line in lines[-20:]:
                        print(line.strip())
            except:
                pass
            return
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return
    
    # 切换回原目录
    os.chdir(original_dir)
    
    print("\n训练完成!")
    print(f"结果保存在: {os.path.join(SAVE_PATH, ID)}")
    print(f"日志保存在: {log_file}")

if __name__ == "__main__":
    main() 