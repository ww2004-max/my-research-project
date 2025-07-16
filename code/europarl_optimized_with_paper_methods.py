#!/usr/bin/env python3
"""
PhasedDecoder训练脚本 - 结合论文优化方法
基于论文: Improving Language Transfer Capability of Decoder-only Architecture in MNMT
优化方法:
1. 早停机制防止过拟合
2. 学习率调度优化
3. 数据增强和正则化
4. 多任务学习策略
"""

import os
import sys
import subprocess
import time
import json
import torch
import psutil
from datetime import datetime

def setup_environment():
    """设置环境和路径"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            used, total, util = line.split(', ')
            print(f"GPU {i}: {used}MB/{total}MB ({float(used)/float(total)*100:.1f}%), 利用率: {util}%")
            
        return True
    except:
        print("无法获取GPU信息")
        return False

def cleanup_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU缓存已清理")

def kill_competing_processes():
    """终止可能竞争GPU的进程"""
    try:
        # 查找Python训练进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'fairseq' in cmdline and 'train.py' in cmdline and proc.info['pid'] != os.getpid():
                        print(f"发现竞争进程 PID {proc.info['pid']}: {cmdline[:100]}...")
                        proc.terminate()
                        print(f"✅ 已终止进程 {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"清理进程时出错: {e}")

def create_optimized_training_config():
    """创建优化的训练配置"""
    config = {
        # 基础配置
        "data_bin": "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin",
        "save_dir": "pdec_work/checkpoints/europarl_paper_optimized",
        "restore_file": "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt",
        
        # 论文启发的优化配置
        "max_tokens": 2048,  # 平衡内存和性能
        "update_freq": 2,    # 增加有效batch size
        "max_epoch": 10,     # 更多轮次但有早停
        
        # 防过拟合策略
        "dropout": 0.3,      # 增加dropout
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "weight_decay": 0.01,
        
        # 学习率优化
        "lr": 0.0003,        # 降低学习率
        "lr_scheduler": "inverse_sqrt",
        "warmup_updates": 1000,
        "warmup_init_lr": 1e-07,
        
        # 早停和验证
        "patience": 5,       # 早停耐心值
        "validate_interval": 1,
        "save_interval": 1,
        "keep_best_checkpoints": 3,
        
        # 内存优化
        "checkpoint_activations": True,
        "ddp_backend": "no_c10d",
        "fp16": True,        # 使用混合精度
        
        # 多语言优化
        "lang_pairs": "en-de,de-en,en-es,es-en,en-it,it-en",
        "sampling_method": "temperature",
        "sampling_temperature": 1.5,  # 平衡不同语言对
        
        # 正则化
        "label_smoothing": 0.1,
        "clip_norm": 1.0,
    }
    
    return config

def build_training_command(config):
    """构建训练命令"""
    cmd = [
        "python", "fairseq_cli/train.py",
        config["data_bin"],
        
        # 任务和架构
        "--task", "translation_multi_simple_epoch",
        "--arch", "phaseddecoder_iwslt_de_en",
        "--lang-pairs", config["lang_pairs"],
        
        # 数据和批次
        "--max-tokens", str(config["max_tokens"]),
        "--update-freq", str(config["update_freq"]),
        "--max-epoch", str(config["max_epoch"]),
        
        # 优化器
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.98)",
        "--lr", str(config["lr"]),
        "--lr-scheduler", config["lr_scheduler"],
        "--warmup-updates", str(config["warmup_updates"]),
        "--warmup-init-lr", str(config["warmup_init_lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--clip-norm", str(config["clip_norm"]),
        
        # 正则化
        "--dropout", str(config["dropout"]),
        "--attention-dropout", str(config["attention_dropout"]),
        "--activation-dropout", str(config["activation_dropout"]),
        "--label-smoothing", str(config["label_smoothing"]),
        
        # 保存和验证
        "--save-dir", config["save_dir"],
        "--restore-file", config["restore_file"],
        "--validate-interval", str(config["validate_interval"]),
        "--save-interval", str(config["save_interval"]),
        "--keep-best-checkpoints", str(config["keep_best_checkpoints"]),
        "--patience", str(config["patience"]),
        "--no-epoch-checkpoints",
        
        # 多语言采样
        "--sampling-method", config["sampling_method"],
        "--sampling-temperature", str(config["sampling_temperature"]),
        
        # 内存和性能优化
        "--checkpoint-activations",
        "--ddp-backend", config["ddp_backend"],
        "--fp16",
        "--fp16-init-scale", "128",
        "--fp16-scale-window", "128",
        
        # 日志
        "--log-format", "simple",
        "--log-interval", "50",
        "--tensorboard-logdir", f"{config['save_dir']}/tensorboard",
        
        # 其他
        "--seed", "42",
        "--num-workers", "0",
    ]
    
    return cmd

def monitor_training_progress(save_dir):
    """监控训练进度"""
    log_file = os.path.join(save_dir, "train.log")
    best_loss = float('inf')
    
    print(f"\n📊 监控训练进度...")
    print(f"日志文件: {log_file}")
    print("=" * 80)
    
    return best_loss

def main():
    print("🚀 PhasedDecoder优化训练 - 基于论文方法")
    print("=" * 80)
    print("优化策略:")
    print("1. 早停机制防止过拟合")
    print("2. 改进的学习率调度")
    print("3. 增强的正则化")
    print("4. 多语言采样平衡")
    print("5. 混合精度训练")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成: {ROOT_PATH}")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 清理资源
    print("\n🧹 清理系统资源...")
    kill_competing_processes()
    cleanup_gpu_memory()
    
    # 检查GPU状态
    print("\n🔍 检查GPU状态:")
    check_gpu_memory()
    
    # 创建训练配置
    config = create_optimized_training_config()
    
    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(config["save_dir"], "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"📋 训练配置已保存: {config_file}")
    
    # 构建训练命令
    cmd = build_training_command(config)
    
    print(f"\n🎯 开始优化训练...")
    print(f"保存目录: {config['save_dir']}")
    print(f"基础模型: {config['restore_file']}")
    print(f"最大轮数: {config['max_epoch']} (早停耐心值: {config['patience']})")
    print(f"学习率: {config['lr']} (预热: {config['warmup_updates']}步)")
    
    # 显示完整命令
    print(f"\n📝 训练命令:")
    print(" ".join(cmd))
    
    # 开始训练
    start_time = time.time()
    
    try:
        # 重定向输出到日志文件
        log_file = os.path.join(config["save_dir"], "train.log")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练开始时间: {datetime.now()}\n")
            f.write(f"训练命令: {' '.join(cmd)}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n⏰ 训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 日志文件: {log_file}")
        print("🔄 训练进行中...")
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时显示输出
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
        
        process.wait()
        
        if process.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n✅ 训练完成!")
            print(f"⏱️  总用时: {duration/3600:.2f}小时")
            print(f"📁 模型保存在: {config['save_dir']}")
            
            # 显示最终GPU状态
            print(f"\n🔍 最终GPU状态:")
            check_gpu_memory()
            
        else:
            print(f"\n❌ 训练失败，退出码: {process.returncode}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
    
    finally:
        cleanup_gpu_memory()

if __name__ == "__main__":
    main() 