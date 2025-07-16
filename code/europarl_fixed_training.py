#!/usr/bin/env python3
"""
修复后的Europarl训练脚本
只使用实际存在的语言对: de-en, es-en, it-en
"""

import os
import subprocess
import time
import psutil
import GPUtil

def monitor_resources():
    """监控系统资源"""
    try:
        # GPU信息
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"🖥️  GPU: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        
        # 内存信息
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.used//1024//1024//1024}GB/{memory.total//1024//1024//1024}GB ({memory.percent:.1f}%)")
        
    except Exception as e:
        print(f"⚠️  资源监控失败: {e}")

def main():
    print("🚀 修复后的Europarl训练")
    print("=" * 60)
    print("📋 使用的语言对: de-en, es-en, it-en")
    print("🎯 支持翻译方向: 德语→英语, 西语→英语, 意语→英语")
    print("=" * 60)
    
    # 监控资源
    monitor_resources()
    
    # 设置路径
    ROOT_PATH = "C:/Users/33491/PycharmProjects/machine"
    DATA_BIN = os.path.join(ROOT_PATH, "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin")
    FAIR_PATH = os.path.join(ROOT_PATH, "fairseq")
    WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
    
    # 训练参数
    METHOD = "europarl_fixed"
    ID = "1"
    
    # 创建目录
    checkpoint_dir = os.path.join(WORK_PATH, "checkpoints", METHOD, ID)
    log_dir = os.path.join(WORK_PATH, "logs", METHOD)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📂 检查点目录: {checkpoint_dir}")
    print(f"📝 日志目录: {log_dir}")
    
    # 切换到fairseq目录
    original_dir = os.getcwd()
    os.chdir(FAIR_PATH)
    print(f"📁 工作目录: {FAIR_PATH}")
    
    # 构建训练命令 - 修复后的配置
    train_cmd = [
        "fairseq-train", DATA_BIN,
        "--user-dir", "models/PhasedDecoder-main/PhasedDecoder/",
        "--seed", "0",
        "--fp16",
        "--ddp-backend=no_c10d",
        "--arch", "transformer_pdec_6_e_6_d",
        "--task", "translation_multi_simple_epoch",
        "--sampling-method", "temperature",
        "--sampling-temperature", "5",
        "--attention-position-bias", "1",
        "--adaption-flag", "True",
        "--adaption-inner-size", "2048",
        "--adaption-dropout", "0.1",
        "--contrastive-flag", "True",
        "--contrastive-type", "enc",
        "--dim", "512",
        "--mode", "1",
        "--cl-position", "6",
        "--temperature", "1.0",
        # 修复后的语言配置 - 只使用存在的语言对
        "--langs", "de,en,es,it",
        "--lang-pairs", "de-en,es-en,it-en",
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy_instruction",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0003",  # 降低学习率
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--max-epoch", "15",  # 减少epoch防止过拟合
        "--max-tokens", "2048",  # 适中的batch size
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--dropout", "0.3",  # 增加dropout
        "--attention-dropout", "0.1",
        "--activation-dropout", "0.1",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "3",
        "--patience", "5",  # 早停
        "--log-interval", "100",
        "--log-format", "simple",
        "--save-dir", checkpoint_dir,
        "--checkpoint-activations",  # 节省内存
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 开始训练
    print("\n🚀 开始训练...")
    print("⏰ 预计训练时间: 2-3小时")
    
    log_file = os.path.join(log_dir, f"{ID}.log")
    
    try:
        print(f"📝 日志文件: {log_file}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=== 修复后的Europarl训练日志 ===\n")
            f.write(f"语言对: de-en, es-en, it-en\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            process = subprocess.Popen(
                train_cmd, 
                env=env, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时显示训练进度
            for line in process.stdout:
                print(line.strip())
                f.write(line)
                f.flush()
            
            process.wait()
        
        if process.returncode == 0:
            print("\n🎉 训练完成!")
            print(f"📂 模型保存在: {checkpoint_dir}")
            print(f"📝 日志保存在: {log_file}")
            
            # 显示最佳模型信息
            best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint):
                size_mb = os.path.getsize(best_checkpoint) / (1024*1024)
                print(f"💾 最佳模型大小: {size_mb:.1f}MB")
            
        else:
            print(f"\n❌ 训练失败，返回码: {process.returncode}")
            print(f"📝 请检查日志: {log_file}")
            
    except Exception as e:
        print(f"\n💥 训练过程出错: {e}")
    
    finally:
        # 切换回原目录
        os.chdir(original_dir)
        
        # 最终资源监控
        print("\n" + "="*60)
        print("🔍 训练后资源状态:")
        monitor_resources()

if __name__ == "__main__":
    main()
