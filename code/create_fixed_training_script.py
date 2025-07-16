#!/usr/bin/env python3
"""
创建修复后的训练脚本
基于实际可用的数据集重新配置训练
"""

import os

def create_fixed_training_script():
    """创建修复后的训练脚本"""
    
    script_content = '''#!/usr/bin/env python3
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
    print("\\n🚀 开始训练...")
    print("⏰ 预计训练时间: 2-3小时")
    
    log_file = os.path.join(log_dir, f"{ID}.log")
    
    try:
        print(f"📝 日志文件: {log_file}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=== 修复后的Europarl训练日志 ===\\n")
            f.write(f"语言对: de-en, es-en, it-en\\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write("=" * 50 + "\\n")
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
            print("\\n🎉 训练完成!")
            print(f"📂 模型保存在: {checkpoint_dir}")
            print(f"📝 日志保存在: {log_file}")
            
            # 显示最佳模型信息
            best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint):
                size_mb = os.path.getsize(best_checkpoint) / (1024*1024)
                print(f"💾 最佳模型大小: {size_mb:.1f}MB")
            
        else:
            print(f"\\n❌ 训练失败，返回码: {process.returncode}")
            print(f"📝 请检查日志: {log_file}")
            
    except Exception as e:
        print(f"\\n💥 训练过程出错: {e}")
    
    finally:
        # 切换回原目录
        os.chdir(original_dir)
        
        # 最终资源监控
        print("\\n" + "="*60)
        print("🔍 训练后资源状态:")
        monitor_resources()

if __name__ == "__main__":
    main()
'''
    
    # 保存脚本
    script_file = "europarl_fixed_training.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 创建了修复后的训练脚本: {script_file}")
    return script_file

def create_data_preprocessing_check():
    """创建数据预处理检查脚本"""
    
    check_content = '''#!/usr/bin/env python3
"""
检查数据预处理是否需要重新进行
"""

import os

def main():
    print("🔍 检查数据预处理状态")
    print("=" * 50)
    
    bin_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 需要的语言对（修复后）
    required_pairs = ["de-en", "es-en", "it-en"]
    
    print("📋 检查二进制数据文件:")
    
    all_exist = True
    for pair in required_pairs:
        src_lang, tgt_lang = pair.split("-")
        
        files_to_check = [
            f"train.{pair}.{src_lang}.bin",
            f"train.{pair}.{src_lang}.idx", 
            f"train.{pair}.{tgt_lang}.bin",
            f"train.{pair}.{tgt_lang}.idx",
            f"valid.{pair}.{src_lang}.bin",
            f"valid.{pair}.{src_lang}.idx",
            f"valid.{pair}.{tgt_lang}.bin", 
            f"valid.{pair}.{tgt_lang}.idx",
        ]
        
        pair_complete = True
        for file_name in files_to_check:
            file_path = os.path.join(bin_dir, file_name)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"  ✅ {file_name}: {size_mb:.1f}MB")
            else:
                print(f"  ❌ {file_name}: 不存在")
                pair_complete = False
                all_exist = False
        
        if pair_complete:
            print(f"  🎉 {pair}: 完整")
        else:
            print(f"  ⚠️  {pair}: 不完整")
        print()
    
    # 检查词典文件
    print("📚 检查词典文件:")
    required_langs = ["de", "en", "es", "it"]
    for lang in required_langs:
        dict_file = os.path.join(bin_dir, f"dict.{lang}.txt")
        if os.path.exists(dict_file):
            size_kb = os.path.getsize(dict_file) / 1024
            print(f"  ✅ dict.{lang}.txt: {size_kb:.1f}KB")
        else:
            print(f"  ❌ dict.{lang}.txt: 不存在")
            all_exist = False
    
    print("\\n" + "="*50)
    if all_exist:
        print("🎉 所有数据文件都存在，可以直接开始训练!")
        print("\\n🚀 运行命令:")
        print("python europarl_fixed_training.py")
    else:
        print("⚠️  部分数据文件缺失，需要重新预处理数据")
        print("\\n🔧 解决方案:")
        print("1. 重新运行数据预处理脚本")
        print("2. 或者检查原始数据是否完整")

if __name__ == "__main__":
    main()
'''
    
    check_file = "check_data_preprocessing.py"
    with open(check_file, 'w', encoding='utf-8') as f:
        f.write(check_content)
    
    print(f"✅ 创建了数据预处理检查脚本: {check_file}")
    return check_file

def main():
    print("🔧 创建修复方案")
    print("=" * 60)
    
    # 创建修复后的训练脚本
    training_script = create_fixed_training_script()
    
    print()
    
    # 创建数据检查脚本
    check_script = create_data_preprocessing_check()
    
    print()
    print("🎯 下一步操作:")
    print("1. 运行数据检查:")
    print(f"   python {check_script}")
    print()
    print("2. 如果数据完整，开始训练:")
    print(f"   python {training_script}")
    print()
    print("💡 修复要点:")
    print("- 只使用存在的3个语言对")
    print("- 降低学习率防止过拟合")
    print("- 增加dropout和正则化")
    print("- 添加早停机制")
    print("- 减少训练轮数")
    print()
    print("🎉 修复后的模型将支持:")
    print("- 德语 → 英语")
    print("- 西班牙语 → 英语") 
    print("- 意大利语 → 英语")

if __name__ == "__main__":
    main() 