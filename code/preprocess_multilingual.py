#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言数据预处理脚本
"""

import os
import subprocess

def preprocess_multilingual():
    """预处理多语言数据"""
    print("🔄 开始多语言数据预处理...")
    
    languages = ['en', 'de', 'es']
    output_dir = "multilingual_data"
    bin_dir = "multilingual_data-bin"
    
    # 创建输出目录
    os.makedirs(bin_dir, exist_ok=True)
    
    # 构建fairseq-preprocess命令
    cmd = [
        "python", "-m", "fairseq_cli.preprocess",
        "--source-lang", "src",
        "--target-lang", "tgt", 
        "--trainpref", f"{output_dir}/train",
        "--validpref", f"{output_dir}/valid",
        "--testpref", f"{output_dir}/test",
        "--destdir", bin_dir,
        "--workers", "4",
        "--joined-dictionary",  # 共享词典
        "--bpe", "subword_nmt",
        "--bpe-codes", "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl.bpe.model"
    ]
    
    print("💻 运行命令:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 预处理成功!")
        print(f"📁 二进制数据保存在: {bin_dir}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 预处理失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def create_training_script():
    """创建多语言训练脚本"""
    print("📝 创建多语言训练脚本...")
    
    training_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess

def train_multilingual():
    cmd = [
        "python", "-m", "fairseq_cli.train",
        "multilingual_data-bin",
        "--arch", "transformer",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--lr", "0.0005",
        "--min-lr", "1e-09",
        "--lr-scheduler", "inverse_sqrt",
        "--weight-decay", "0.0001",
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",
        "--max-tokens", "4096",
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
        "--eval-bleu-detok", "moses",
        "--eval-bleu-remove-bpe",
        "--best-checkpoint-metric", "bleu",
        "--maximize-best-checkpoint-metric",
        "--save-dir", "pdec_work/checkpoints/multilingual",
        "--tensorboard-logdir", "pdec_work/logs/multilingual",
        "--max-epoch", "3",
        "--patience", "10"
    ]
    
    print("🚀 开始多语言训练...")
    subprocess.run(cmd)

if __name__ == "__main__":
    train_multilingual()
"""
    
    with open("train_multilingual.py", "w", encoding="utf-8") as f:
        f.write(training_script)
    
    print("✅ 创建了 train_multilingual.py")

if __name__ == "__main__":
    if preprocess_multilingual():
        create_training_script()
        print("\n🎉 多语言预处理完成!")
        print("🚀 下一步: python train_multilingual.py")
