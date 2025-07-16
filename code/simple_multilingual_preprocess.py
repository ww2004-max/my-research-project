#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的多语言数据预处理
直接使用现有的BPE处理方式
"""

import os
import shutil
import subprocess

def prepare_multilingual_data():
    """准备多语言数据，使用现有的BPE方式"""
    print("🌍 准备多语言翻译数据...")
    
    # 检查数据
    data_dir = "multilingual_data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    print(f"📂 数据目录: {data_dir}")
    
    # 列出所有文件
    files = os.listdir(data_dir)
    print(f"📋 找到 {len(files)} 个文件:")
    
    # 按语言对分组
    language_pairs = {}
    for file in files:
        if '.' in file:
            parts = file.split('.')
            if len(parts) >= 3:
                split = parts[0]  # train/valid/test
                pair = parts[1]   # en-de, de-es等
                lang = parts[2]   # en, de, es
                
                if pair not in language_pairs:
                    language_pairs[pair] = {}
                if split not in language_pairs[pair]:
                    language_pairs[pair][split] = {}
                
                language_pairs[pair][split][lang] = file
    
    print(f"\n🔍 发现的语言对:")
    for pair, splits in language_pairs.items():
        src_lang, tgt_lang = pair.split('-')
        print(f"  {pair}: {src_lang} → {tgt_lang}")
        for split in ['train', 'valid', 'test']:
            if split in splits:
                src_file = splits[split].get(src_lang, '❌')
                tgt_file = splits[split].get(tgt_lang, '❌')
                print(f"    {split}: {src_file} + {tgt_file}")
    
    # 创建合并的多语言数据
    create_combined_data(data_dir, language_pairs)
    
    return True

def create_combined_data(data_dir, language_pairs):
    """创建合并的多语言数据"""
    print(f"\n🔄 创建合并的多语言数据...")
    
    output_dir = "multilingual_combined"
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个split创建合并文件
    for split in ['train', 'valid', 'test']:
        print(f"\n📝 处理 {split} 数据...")
        
        src_combined = os.path.join(output_dir, f"{split}.src")
        tgt_combined = os.path.join(output_dir, f"{split}.tgt")
        
        with open(src_combined, 'w', encoding='utf-8') as src_out, \
             open(tgt_combined, 'w', encoding='utf-8') as tgt_out:
            
            total_lines = 0
            
            for pair, splits in language_pairs.items():
                if split not in splits:
                    continue
                
                src_lang, tgt_lang = pair.split('-')
                
                if src_lang in splits[split] and tgt_lang in splits[split]:
                    src_file = os.path.join(data_dir, splits[split][src_lang])
                    tgt_file = os.path.join(data_dir, splits[split][tgt_lang])
                    
                    if os.path.exists(src_file) and os.path.exists(tgt_file):
                        # 读取并写入源文件
                        with open(src_file, 'r', encoding='utf-8') as f:
                            src_lines = f.readlines()
                        
                        # 读取并写入目标文件
                        with open(tgt_file, 'r', encoding='utf-8') as f:
                            tgt_lines = f.readlines()
                        
                        # 确保行数匹配
                        min_lines = min(len(src_lines), len(tgt_lines))
                        
                        for i in range(min_lines):
                            # 添加语言标记
                            src_line = f"<{tgt_lang}> " + src_lines[i].strip() + "\n"
                            tgt_line = tgt_lines[i]
                            
                            src_out.write(src_line)
                            tgt_out.write(tgt_line)
                        
                        total_lines += min_lines
                        print(f"  ✅ {pair}: {min_lines} 行")
            
            print(f"  📊 {split} 总计: {total_lines} 行")
    
    print(f"\n✅ 合并数据创建完成!")
    print(f"📁 位置: {output_dir}/")
    
    # 创建简单的训练脚本
    create_simple_training_script(output_dir)

def create_simple_training_script(data_dir):
    """创建简单的训练脚本"""
    print(f"\n📝 创建训练脚本...")
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言模型训练脚本
"""

import os
import sys
import subprocess

# 添加fairseq路径
sys.path.insert(0, os.path.abspath('fairseq'))

def train_multilingual():
    """训练多语言模型"""
    print("🚀 开始多语言模型训练...")
    
    # 首先预处理数据
    print("🔄 预处理数据...")
    
    preprocess_cmd = [
        "python", "fairseq/fairseq_cli/preprocess.py",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--trainpref", "{data_dir}/train",
        "--validpref", "{data_dir}/valid", 
        "--testpref", "{data_dir}/test",
        "--destdir", "{data_dir}-bin",
        "--workers", "4"
    ]
    
    try:
        subprocess.run(preprocess_cmd, check=True)
        print("✅ 预处理成功!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 预处理失败: {{e}}")
        return False
    
    # 训练模型
    print("🏋️ 开始训练...")
    
    train_cmd = [
        "python", "fairseq/fairseq_cli/train.py",
        "{data_dir}-bin",
        "--arch", "transformer_iwslt_de_en",
        "--source-lang", "src",
        "--target-lang", "tgt",
        "--lr", "0.001",
        "--min-lr", "1e-09",
        "--lr-scheduler", "inverse_sqrt",
        "--weight-decay", "0.0001",
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", "0.1",
        "--max-tokens", "2048",
        "--eval-bleu",
        "--eval-bleu-args", '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}',
        "--eval-bleu-detok", "moses",
        "--eval-bleu-remove-bpe",
        "--best-checkpoint-metric", "bleu",
        "--maximize-best-checkpoint-metric",
        "--save-dir", "pdec_work/checkpoints/multilingual",
        "--tensorboard-logdir", "pdec_work/logs/multilingual",
        "--max-epoch", "3",
        "--patience", "10",
        "--warmup-updates", "4000",
        "--dropout", "0.3",
        "--attention-dropout", "0.1"
    ]
    
    try:
        subprocess.run(train_cmd, check=True)
        print("✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {{e}}")
        return False

if __name__ == "__main__":
    train_multilingual()
'''
    
    with open("train_multilingual_simple.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 创建了 train_multilingual_simple.py")

if __name__ == "__main__":
    if prepare_multilingual_data():
        print("\n🎉 多语言数据准备完成!")
        print("🚀 下一步: python train_multilingual_simple.py")
    else:
        print("❌ 数据准备失败") 