#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建多语言翻译数据
"""

import os
import shutil

def create_multilingual_pairs():
    """创建多语言翻译对"""
    print("🌍 创建多语言翻译数据...")
    
    # 选择的语言
    languages = ['en', 'de', 'es']  # 英语、德语、西班牙语
    lang_names = {
        'en': '英语',
        'de': '德语', 
        'es': '西班牙语'
    }
    
    source_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/tokenized"
    output_dir = "multilingual_data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 源目录: {source_dir}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🌍 选择语言: {', '.join([f'{lang}({lang_names[lang]})' for lang in languages])}")
    
    # 创建所有语言对
    pairs = []
    for i, src_lang in enumerate(languages):
        for j, tgt_lang in enumerate(languages):
            if i != j:  # 不同语言之间
                pairs.append((src_lang, tgt_lang))
    
    print(f"\n📋 将创建 {len(pairs)} 个翻译方向:")
    for src, tgt in pairs:
        print(f"  {src}({lang_names[src]}) → {tgt}({lang_names[tgt]})")
    
    # 复制和重命名文件
    for split in ['train', 'valid', 'test']:
        print(f"\n🔄 处理 {split} 数据...")
        
        for src_lang, tgt_lang in pairs:
            pair_name = f"{src_lang}-{tgt_lang}"
            
            # 源文件路径
            src_file = os.path.join(source_dir, f"{split}.{src_lang}")
            tgt_file = os.path.join(source_dir, f"{split}.{tgt_lang}")
            
            # 目标文件路径
            out_src_file = os.path.join(output_dir, f"{split}.{pair_name}.{src_lang}")
            out_tgt_file = os.path.join(output_dir, f"{split}.{pair_name}.{tgt_lang}")
            
            if os.path.exists(src_file) and os.path.exists(tgt_file):
                shutil.copy2(src_file, out_src_file)
                shutil.copy2(tgt_file, out_tgt_file)
                
                # 检查文件大小
                src_size = os.path.getsize(out_src_file) / (1024*1024)  # MB
                tgt_size = os.path.getsize(out_tgt_file) / (1024*1024)  # MB
                
                print(f"  ✅ {pair_name}: {src_size:.1f}MB + {tgt_size:.1f}MB")
            else:
                print(f"  ❌ {pair_name}: 文件不存在")
    
    # 创建预处理脚本
    create_preprocess_script(languages, output_dir)
    
    print(f"\n✅ 多语言数据创建完成!")
    print(f"📁 数据位置: {output_dir}/")
    print(f"🚀 下一步: 运行 python preprocess_multilingual.py")

def create_preprocess_script(languages, output_dir):
    """创建预处理脚本"""
    print(f"\n📝 创建预处理脚本...")
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言数据预处理脚本
"""

import os
import subprocess

def preprocess_multilingual():
    """预处理多语言数据"""
    print("🔄 开始多语言数据预处理...")
    
    languages = {languages}
    output_dir = "{output_dir}"
    bin_dir = "{output_dir}-bin"
    
    # 创建输出目录
    os.makedirs(bin_dir, exist_ok=True)
    
    # 构建fairseq-preprocess命令
    cmd = [
        "python", "-m", "fairseq_cli.preprocess",
        "--source-lang", "src",
        "--target-lang", "tgt", 
        "--trainpref", f"{{output_dir}}/train",
        "--validpref", f"{{output_dir}}/valid",
        "--testpref", f"{{output_dir}}/test",
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
        print(f"📁 二进制数据保存在: {{bin_dir}}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 预处理失败: {{e}}")
        print(f"错误输出: {{e.stderr}}")
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
        "{output_dir}-bin",
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
        "--eval-bleu-args", '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}',
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
        print("\\n🎉 多语言预处理完成!")
        print("🚀 下一步: python train_multilingual.py")
'''
    
    with open("preprocess_multilingual.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 创建了 preprocess_multilingual.py")

if __name__ == "__main__":
    create_multilingual_pairs() 