#!/usr/bin/env python3
"""
检查训练配置和数据集匹配情况
"""

import os
import json

def check_training_config():
    """检查训练配置"""
    print("🔍 检查训练配置和数据集匹配")
    print("=" * 60)
    
    # 1. 检查训练脚本中的语言对配置
    print("📋 训练脚本配置:")
    print("语言: en,de,es,it")
    print("语言对: en-de,de-en,en-es,es-en,en-it,it-en")
    
    # 2. 检查实际数据集中的语言
    data_dir = "fairseq/models/ZeroTrans/europarl_scripts/mmcr4nlp/europarl/21lingual"
    
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        
        # 提取所有语言对
        language_pairs = set()
        for file in files:
            if file.startswith("train.21langmultiway.europarl-v7."):
                parts = file.split(".")
                if len(parts) >= 4:
                    lang_pair = parts[3]  # 例如: de-en
                    if "-" in lang_pair:
                        language_pairs.add(lang_pair)
        
        print(f"\n📊 数据集中实际可用的语言对:")
        sorted_pairs = sorted(list(language_pairs))
        for pair in sorted_pairs:
            print(f"  {pair}")
        
        print(f"\n总计: {len(sorted_pairs)} 个语言对")
        
        # 3. 检查训练配置中的语言对是否都存在
        config_pairs = ["en-de", "de-en", "en-es", "es-en", "en-it", "it-en"]
        print(f"\n✅ 配置匹配检查:")
        
        missing_pairs = []
        for pair in config_pairs:
            if pair in sorted_pairs:
                print(f"  ✅ {pair}: 存在")
            else:
                print(f"  ❌ {pair}: 不存在")
                missing_pairs.append(pair)
        
        if missing_pairs:
            print(f"\n⚠️  警告: 以下语言对在数据集中不存在:")
            for pair in missing_pairs:
                print(f"    {pair}")
        else:
            print(f"\n🎉 所有配置的语言对都存在于数据集中!")
        
        # 4. 检查数据集中额外的语言对
        extra_pairs = [pair for pair in sorted_pairs if pair not in config_pairs]
        if extra_pairs:
            print(f"\n📈 数据集中还有以下额外的语言对可用:")
            for pair in extra_pairs:
                print(f"    {pair}")
        
        # 5. 检查具体的数据文件大小
        print(f"\n📁 配置语言对的数据文件大小:")
        for pair in config_pairs:
            if pair in sorted_pairs:
                src_lang, tgt_lang = pair.split("-")
                train_src = os.path.join(data_dir, f"train.21langmultiway.europarl-v7.{pair}.{src_lang}")
                train_tgt = os.path.join(data_dir, f"train.21langmultiway.europarl-v7.{pair}.{tgt_lang}")
                
                if os.path.exists(train_src) and os.path.exists(train_tgt):
                    src_size = os.path.getsize(train_src) / (1024*1024)  # MB
                    tgt_size = os.path.getsize(train_tgt) / (1024*1024)  # MB
                    print(f"  {pair}: {src_lang}={src_size:.1f}MB, {tgt_lang}={tgt_size:.1f}MB")
    
    else:
        print(f"❌ 数据目录不存在: {data_dir}")

def check_processed_data_config():
    """检查处理后的数据配置"""
    print(f"\n🔧 检查处理后的数据配置")
    print("=" * 60)
    
    # 检查二进制数据目录
    bin_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    if os.path.exists(bin_dir):
        files = os.listdir(bin_dir)
        
        # 检查训练数据文件
        config_pairs = ["en-de", "de-en", "en-es", "es-en", "en-it", "it-en"]
        
        print("📊 二进制训练数据文件:")
        for pair in config_pairs:
            src_lang, tgt_lang = pair.split("-")
            
            train_src_bin = f"train.{pair}.{src_lang}.bin"
            train_tgt_bin = f"train.{pair}.{tgt_lang}.bin"
            train_src_idx = f"train.{pair}.{src_lang}.idx"
            train_tgt_idx = f"train.{pair}.{tgt_lang}.idx"
            
            src_exists = train_src_bin in files and train_src_idx in files
            tgt_exists = train_tgt_bin in files and train_tgt_idx in files
            
            if src_exists and tgt_exists:
                src_size = os.path.getsize(os.path.join(bin_dir, train_src_bin)) / (1024*1024)
                tgt_size = os.path.getsize(os.path.join(bin_dir, train_tgt_bin)) / (1024*1024)
                print(f"  ✅ {pair}: {src_lang}={src_size:.1f}MB, {tgt_lang}={tgt_size:.1f}MB")
            else:
                print(f"  ❌ {pair}: 缺失文件")
        
        # 检查词典文件
        print(f"\n📚 词典文件:")
        config_langs = ["en", "de", "es", "it"]
        for lang in config_langs:
            dict_file = f"dict.{lang}.txt"
            if dict_file in files:
                dict_size = os.path.getsize(os.path.join(bin_dir, dict_file)) / 1024  # KB
                print(f"  ✅ {lang}: {dict_size:.1f}KB")
            else:
                print(f"  ❌ {lang}: 词典文件不存在")
    
    else:
        print(f"❌ 二进制数据目录不存在: {bin_dir}")

def analyze_data_balance():
    """分析数据平衡性"""
    print(f"\n⚖️  数据平衡性分析")
    print("=" * 60)
    
    data_dir = "fairseq/models/ZeroTrans/europarl_scripts/mmcr4nlp/europarl/21lingual"
    config_pairs = ["en-de", "de-en", "en-es", "es-en", "en-it", "it-en"]
    
    if os.path.exists(data_dir):
        print("📊 各语言对的句子数量:")
        
        for pair in config_pairs:
            src_lang, tgt_lang = pair.split("-")
            train_src = os.path.join(data_dir, f"train.21langmultiway.europarl-v7.{pair}.{src_lang}")
            
            if os.path.exists(train_src):
                try:
                    with open(train_src, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    print(f"  {pair}: {line_count:,} 句子")
                except Exception as e:
                    print(f"  {pair}: 读取失败 - {e}")
            else:
                print(f"  {pair}: 文件不存在")
        
        # 检查数据是否平衡
        print(f"\n💡 数据平衡性评估:")
        print("- 如果所有语言对的句子数量相同，说明数据是平衡的")
        print("- 如果差异很大，可能导致某些语言对训练不充分")
        print("- Europarl数据通常是平衡的，因为来自同一个议会语料库")

def main():
    print("🔍 训练配置和数据集匹配检查")
    print("=" * 60)
    
    # 检查训练配置
    check_training_config()
    
    # 检查处理后的数据
    check_processed_data_config()
    
    # 分析数据平衡性
    analyze_data_balance()
    
    print(f"\n🎯 总结:")
    print("1. 训练配置使用了6个语言对: en-de, de-en, en-es, es-en, en-it, it-en")
    print("2. 这些都是常见的欧洲语言对，适合机器翻译任务")
    print("3. 数据集是Europarl v7，专门用于议会文档翻译")
    print("4. 如果数据匹配正确，问题可能在于:")
    print("   - 模型过拟合")
    print("   - 学习率过高")
    print("   - 数据过于专业化")
    print("   - 需要更好的正则化")

if __name__ == "__main__":
    main() 