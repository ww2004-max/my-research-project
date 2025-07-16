#!/usr/bin/env python3
"""
检查缺失的数据文件
"""

import os

def main():
    print("🔍 检查数据文件完整性")
    print("=" * 50)
    
    data_dir = "fairseq/models/ZeroTrans/europarl_scripts/mmcr4nlp/europarl/21lingual"
    
    # 需要的语言对
    required_pairs = ["en-de", "de-en", "en-es", "es-en", "en-it", "it-en"]
    
    print("📋 检查训练配置需要的语言对:")
    
    missing_files = []
    existing_files = []
    
    for pair in required_pairs:
        src_lang, tgt_lang = pair.split("-")
        
        # 检查训练文件
        train_src = os.path.join(data_dir, f"train.21langmultiway.europarl-v7.{pair}.{src_lang}")
        train_tgt = os.path.join(data_dir, f"train.21langmultiway.europarl-v7.{pair}.{tgt_lang}")
        
        src_exists = os.path.exists(train_src)
        tgt_exists = os.path.exists(train_tgt)
        
        if src_exists and tgt_exists:
            print(f"  ✅ {pair}: 完整")
            existing_files.append(pair)
        else:
            print(f"  ❌ {pair}: 缺失")
            if not src_exists:
                print(f"     缺失: {train_src}")
            if not tgt_exists:
                print(f"     缺失: {train_tgt}")
            missing_files.append(pair)
    
    print(f"\n📊 统计:")
    print(f"  完整的语言对: {len(existing_files)}")
    print(f"  缺失的语言对: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  问题分析:")
        print("训练配置要求6个语言对，但数据集中缺失了:")
        for pair in missing_files:
            print(f"  - {pair}")
        
        print(f"\n🔧 可能的解决方案:")
        print("1. 修改训练配置，只使用存在的语言对")
        print("2. 寻找完整的Europarl数据集")
        print("3. 使用其他数据集")
        
        # 建议新的配置
        if existing_files:
            print(f"\n💡 建议的训练配置:")
            existing_langs = set()
            for pair in existing_files:
                src, tgt = pair.split("-")
                existing_langs.add(src)
                existing_langs.add(tgt)
            
            print(f"  --langs {','.join(sorted(existing_langs))}")
            print(f"  --lang-pairs {','.join(existing_files)}")
    else:
        print(f"\n🎉 所有需要的语言对都存在!")
    
    # 检查实际可用的语言对
    print(f"\n📂 检查数据目录中所有可用的语言对:")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        available_pairs = set()
        
        for file in files:
            if file.startswith("train.21langmultiway.europarl-v7.") and not file.endswith(".en"):
                parts = file.split(".")
                if len(parts) >= 4:
                    lang_pair = parts[3]
                    if "-" in lang_pair:
                        available_pairs.add(lang_pair)
        
        print(f"  可用语言对总数: {len(available_pairs)}")
        print(f"  包含en的语言对:")
        en_pairs = [pair for pair in sorted(available_pairs) if "en" in pair]
        for pair in en_pairs:
            print(f"    {pair}")

if __name__ == "__main__":
    main() 