#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查双向数据集完整性
"""

import os

def check_data_files():
    print("🔍 检查双向数据集完整性")
    print("=" * 50)
    
    base_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 要检查的6个语言对
    lang_pairs = [
        "en-de", "de-en",
        "en-es", "es-en", 
        "en-it", "it-en"
    ]
    
    # 检查每个语言对的文件
    all_good = True
    total_train_size = 0
    total_valid_size = 0
    
    for pair in lang_pairs:
        print(f"\n📋 检查语言对: {pair}")
        
        # 训练文件
        src_lang, tgt_lang = pair.split('-')
        train_files = [
            f"train.{pair}.{src_lang}.bin",
            f"train.{pair}.{src_lang}.idx", 
            f"train.{pair}.{tgt_lang}.bin",
            f"train.{pair}.{tgt_lang}.idx"
        ]
        
        # 验证文件
        valid_files = [
            f"valid.{pair}.{src_lang}.bin",
            f"valid.{pair}.{src_lang}.idx",
            f"valid.{pair}.{tgt_lang}.bin", 
            f"valid.{pair}.{tgt_lang}.idx"
        ]
        
        # 检查训练文件
        train_ok = True
        for f in train_files:
            path = os.path.join(base_dir, f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✅ {f}: {size:,} bytes")
                if f.endswith('.bin'):
                    total_train_size += size
            else:
                print(f"  ❌ {f}: 文件不存在")
                train_ok = False
                all_good = False
        
        # 检查验证文件
        valid_ok = True
        for f in valid_files:
            path = os.path.join(base_dir, f)
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✅ {f}: {size:,} bytes")
                if f.endswith('.bin'):
                    total_valid_size += size
            else:
                print(f"  ❌ {f}: 文件不存在")
                valid_ok = False
                all_good = False
        
        if train_ok and valid_ok:
            print(f"  🎯 {pair}: 完整 ✅")
        else:
            print(f"  ⚠️  {pair}: 不完整 ❌")
    
    print("\n" + "=" * 50)
    print("📊 数据集汇总:")
    print(f"总训练数据大小: {total_train_size / (1024*1024):.1f} MB")
    print(f"总验证数据大小: {total_valid_size / (1024*1024):.1f} MB")
    
    if all_good:
        print("🎉 所有6个语言对的数据都完整!")
        print("✅ 可以开始双向训练")
        
        # 估算训练数据量
        print("\n📈 预估训练规模:")
        print("• 6个语言对")
        print("• 每个语言对约18.5万句子对")
        print("• 总计约111万训练样本")
        print("• 预计训练时间: 4-5小时")
        
        return True
    else:
        print("❌ 数据集不完整，无法进行双向训练")
        return False

if __name__ == "__main__":
    check_data_files() 