#!/usr/bin/env python3
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
    
    print("\n" + "="*50)
    if all_exist:
        print("🎉 所有数据文件都存在，可以直接开始训练!")
        print("\n🚀 运行命令:")
        print("python europarl_fixed_training.py")
    else:
        print("⚠️  部分数据文件缺失，需要重新预处理数据")
        print("\n🔧 解决方案:")
        print("1. 重新运行数据预处理脚本")
        print("2. 或者检查原始数据是否完整")

if __name__ == "__main__":
    main()
