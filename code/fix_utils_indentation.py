#!/usr/bin/env python3
import os
import re

def fix_utils_indentation():
    """修复utils.py文件中的缩进问题"""
    utils_file = os.path.join('fairseq', 'fairseq', 'utils.py')
    
    print(f"开始修复 {utils_file}...")
    
    # 读取文件内容
    with open(utils_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 备份原文件
    backup_file = utils_file + '.bak2'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"原文件已备份为 {backup_file}")
    
    # 修复缩进问题 - 第477行附近
    for i in range(len(lines)):
        if "if os.path.isfile(module_path):" in lines[i] and lines[i].startswith("        "):
            # 修复缩进
            lines[i] = "    if os.path.isfile(module_path):\n"
            print(f"修复了第 {i+1} 行的缩进")
    
    # 写入修复后的内容
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"修复完成!")

if __name__ == "__main__":
    fix_utils_indentation() 