#!/usr/bin/env python3
import os
import re

def fix_utils_file():
    """修复fairseq/utils.py文件中的语法错误"""
    utils_file = os.path.join('fairseq', 'fairseq', 'utils.py')
    
    print(f"开始修复 {utils_file}...")
    
    # 读取文件内容
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_file = utils_file + '.bak'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"原文件已备份为 {backup_file}")
    
    # 使用简单的字符串替换而不是复杂的正则表达式
    problematic_code = """    if os.path.isfile(module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        module_path = os.path.dirname(module_path)
        if module_path != "":
            sys.path.insert(0, module_path)
            else:
            sys.path.insert(0, ".")
                else:"""
                
    fixed_code = """    if os.path.isfile(module_path):
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        module_path = os.path.dirname(module_path)
        if module_path != "":
            sys.path.insert(0, module_path)
        else:
            sys.path.insert(0, ".")
    else:"""
    
    fixed_content = content.replace(problematic_code, fixed_code)
    
    # 写入修复后的内容
    with open(utils_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print(f"修复完成!")

if __name__ == "__main__":
    fix_utils_file() 