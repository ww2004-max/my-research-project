#!/usr/bin/env python3
import os
import sys
import shutil
import site

def fix_data_utils_import():
    """修复data_utils导入问题"""
    # 源文件路径
    source_file = os.path.join('fairseq', 'fairseq', 'data', 'data_utils.py')
    
    if not os.path.exists(source_file):
        print(f"错误: 源文件 {source_file} 不存在!")
        return False
    
    # 获取site-packages目录
    site_packages = site.getsitepackages()[0]
    target_dir = os.path.join(site_packages, 'fairseq', 'data')
    
    if not os.path.exists(target_dir):
        print(f"创建目录: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
    
    # 目标文件路径
    target_file = os.path.join(target_dir, 'data_utils.py')
    
    # 复制文件
    print(f"复制 {source_file} 到 {target_file}")
    shutil.copy2(source_file, target_file)
    
    # 创建或更新__init__.py文件
    init_file = os.path.join(target_dir, '__init__.py')
    if not os.path.exists(init_file):
        print(f"创建 {init_file}")
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write("from .data_utils import *\n")
    else:
        print(f"更新 {init_file}")
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if "from .data_utils import" not in content:
            with open(init_file, 'a', encoding='utf-8') as f:
                f.write("\nfrom .data_utils import *\n")
    
    print("修复完成!")
    print("尝试导入data_utils...")
    try:
        from fairseq.data import data_utils
        print("导入成功!")
        return True
    except ImportError as e:
        print(f"导入失败: {e}")
        return False

if __name__ == "__main__":
    fix_data_utils_import() 