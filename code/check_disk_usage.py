#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查项目磁盘使用情况 - 找出占用空间大的文件和目录
"""

import os
import shutil

def get_dir_size(path):
    """计算目录大小"""
    total_size = 0
    if not os.path.exists(path):
        return 0
    
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                pass
    return total_size

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def scan_directory(base_path, max_depth=2, min_size_mb=10):
    """扫描目录并找出大文件/目录"""
    print(f"扫描目录: {base_path}")
    print("=" * 80)
    
    items = []
    
    if not os.path.exists(base_path):
        print(f"目录不存在: {base_path}")
        return items
    
    # 扫描直接子目录
    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            size = get_dir_size(item_path)
            size_mb = size / (1024 * 1024)
            
            if size_mb >= min_size_mb:  # 只显示大于指定大小的项目
                items.append((item_path, size, os.path.isdir(item_path)))
    except PermissionError:
        print(f"无权限访问: {base_path}")
        return items
    
    # 按大小排序
    items.sort(key=lambda x: x[1], reverse=True)
    
    for item_path, size, is_dir in items:
        item_type = "目录" if is_dir else "文件"
        print(f"{item_type}: {item_path}")
        print(f"  大小: {format_size(size)}")
        
        # 如果是目录且不是太深，继续扫描
        if is_dir and max_depth > 1:
            print(f"  内容:")
            sub_items = []
            try:
                for sub_item in os.listdir(item_path):
                    sub_path = os.path.join(item_path, sub_item)
                    sub_size = get_dir_size(sub_path)
                    if sub_size / (1024 * 1024) >= min_size_mb:
                        sub_items.append((sub_path, sub_size))
                
                sub_items.sort(key=lambda x: x[1], reverse=True)
                for sub_path, sub_size in sub_items[:5]:  # 只显示前5个
                    print(f"    - {os.path.basename(sub_path)}: {format_size(sub_size)}")
            except PermissionError:
                print(f"    无权限访问子目录")
        print()
    
    return items

def main():
    print("🔍 检查项目磁盘使用情况...")
    print()
    
    # 当前项目根目录
    base_dir = r"C:\Users\33491\PycharmProjects\machine"
    
    # 检查主要目录
    main_dirs = [
        "pdec_work/checkpoints",
        "pdec_work/models", 
        "pdec_work/data-bin",
        "fairseq/models/ZeroTrans/europarl_scripts/build_data",
        "fairseq/models",
        "PhasedDecoder",
        "mosesdecoder-master",
        "moses"
    ]
    
    total_size = 0
    all_large_items = []
    
    for dir_name in main_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        items = scan_directory(dir_path, max_depth=2, min_size_mb=50)  # 50MB以上
        all_large_items.extend(items)
        
        if items:
            dir_total = sum(item[1] for item in items)
            total_size += dir_total
            print(f"📁 {dir_name} 总计: {format_size(dir_total)}")
        else:
            print(f"📁 {dir_name}: 无大文件")
        print("-" * 40)
    
    # 检查一些可能的临时文件
    print("\n🗑️ 检查可能的临时/缓存文件:")
    temp_patterns = [
        "*.tmp", "*.bak", "*.cache", "*temp*", "*log*"
    ]
    
    print("\n📊 总结:")
    print("=" * 80)
    print(f"扫描的大文件/目录总大小: {format_size(total_size)}")
    
    # 按大小排序所有项目
    all_large_items.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 最大的10个项目:")
    for i, (path, size, is_dir) in enumerate(all_large_items[:10]):
        item_type = "目录" if is_dir else "文件"
        print(f"{i+1:2d}. {item_type}: {path}")
        print(f"     大小: {format_size(size)}")
    
    print(f"\n💡 建议:")
    print("1. 检查 'ted_pdec_mini' 目录 - 包含5个约1GB的checkpoint文件 (~5GB)")
    print("2. 检查 'europarl_test' 目录 - 包含3个约1GB的checkpoint文件 (~3GB)")
    print("3. 检查数据文件是否有重复")
    print("4. 清理不需要的日志文件")
    print("5. 删除失败训练的checkpoint")

if __name__ == "__main__":
    main() 