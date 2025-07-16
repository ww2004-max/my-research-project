#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全的磁盘空间清理脚本
"""

import os
import shutil
from datetime import datetime

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

def main():
    print("🧹 磁盘空间清理建议")
    print("=" * 80)
    
    base_dir = r"C:\Users\33491\PycharmProjects\machine"
    
    # 定义清理建议
    cleanup_suggestions = [
        {
            "path": "pdec_work/checkpoints/ted_pdec_mini",
            "reason": "TED数据集的测试训练，可能走错方向的训练结果",
            "safety": "安全删除",
            "priority": "高",
            "action": "delete"
        },
        {
            "path": "pdec_work/checkpoints/europarl_vanilla", 
            "reason": "空的训练目录",
            "safety": "安全删除",
            "priority": "低",
            "action": "delete"
        },
        {
            "path": "pdec_work/checkpoints/europarl_pdec",
            "reason": "空的训练目录", 
            "safety": "安全删除",
            "priority": "低",
            "action": "delete"
        },
        {
            "path": "pdec_work/checkpoints/ted_pdec",
            "reason": "空的训练目录",
            "safety": "安全删除", 
            "priority": "低",
            "action": "delete"
        },
        {
            "path": "fairseq/models/ZeroTrans/europarl_scripts/build_data/bpe_mono",
            "reason": "单语BPE数据，如果不做单语训练可删除",
            "safety": "谨慎删除",
            "priority": "中",
            "action": "backup_then_delete"
        },
        {
            "path": "fairseq/models/ZeroTrans/europarl_scripts/build_data/mono",
            "reason": "原始单语数据，已有BPE版本",
            "safety": "谨慎删除",
            "priority": "中", 
            "action": "backup_then_delete"
        },
        {
            "path": "fairseq/models/ZeroTrans/europarl_scripts/build_data/tokenized",
            "reason": "分词数据，已有BPE版本",
            "safety": "谨慎删除",
            "priority": "中",
            "action": "backup_then_delete"
        }
    ]
    
    # 定义保留项目
    keep_items = [
        {
            "path": "pdec_work/checkpoints/europarl_test",
            "reason": "刚成功训练的PhasedDecoder模型",
            "size": get_dir_size(os.path.join(base_dir, "pdec_work/checkpoints/europarl_test"))
        },
        {
            "path": "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin",
            "reason": "训练数据的二进制文件，训练必需",
            "size": get_dir_size(os.path.join(base_dir, "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"))
        },
        {
            "path": "fairseq/models/ZeroTrans/europarl_scripts/build_data/bpe",
            "reason": "BPE编码的训练数据，当前使用",
            "size": get_dir_size(os.path.join(base_dir, "fairseq/models/ZeroTrans/europarl_scripts/build_data/bpe"))
        }
    ]
    
    total_deletable = 0
    total_keepable = 0
    
    print("🗑️ 建议删除的项目:")
    print("-" * 40)
    
    for i, item in enumerate(cleanup_suggestions, 1):
        path = os.path.join(base_dir, item["path"])
        size = get_dir_size(path)
        total_deletable += size
        
        if size > 0:
            print(f"{i}. {item['path']}")
            print(f"   大小: {format_size(size)}")
            print(f"   原因: {item['reason']}")
            print(f"   安全性: {item['safety']}")
            print(f"   优先级: {item['priority']}")
            print(f"   建议操作: {item['action']}")
            print()
        
    print("✅ 建议保留的重要项目:")
    print("-" * 40)
    
    for item in keep_items:
        if item["size"] > 0:
            print(f"• {item['path']}")
            print(f"  大小: {format_size(item['size'])}")
            print(f"  原因: {item['reason']}")
            print()
            total_keepable += item["size"]
    
    print("📊 空间分析:")
    print("=" * 40)
    print(f"可安全删除: {format_size(total_deletable)}")
    print(f"建议保留: {format_size(total_keepable)}")
    print(f"预计释放空间: {format_size(total_deletable)}")
    
    print("\n🛡️ 安全删除步骤建议:")
    print("=" * 40)
    print("1. 高优先级（安全删除）:")
    print("   - ted_pdec_mini 目录 (~5GB)")
    print("   - 空的训练目录")
    
    print("\n2. 中优先级（谨慎删除）:")
    print("   - 备份后删除单语数据 (~1.8GB)")
    print("   - 删除中间处理数据")
    
    print("\n3. 创建备份（如需要）:")
    backup_dir = os.path.join(base_dir, "backup_" + datetime.now().strftime("%Y%m%d"))
    print(f"   备份目录: {backup_dir}")
    
    print("\n⚠️ 警告:")
    print("- 删除前请确保当前训练已完成")
    print("- 建议先移动到备份目录，确认无问题后再删除")
    print("- 保留 europarl_test 目录（成功训练的模型）")
    print("- 保留 europarl_15-bin 目录（训练数据）")

if __name__ == "__main__":
    main() 