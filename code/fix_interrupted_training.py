#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复被中断的双向训练
"""

import os
import shutil

def fix_interrupted_training():
    print("🔧 修复被中断的双向训练")
    print("=" * 50)
    
    checkpoint_dir = "pdec_work/checkpoints/europarl_bidirectional/1"
    
    # 检查现有文件
    files = os.listdir(checkpoint_dir)
    print(f"📁 目录内容: {files}")
    
    tmp_file = os.path.join(checkpoint_dir, "checkpoint_best.pt.tmp")
    best_file = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    last_file = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    
    if os.path.exists(tmp_file):
        tmp_size = os.path.getsize(tmp_file) / (1024*1024)
        print(f"📄 找到临时文件: checkpoint_best.pt.tmp ({tmp_size:.1f} MB)")
        
        # 检查临时文件是否完整（通常完整的checkpoint约1GB）
        if tmp_size < 500:  # 小于500MB说明不完整
            print("⚠️  临时文件不完整，删除并重新开始训练")
            try:
                os.remove(tmp_file)
                print("✅ 已删除不完整的临时文件")
            except Exception as e:
                print(f"❌ 删除失败: {e}")
        else:
            print("🎯 临时文件可能完整，尝试恢复...")
            try:
                # 尝试重命名为正式文件
                shutil.move(tmp_file, best_file)
                print("✅ 临时文件已恢复为checkpoint_best.pt")
                
                # 同时创建last文件
                shutil.copy(best_file, last_file)
                print("✅ 创建了checkpoint_last.pt")
                
                return True
            except Exception as e:
                print(f"❌ 恢复失败: {e}")
    
    # 如果没有可用的checkpoint，准备重新开始
    print("\n🚀 准备重新开始双向训练...")
    
    # 清理目录
    try:
        for f in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("✅ 已清理checkpoint目录")
    except Exception as e:
        print(f"⚠️  清理时出错: {e}")
    
    return False

def check_disk_space():
    """检查磁盘空间"""
    print("\n💾 检查磁盘空间:")
    
    try:
        import shutil
        # 检查C盘空间
        c_total, c_used, c_free = shutil.disk_usage("C:/")
        c_free_gb = c_free / (1024**3)
        
        # 检查D盘空间  
        d_total, d_used, d_free = shutil.disk_usage("D:/")
        d_free_gb = d_free / (1024**3)
        
        print(f"C盘剩余: {c_free_gb:.2f} GB")
        print(f"D盘剩余: {d_free_gb:.2f} GB")
        
        if c_free_gb > 5 and d_free_gb > 5:
            print("✅ 磁盘空间充足，可以继续训练")
            return True
        else:
            print("⚠️  磁盘空间不足，需要进一步清理")
            return False
            
    except Exception as e:
        print(f"❌ 检查磁盘空间失败: {e}")
        return False

if __name__ == "__main__":
    print("🔧 双向训练修复工具")
    print("=" * 30)
    
    # 检查磁盘空间
    space_ok = check_disk_space()
    
    # 修复训练状态
    recovered = fix_interrupted_training()
    
    print("\n" + "=" * 50)
    if recovered:
        print("🎉 训练状态已恢复！")
        print("📂 模型文件已准备就绪")
        print("🎯 可以直接使用模型或继续训练")
    else:
        print("🚀 准备重新开始双向训练")
        if space_ok:
            print("✅ 磁盘空间充足，可以开始训练")
            print("💡 运行: python europarl_bidirectional_training.py")
        else:
            print("⚠️  请先清理更多磁盘空间") 