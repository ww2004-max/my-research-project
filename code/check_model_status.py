#!/usr/bin/env python3
"""
检查当前模型训练状态
"""
import torch
import os
from datetime import datetime

def check_model_status():
    """检查模型训练状态"""
    print("🔍 检查模型训练状态")
    print("=" * 60)
    
    # 检查点路径
    checkpoint_dir = r"pdec_work\checkpoints\multilingual_方案1_三语言\1"
    
    if not os.path.exists(checkpoint_dir):
        print("❌ 检查点目录不存在")
        return
    
    # 检查文件
    files = os.listdir(checkpoint_dir)
    print(f"📁 检查点文件:")
    for f in files:
        if f.endswith('.pt'):
            size = os.path.getsize(os.path.join(checkpoint_dir, f)) / (1024**3)
            print(f"  {f}: {size:.1f}GB")
    
    # 加载最佳检查点
    best_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    if os.path.exists(best_checkpoint):
        print(f"\n📊 分析最佳检查点...")
        try:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            
            # 基本信息
            extra_state = checkpoint.get('extra_state', {})
            
            print(f"训练轮数: {extra_state.get('epoch', 'N/A')}")
            print(f"更新步数: {extra_state.get('num_updates', 'N/A')}")
            print(f"最佳损失: {extra_state.get('best_loss', 'N/A')}")
            
            # 学习率信息
            if 'lr_scheduler' in extra_state:
                lr_info = extra_state['lr_scheduler']
                if isinstance(lr_info, dict) and 'lr' in lr_info:
                    print(f"当前学习率: {lr_info['lr']}")
            
            # 优化器信息
            if 'optimizer' in checkpoint:
                opt_state = checkpoint['optimizer']
                if 'state' in opt_state:
                    print(f"优化器状态: 已保存")
            
            # 模型参数统计
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
                print(f"模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
            
            print(f"\n⏰ 检查点创建时间: {datetime.fromtimestamp(os.path.getmtime(best_checkpoint))}")
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
    
    # 检查训练进程
    print(f"\n🔄 检查训练进程...")
    import psutil
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'fairseq' in cmdline or 'train' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cmd': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_info'].rss / (1024**3) if proc.info['memory_info'] else 0
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if python_processes:
        print("✅ 发现训练进程:")
        for proc in python_processes:
            print(f"  PID {proc['pid']}: CPU {proc['cpu']:.1f}%, 内存 {proc['memory']:.1f}GB")
            print(f"    命令: {proc['cmd']}")
    else:
        print("❌ 未发现活跃的训练进程")

if __name__ == "__main__":
    check_model_status() 