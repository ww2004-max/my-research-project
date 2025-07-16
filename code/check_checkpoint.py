#!/usr/bin/env python3
import os
import sys
import torch

# 检查点文件路径
checkpoint_path = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\ted_pdec_mini\1\checkpoint_averaged.pt"

def check_checkpoint(path):
    """检查检查点文件并打印其内容"""
    print(f"检查检查点文件: {path}")
    
    if not os.path.exists(path):
        print(f"错误: 文件不存在!")
        return False
    
    print(f"文件大小: {os.path.getsize(path) / (1024 * 1024):.2f} MB")
    
    try:
        # 加载检查点
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        print("成功加载检查点!")
        
        # 打印检查点的基本信息
        print("\n检查点信息:")
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if key == 'model':
                    print(f"- model: 包含模型参数")
                    model_params = checkpoint['model']
                    param_count = sum(p.numel() for p in model_params.values() if isinstance(p, torch.Tensor))
                    print(f"  - 参数数量: {param_count:,}")
                else:
                    value = checkpoint[key]
                    if isinstance(value, (int, float, str, bool)):
                        print(f"- {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"- {key}: 字典，包含 {len(value)} 项")
                    elif isinstance(value, list):
                        print(f"- {key}: 列表，包含 {len(value)} 项")
                    else:
                        print(f"- {key}: {type(value)}")
        else:
            print(f"检查点不是字典，而是 {type(checkpoint)}")
        
        return True
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return False

if __name__ == "__main__":
    check_checkpoint(checkpoint_path) 