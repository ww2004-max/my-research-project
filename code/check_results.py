import os
import torch

# 设置检查点路径
checkpoint_path = 'C:/Users/33491/PycharmProjects/machine/pdec_work/checkpoints/ted_pdec_mini/1/checkpoint_averaged.pt'

print(f"检查合并后的检查点文件: {checkpoint_path}")

# 检查文件是否存在
if not os.path.exists(checkpoint_path):
    print(f"错误: 检查点文件不存在: {checkpoint_path}")
    exit(1)

# 加载检查点
try:
    print("正在加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 打印检查点信息
    print("\n检查点信息:")
    print(f"- 文件大小: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")
    
    # 打印检查点键
    print("- 检查点键:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # 如果有模型参数，打印模型结构
    if 'model' in checkpoint:
        print("\n模型参数:")
        model_params = 0
        for key, param in checkpoint['model'].items():
            if hasattr(param, 'shape'):
                model_params += param.numel()
                if model_params < 1000000:  # 只打印前面的一些参数
                    print(f"  - {key}: {param.shape}")
        print(f"\n总参数数量: {model_params:,}")
    
    print("\n检查点加载成功!")
except Exception as e:
    print(f"加载检查点时出错: {e}") 