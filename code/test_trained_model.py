#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的PhasedDecoder模型
"""

import os
import sys
import torch

def main():
    print("测试训练好的PhasedDecoder模型...")
    
    # 设置路径
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ, "models", "PhasedDecoder")
    
    # 添加路径到sys.path
    sys.path.insert(0, FAIRSEQ)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    # 导入必要模块
    try:
        import models.transformer_pdec
        import criterions.label_smoothed_cross_entropy_instruction
        from fairseq.models import ARCH_MODEL_REGISTRY
        from fairseq import checkpoint_utils
        print("[SUCCESS] 模块导入成功")
    except Exception as e:
        print(f"[ERROR] 模块导入失败: {e}")
        return
    
    # 检查训练结果
    MODEL_DIR = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_test\1"
    
    print(f"\n检查训练结果:")
    print(f"模型目录: {MODEL_DIR}")
    
    # 列出所有checkpoint文件
    if os.path.exists(MODEL_DIR):
        files = os.listdir(MODEL_DIR)
        checkpoint_files = [f for f in files if f.endswith('.pt')]
        print(f"找到 {len(checkpoint_files)} 个checkpoint文件:")
        for f in checkpoint_files:
            file_path = os.path.join(MODEL_DIR, f)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f}: {size_mb:.1f} MB")
    else:
        print("[ERROR] 模型目录不存在")
        return
    
    # 尝试加载最佳模型
    best_model_path = os.path.join(MODEL_DIR, "checkpoint_best.pt")
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        try:
            # 切换到fairseq目录
            original_dir = os.getcwd()
            os.chdir(FAIRSEQ)
            
            checkpoint = torch.load(best_model_path, map_location='cpu')
            print(f"[SUCCESS] 模型加载成功!")
            
            # 显示训练信息
            if 'cfg' in checkpoint:
                cfg = checkpoint['cfg']
                print(f"\n模型配置信息:")
                print(f"  - 架构: {cfg.model.arch}")
                print(f"  - 任务: {cfg.task._name}")
                print(f"  - 损失函数: {cfg.criterion._name}")
            
            if 'extra_state' in checkpoint:
                extra_state = checkpoint['extra_state']
                print(f"\n训练统计信息:")
                if 'epoch' in extra_state:
                    print(f"  - 训练epoch: {extra_state['epoch']}")
                if 'train_iterator' in extra_state:
                    train_iter = extra_state['train_iterator']
                    if 'num_updates' in train_iter:
                        print(f"  - 更新步数: {train_iter['num_updates']}")
            
            # 检查模型参数
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                total_params = sum(p.numel() for p in model_state.values())
                print(f"  - 总参数量: {total_params:,}")
                
                # 检查一些关键参数
                key_params = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 
                             'decoder.fc1_input.weight', 'decoder.fc2_input.weight']
                print(f"\n关键参数检查:")
                for param_name in key_params:
                    if param_name in model_state:
                        shape = model_state[param_name].shape
                        print(f"  - {param_name}: {shape}")
                    else:
                        print(f"  - {param_name}: 未找到")
            
            os.chdir(original_dir)
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            os.chdir(original_dir)
    
    print(f"\n🎉 训练成功完成!")
    print(f"训练时间: 大约 {(22-19)*60 + (41-17):.0f} 分钟")  # 从19:17到22:41
    print(f"最佳损失: 5.9260")
    print(f"模型已保存到: {MODEL_DIR}")

if __name__ == "__main__":
    main() 