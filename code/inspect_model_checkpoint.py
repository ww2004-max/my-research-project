#!/usr/bin/env python3
"""
检查现有模型checkpoint的详细信息
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """检查checkpoint文件的详细信息"""
    print(f"🔍 检查模型: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("📊 Checkpoint基本信息:")
        print(f"   文件大小: {Path(checkpoint_path).stat().st_size / (1024*1024):.1f} MB")
        
        # 检查包含的键
        print(f"\n🔑 Checkpoint包含的键:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"   {key}: dict ({len(checkpoint[key])} items)")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"   {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"   {key}: {type(checkpoint[key])}")
        
        # 检查args (如果存在)
        if 'args' in checkpoint and checkpoint['args'] is not None:
            args = checkpoint['args']
            print(f"\n⚙️  模型配置 (args):")
            if hasattr(args, '__dict__'):
                for attr, value in vars(args).items():
                    if not attr.startswith('_'):
                        print(f"   {attr}: {value}")
            else:
                print(f"   args类型: {type(args)}")
        else:
            print(f"\n⚠️  未找到args配置")
        
        # 检查模型状态字典
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"\n🧠 模型状态字典:")
            print(f"   参数总数: {sum(p.numel() for p in model_state.values()):,}")
            
            # 显示前10个参数的名称和形状
            print(f"   前10个参数:")
            for i, (name, param) in enumerate(model_state.items()):
                if i < 10:
                    print(f"     {name}: {param.shape}")
                else:
                    break
            
            if len(model_state) > 10:
                print(f"   ... 还有 {len(model_state) - 10} 个参数")
        
        # 检查其他信息
        if 'epoch' in checkpoint:
            print(f"\n📈 训练信息:")
            print(f"   Epoch: {checkpoint['epoch']}")
        
        if 'best_loss' in checkpoint:
            print(f"   最佳损失: {checkpoint['best_loss']}")
        
        if 'optimizer' in checkpoint:
            print(f"   优化器状态: 已保存")
        
        if 'lr_scheduler' in checkpoint:
            print(f"   学习率调度器: 已保存")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return None

def main():
    """主函数"""
    print("🌟 模型Checkpoint检查器")
    print("=" * 60)
    
    # 要检查的模型列表
    models_to_check = [
        "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_bidirectional/1/checkpoint_best.pt",
    ]
    
    checkpoints_info = {}
    
    for model_path in models_to_check:
        if Path(model_path).exists():
            print(f"\n" + "="*80)
            checkpoint = inspect_checkpoint(model_path)
            if checkpoint:
                checkpoints_info[model_path] = checkpoint
        else:
            print(f"⚠️  模型文件不存在: {model_path}")
    
    # 保存检查结果
    if checkpoints_info:
        print(f"\n💾 保存检查结果...")
        
        # 创建简化的信息用于保存
        simplified_info = {}
        for path, checkpoint in checkpoints_info.items():
            info = {
                'file_size_mb': Path(path).stat().st_size / (1024*1024),
                'keys': list(checkpoint.keys()),
                'has_args': 'args' in checkpoint and checkpoint['args'] is not None,
                'has_model': 'model' in checkpoint,
                'model_params': sum(p.numel() for p in checkpoint['model'].values()) if 'model' in checkpoint else 0,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_loss': checkpoint.get('best_loss', 'unknown')
            }
            
            # 如果有args，保存一些关键信息
            if info['has_args']:
                args = checkpoint['args']
                if hasattr(args, '__dict__'):
                    info['arch'] = getattr(args, 'arch', 'unknown')
                    info['task'] = getattr(args, 'task', 'unknown')
                    info['encoder_layers'] = getattr(args, 'encoder_layers', 'unknown')
                    info['decoder_layers'] = getattr(args, 'decoder_layers', 'unknown')
                    info['encoder_embed_dim'] = getattr(args, 'encoder_embed_dim', 'unknown')
            
            simplified_info[path] = info
        
        # 保存到JSON文件
        with open('model_inspection_results.json', 'w', encoding='utf-8') as f:
            json.dump(simplified_info, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 检查结果已保存到: model_inspection_results.json")
        
        # 显示总结
        print(f"\n📋 模型总结:")
        for path, info in simplified_info.items():
            model_name = Path(path).parent.parent.name
            print(f"   {model_name}:")
            print(f"     参数量: {info['model_params']:,}")
            print(f"     文件大小: {info['file_size_mb']:.1f} MB")
            if 'arch' in info:
                print(f"     架构: {info['arch']}")
            if 'best_loss' in info and info['best_loss'] != 'unknown':
                print(f"     最佳损失: {info['best_loss']}")

if __name__ == "__main__":
    main() 