#!/usr/bin/env python3
import torch
import os

def check_checkpoint(checkpoint_path):
    """检查checkpoint的详细信息"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return
    
    try:
        print(f"📁 检查checkpoint: {checkpoint_path}")
        print(f"📊 文件大小: {os.path.getsize(checkpoint_path) / 1024**2:.1f} MB")
        
        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n📋 Checkpoint信息:")
        print(f"  - 顶级键: {list(ckpt.keys())}")
        
        # 检查不同可能的键名
        epoch_keys = ['epoch', 'epochs_done', 'epoch_done']
        for key in epoch_keys:
            if key in ckpt:
                print(f"  - {key}: {ckpt[key]}")
                break
        else:
            print(f"  - Epoch: N/A")
            
        # 检查损失相关的键
        loss_keys = ['best_loss', 'loss', 'valid_loss', 'train_loss']
        for key in loss_keys:
            if key in ckpt:
                print(f"  - {key}: {ckpt[key]}")
                break
        else:
            print(f"  - Loss: N/A")
            
        # 检查更新步数
        update_keys = ['num_updates', 'updates', 'step']
        for key in update_keys:
            if key in ckpt:
                print(f"  - {key}: {ckpt[key]}")
                break
        else:
            print(f"  - Updates: N/A")
        
        # 检查优化器状态
        if 'optimizer_history' in ckpt and ckpt['optimizer_history']:
            opt_state = ckpt['optimizer_history'][-1]
            print(f"  - 优化器状态: 存在 ({len(opt_state)} 个键)")
            if 'lr' in opt_state:
                print(f"  - 学习率: {opt_state['lr']}")
        else:
            print(f"  - 优化器状态: 不存在")
            
        # 检查模型参数
        if 'model' in ckpt:
            model_params = sum(p.numel() for p in ckpt['model'].values() if isinstance(p, torch.Tensor))
            print(f"  - 模型参数数量: {model_params:,}")
            
        # 检查extra_state中的训练信息
        if 'extra_state' in ckpt and ckpt['extra_state']:
            extra = ckpt['extra_state']
            print(f"  - extra_state键: {list(extra.keys()) if isinstance(extra, dict) else 'Not dict'}")
            if isinstance(extra, dict):
                for key in ['epoch', 'num_updates', 'best_loss', 'train_iterator']:
                    if key in extra:
                        print(f"  - extra_state.{key}: {extra[key]}")
        
        # 检查task_state
        if 'task_state' in ckpt and ckpt['task_state']:
            task = ckpt['task_state']
            print(f"  - task_state: {task}")
            
        # 检查args参数
        if 'args' in ckpt:
            args = ckpt['args']
            if hasattr(args, 'max_epoch'):
                print(f"  - 最大epoch: {args.max_epoch}")
            if hasattr(args, 'lr'):
                print(f"  - 初始学习率: {args.lr}")
                
        # 检查cfg参数
        if 'cfg' in ckpt:
            cfg = ckpt['cfg']
            print(f"  - cfg类型: {type(cfg)}")
            if hasattr(cfg, 'optimization') and hasattr(cfg.optimization, 'lr'):
                print(f"  - cfg学习率: {cfg.optimization.lr}")
            if hasattr(cfg, 'checkpoint') and hasattr(cfg.checkpoint, 'best_checkpoint_metric'):
                print(f"  - 最佳指标: {cfg.checkpoint.best_checkpoint_metric}")
        
        print("✅ Checkpoint检查完成\n")
        
    except Exception as e:
        print(f"❌ 检查checkpoint时出错: {e}\n")

if __name__ == "__main__":
    # 检查所有相关的checkpoint
    checkpoints = [
        "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint.best_loss_5.4854.pt"
    ]
    
    for ckpt_path in checkpoints:
        check_checkpoint(ckpt_path) 