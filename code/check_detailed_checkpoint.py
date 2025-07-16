#!/usr/bin/env python3
import torch
import json

RESTORE_FILE = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_continue_5epochs\checkpoint_best.pt"

def check_detailed_checkpoint():
    """详细检查最佳checkpoint"""
    checkpoint_path = RESTORE_FILE
    
    print(f"🔍 详细检查最佳checkpoint: {checkpoint_path}")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 检查extra_state的详细信息
        if 'extra_state' in ckpt:
            extra = ckpt['extra_state']
            print("\n📊 Extra State详细信息:")
            
            # 训练迭代器信息
            if 'train_iterator' in extra:
                train_iter = extra['train_iterator']
                print(f"  🔄 训练迭代器:")
                print(f"    - 当前epoch: {train_iter.get('epoch', 'N/A')}")
                print(f"    - epoch内迭代: {train_iter.get('iterations_in_epoch', 'N/A')}")
                print(f"    - 版本: {train_iter.get('version', 'N/A')}")
                print(f"    - 是否shuffle: {train_iter.get('shuffle', 'N/A')}")
            
            # 最佳指标信息
            if 'best' in extra:
                best = extra['best']
                print(f"  🏆 最佳指标: {best}")
            
            # 验证损失
            if 'val_loss' in extra:
                val_loss = extra['val_loss']
                print(f"  📉 验证损失: {val_loss}")
            
            # 指标信息
            if 'metrics' in extra:
                metrics = extra['metrics']
                print(f"  📈 指标信息:")
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        print(f"    - {key}: {value}")
                else:
                    print(f"    - {metrics}")
            
            # 训练时间
            if 'previous_training_time' in extra:
                training_time = extra['previous_training_time']
                print(f"  ⏱️ 之前训练时间: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")
        
        # 检查优化器历史
        if 'optimizer_history' in ckpt and ckpt['optimizer_history']:
            opt_history = ckpt['optimizer_history']
            print(f"\n🔧 优化器历史 (共{len(opt_history)}个状态):")
            
            # 显示最后一个优化器状态
            if opt_history:
                last_opt = opt_history[-1]
                print(f"  📋 最后优化器状态键: {list(last_opt.keys())}")
                
                if 'lr' in last_opt:
                    print(f"  📊 当前学习率: {last_opt['lr']}")
                if 'num_updates' in last_opt:
                    print(f"  🔢 更新步数: {last_opt['num_updates']}")
                if 'epoch' in last_opt:
                    print(f"  📅 Epoch: {last_opt['epoch']}")
        
        # 检查args
        if 'args' in ckpt:
            args = ckpt['args']
            print(f"\n⚙️ 训练参数:")
            important_args = ['max_epoch', 'lr', 'max_tokens', 'update_freq', 'save_interval']
            for arg in important_args:
                if hasattr(args, arg):
                    print(f"  - {arg}: {getattr(args, arg)}")
        
        print("\n✅ 详细检查完成")
        
    except Exception as e:
        print(f"❌ 检查出错: {e}")

if __name__ == "__main__":
    check_detailed_checkpoint() 