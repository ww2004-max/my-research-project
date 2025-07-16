#!/usr/bin/env python3
"""
检查checkpoint文件的详细内容
"""

import os
import sys
import torch

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def inspect_checkpoint(checkpoint_path):
    """详细检查checkpoint文件"""
    print(f"🔍 检查checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return
    
    try:
        # 加载checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        
        print("📋 Checkpoint基本信息:")
        print(f"  文件大小: {os.path.getsize(checkpoint_path) / (1024*1024):.1f}MB")
        print(f"  主要键: {list(state.keys())}")
        
        # 检查args
        if 'args' in state:
            args = state['args']
            print(f"\n🔧 训练参数 (args):")
            if hasattr(args, 'arch'):
                print(f"  架构: {args.arch}")
            if hasattr(args, 'task'):
                print(f"  任务: {args.task}")
            if hasattr(args, 'lang_pairs'):
                print(f"  语言对: {args.lang_pairs}")
            if hasattr(args, 'encoder_layers'):
                print(f"  编码器层数: {args.encoder_layers}")
            if hasattr(args, 'decoder_layers'):
                print(f"  解码器层数: {args.decoder_layers}")
            
            # 显示所有args属性
            print(f"\n📝 所有args属性:")
            for attr in sorted(dir(args)):
                if not attr.startswith('_'):
                    try:
                        value = getattr(args, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass
        
        # 检查cfg
        if 'cfg' in state:
            cfg = state['cfg']
            print(f"\n⚙️  配置 (cfg):")
            print(f"  类型: {type(cfg)}")
            if hasattr(cfg, 'model'):
                print(f"  模型配置: {cfg.model}")
            if hasattr(cfg, 'task'):
                print(f"  任务配置: {cfg.task}")
        
        # 检查模型状态
        if 'model' in state:
            model_state = state['model']
            print(f"\n🧠 模型状态:")
            print(f"  模型参数数量: {len(model_state)}")
            print(f"  前10个参数键:")
            for i, key in enumerate(list(model_state.keys())[:10]):
                print(f"    {i+1}. {key}: {model_state[key].shape if hasattr(model_state[key], 'shape') else type(model_state[key])}")
        
        # 检查其他信息
        if 'optimizer_history' in state:
            print(f"\n📈 优化器历史: {len(state['optimizer_history'])} 条记录")
        
        if 'lr_scheduler_state' in state:
            print(f"\n📊 学习率调度器状态: {state['lr_scheduler_state']}")
        
        if 'epoch' in state:
            print(f"\n🔄 训练信息:")
            print(f"  轮数: {state.get('epoch', 'N/A')}")
            print(f"  更新步数: {state.get('num_updates', 'N/A')}")
            print(f"  最佳损失: {state.get('best_loss', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 Checkpoint检查工具")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 检查两个模型
    checkpoints = [
        "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    ]
    
    for checkpoint_path in checkpoints:
        inspect_checkpoint(checkpoint_path)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 