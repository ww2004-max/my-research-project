#!/usr/bin/env python3
"""
修复checkpoint中的args字段
从cfg中提取配置并重建args
"""

import os
import sys
import torch
import argparse

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def namespace_to_args(namespace_obj):
    """将Namespace对象转换为args"""
    if hasattr(namespace_obj, '__dict__'):
        return argparse.Namespace(**namespace_obj.__dict__)
    return namespace_obj

def fix_checkpoint_args(checkpoint_path, output_path=None):
    """修复checkpoint中的args字段"""
    print(f"🔧 修复checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return False
    
    try:
        # 加载checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查当前状态
        print(f"📋 当前args状态: {type(state.get('args'))}")
        print(f"📋 当前cfg状态: {type(state.get('cfg'))}")
        
        if state.get('args') is not None:
            print("✅ args已存在，无需修复")
            return True
        
        if 'cfg' not in state:
            print("❌ 没有cfg配置，无法修复")
            return False
        
        cfg = state['cfg']
        
        # 从cfg中提取模型配置
        if 'model' in cfg and hasattr(cfg['model'], '__dict__'):
            model_args = namespace_to_args(cfg['model'])
            print(f"✅ 从cfg.model提取args: {model_args.arch}")
            
            # 设置args
            state['args'] = model_args
            
            # 确保关键字段存在
            if not hasattr(model_args, 'task'):
                model_args.task = 'translation_multi_simple_epoch'
            if not hasattr(model_args, 'arch'):
                model_args.arch = 'transformer_pdec_6_e_6_d'
            
            print(f"🔧 修复后的关键配置:")
            print(f"  架构: {getattr(model_args, 'arch', 'N/A')}")
            print(f"  任务: {getattr(model_args, 'task', 'N/A')}")
            print(f"  语言对: {getattr(model_args, 'lang_pairs', 'N/A')}")
            
            # 保存修复后的checkpoint
            if output_path is None:
                output_path = checkpoint_path.replace('.pt', '_fixed.pt')
            
            # 备份原文件
            backup_path = checkpoint_path + '.backup'
            if not os.path.exists(backup_path):
                torch.save(torch.load(checkpoint_path, map_location='cpu'), backup_path)
                print(f"📁 已备份原文件: {backup_path}")
            
            # 保存修复后的文件
            torch.save(state, output_path)
            print(f"💾 修复后的checkpoint已保存: {output_path}")
            
            return True
        else:
            print("❌ cfg.model格式不正确，无法提取args")
            return False
            
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_checkpoint(checkpoint_path):
    """测试修复后的checkpoint"""
    print(f"\n🧪 测试修复后的checkpoint: {checkpoint_path}")
    
    try:
        # 加载checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        
        args = state.get('args')
        if args is None:
            print("❌ args仍然为None")
            return False
        
        print(f"✅ args类型: {type(args)}")
        print(f"✅ 架构: {getattr(args, 'arch', 'N/A')}")
        print(f"✅ 任务: {getattr(args, 'task', 'N/A')}")
        
        # 尝试导入fairseq并测试模型加载
        try:
            from fairseq import checkpoint_utils
            
            # 测试加载模型
            print("🔍 测试模型加载...")
            models, model_args = checkpoint_utils.load_model_ensemble([checkpoint_path])
            print("✅ 模型加载成功!")
            print(f"✅ 加载的模型数量: {len(models)}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  模型加载测试失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("🔧 Checkpoint Args修复工具")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 修复两个模型
    checkpoints = [
        "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    ]
    
    for checkpoint_path in checkpoints:
        print(f"\n{'='*80}")
        
        if fix_checkpoint_args(checkpoint_path):
            # 测试修复后的文件
            fixed_path = checkpoint_path.replace('.pt', '_fixed.pt')
            if test_fixed_checkpoint(fixed_path):
                print(f"\n🎉 {checkpoint_path} 修复成功!")
                print(f"💡 使用修复后的文件: {fixed_path}")
            else:
                print(f"\n⚠️  {checkpoint_path} 修复完成但测试失败")
        else:
            print(f"\n❌ {checkpoint_path} 修复失败")
    
    print(f"\n💡 下一步操作:")
    print("1. 使用修复后的checkpoint文件进行翻译测试")
    print("2. 更新训练脚本使用修复后的文件")

if __name__ == "__main__":
    main() 