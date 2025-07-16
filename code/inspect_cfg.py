#!/usr/bin/env python3
"""
专门检查checkpoint中cfg配置的详细内容
"""

import os
import sys
import torch
import json

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def inspect_cfg_detailed(checkpoint_path):
    """详细检查cfg配置"""
    print(f"🔍 详细检查cfg: {checkpoint_path}")
    print("=" * 80)
    
    try:
        # 加载checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        
        if 'cfg' in state:
            cfg = state['cfg']
            print(f"📋 CFG类型: {type(cfg)}")
            
            if isinstance(cfg, dict):
                print(f"📝 CFG键: {list(cfg.keys())}")
                
                # 递归打印所有配置
                def print_dict(d, indent=0):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            print("  " * indent + f"{key}:")
                            print_dict(value, indent + 1)
                        else:
                            print("  " * indent + f"{key}: {value}")
                
                print(f"\n📄 完整CFG内容:")
                print_dict(cfg)
                
                # 特别关注模型配置
                if 'model' in cfg:
                    print(f"\n🧠 模型配置详情:")
                    model_cfg = cfg['model']
                    print(f"  类型: {type(model_cfg)}")
                    if isinstance(model_cfg, dict):
                        for key, value in model_cfg.items():
                            print(f"  {key}: {value}")
                
                # 特别关注任务配置
                if 'task' in cfg:
                    print(f"\n📋 任务配置详情:")
                    task_cfg = cfg['task']
                    print(f"  类型: {type(task_cfg)}")
                    if isinstance(task_cfg, dict):
                        for key, value in task_cfg.items():
                            print(f"  {key}: {value}")
            
            # 保存cfg到JSON文件以便查看
            cfg_file = f"cfg_dump_{os.path.basename(checkpoint_path)}.json"
            try:
                with open(cfg_file, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, default=str)
                print(f"\n💾 CFG已保存到: {cfg_file}")
            except Exception as e:
                print(f"⚠️  保存CFG失败: {e}")
        
        else:
            print("❌ 没有找到cfg配置")
        
        # 也检查args
        if 'args' in state:
            args = state['args']
            print(f"\n🔧 Args类型: {type(args)}")
            if args is None:
                print("❌ args为None")
            else:
                print(f"Args内容: {args}")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 CFG详细检查工具")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 只检查第一个模型
    checkpoint_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt"
    inspect_cfg_detailed(checkpoint_path)

if __name__ == "__main__":
    main() 