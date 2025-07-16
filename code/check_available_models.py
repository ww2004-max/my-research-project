#!/usr/bin/env python3
"""
检查所有可用的模型文件
"""

import torch
from pathlib import Path
import os

def check_model_file(model_path):
    """检查单个模型文件是否可用"""
    try:
        if not Path(model_path).exists():
            return False, "文件不存在"
        
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查基本信息
        has_model = 'model' in checkpoint
        model_params = sum(p.numel() for p in checkpoint['model'].values()) if has_model else 0
        file_size = Path(model_path).stat().st_size / (1024*1024)
        
        return True, {
            'has_model': has_model,
            'model_params': model_params,
            'file_size_mb': file_size,
            'keys': list(checkpoint.keys())
        }
        
    except Exception as e:
        return False, f"加载失败: {str(e)}"

def main():
    """主函数"""
    print("🌟 检查所有可用模型")
    print("=" * 60)
    
    # 要检查的模型路径
    model_paths = [
        "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt",
        "pdec_work/checkpoints/europarl_bidirectional/1/checkpoint_best.pt", 
        "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt",
        "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt.backup",
    ]
    
    available_models = []
    
    for model_path in model_paths:
        print(f"\n🔍 检查: {model_path}")
        
        success, info = check_model_file(model_path)
        
        if success:
            print(f"✅ 可用")
            print(f"   参数量: {info['model_params']:,}")
            print(f"   文件大小: {info['file_size_mb']:.1f} MB")
            print(f"   包含键: {', '.join(info['keys'])}")
            
            available_models.append({
                'path': model_path,
                'name': Path(model_path).parent.parent.name,
                'info': info
            })
        else:
            print(f"❌ 不可用: {info}")
    
    print(f"\n📊 总结:")
    print(f"   总共检查: {len(model_paths)} 个模型")
    print(f"   可用模型: {len(available_models)} 个")
    
    if available_models:
        print(f"\n🎯 可用于多教师蒸馏的模型:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model['name']}")
            print(f"      路径: {model['path']}")
            print(f"      参数: {model['info']['model_params']:,}")
            print(f"      大小: {model['info']['file_size_mb']:.1f} MB")
        
        # 推荐多教师组合
        if len(available_models) >= 2:
            print(f"\n💡 推荐的多教师蒸馏组合:")
            
            # 找出不同的模型（排除备份文件）
            unique_models = []
            seen_names = set()
            for model in available_models:
                if 'backup' not in model['path'] and model['name'] not in seen_names:
                    unique_models.append(model)
                    seen_names.add(model['name'])
            
            if len(unique_models) >= 2:
                print(f"   🎯 组合1: {unique_models[0]['name']} + {unique_models[1]['name']}")
                if len(unique_models) >= 3:
                    print(f"   🎯 组合2: 全部 {len(unique_models)} 个模型")
            else:
                print(f"   ⚠️  只有 {len(unique_models)} 个独特模型，建议单教师蒸馏")
        else:
            print(f"   ⚠️  只有 {len(available_models)} 个可用模型，建议单教师蒸馏")
    
    return available_models

if __name__ == "__main__":
    available_models = main() 