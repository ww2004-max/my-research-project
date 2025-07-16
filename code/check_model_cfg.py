#!/usr/bin/env python3
"""
检查模型的cfg配置信息
"""

import torch
import json
from pathlib import Path

def check_model_cfg(checkpoint_path):
    """检查模型的cfg配置"""
    print(f"🔍 检查模型配置: {checkpoint_path}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'cfg' in checkpoint:
            cfg = checkpoint['cfg']
            print(f"📋 CFG配置信息:")
            
            def print_dict(d, indent=0):
                for key, value in d.items():
                    if isinstance(value, dict):
                        print("  " * indent + f"{key}:")
                        print_dict(value, indent + 1)
                    else:
                        print("  " * indent + f"{key}: {value}")
            
            print_dict(cfg)
            
            # 提取关键信息
            model_info = {}
            if 'model' in cfg:
                model_cfg = cfg['model']
                model_info = {
                    'arch': model_cfg.get('_name', 'unknown'),
                    'encoder_embed_dim': model_cfg.get('encoder_embed_dim', 'unknown'),
                    'encoder_layers': model_cfg.get('encoder_layers', 'unknown'),
                    'decoder_embed_dim': model_cfg.get('decoder_embed_dim', 'unknown'),
                    'decoder_layers': model_cfg.get('decoder_layers', 'unknown'),
                    'encoder_attention_heads': model_cfg.get('encoder_attention_heads', 'unknown'),
                    'decoder_attention_heads': model_cfg.get('decoder_attention_heads', 'unknown'),
                    'encoder_ffn_embed_dim': model_cfg.get('encoder_ffn_embed_dim', 'unknown'),
                    'decoder_ffn_embed_dim': model_cfg.get('decoder_ffn_embed_dim', 'unknown'),
                }
            
            print(f"\n🎯 关键模型参数:")
            for key, value in model_info.items():
                print(f"   {key}: {value}")
                
            return model_info
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return None

def main():
    """主函数"""
    print("🌟 模型CFG配置检查器")
    print("=" * 60)
    
    model_path = "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt"
    
    if Path(model_path).exists():
        model_info = check_model_cfg(model_path)
        
        if model_info:
            # 保存配置信息
            with open('model_config_info.json', 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ 配置信息已保存到: model_config_info.json")
    else:
        print(f"❌ 模型文件不存在: {model_path}")

if __name__ == "__main__":
    main() 