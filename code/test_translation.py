#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的翻译模型
"""

import torch
import sys
import os

# 添加fairseq到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'fairseq'))

def test_model_simple():
    """简单测试模型"""
    print("🚀 开始简单模型测试...")
    
    model_dir = "pdec_work/checkpoints/europarl_bidirectional/1"
    data_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    if not os.path.exists(f"{model_dir}/checkpoint_best.pt"):
        print(f"❌ 找不到模型文件: {model_dir}/checkpoint_best.pt")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ 找不到数据目录: {data_dir}")
        return
    
    try:
        from fairseq.models.transformer import TransformerModel
        
        print("📂 正在加载模型...")
        
        # 使用fairseq的内置方法加载模型
        model = TransformerModel.from_pretrained(
            model_dir,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=data_dir,
            bpe='subword_nmt'
        )
        
        print("✅ 模型加载成功!")
        
        # 测试翻译
        test_sentences = [
            "Hello world",
            "How are you today?",
            "The weather is nice",
            "I love machine learning"
        ]
        
        print("\n🎯 开始翻译测试:")
        print("=" * 50)
        
        for sentence in test_sentences:
            try:
                translation = model.translate(sentence)
                print(f"📝 原文: {sentence}")
                print(f"✨ 译文: {translation}")
                print("-" * 30)
            except Exception as e:
                print(f"❌ 翻译失败: {sentence} -> {e}")
        
        print("\n✅ 测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_model_with_hub():
    """使用hub方式测试模型"""
    print("\n🔄 尝试hub方式加载...")
    
    try:
        import fairseq
        
        # 尝试使用hub方式
        model = fairseq.hub_utils.from_pretrained(
            'pdec_work/checkpoints/europarl_bidirectional/1',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin'
        )
        
        print("✅ Hub方式加载成功!")
        
        # 简单测试
        test_text = "Hello world"
        result = model.translate(test_text)
        print(f"📝 测试: {test_text} -> {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hub方式失败: {e}")
        return False

def check_checkpoint_info():
    """检查checkpoint信息"""
    print("\n🔍 检查checkpoint信息...")
    
    model_path = "pdec_work/checkpoints/europarl_bidirectional/1/checkpoint_best.pt"
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("✅ Checkpoint加载成功!")
        print(f"📊 包含的键: {list(checkpoint.keys())}")
        
        if 'cfg' in checkpoint:
            cfg = checkpoint['cfg']
            print(f"📋 模型配置: {cfg}")
            
            if hasattr(cfg, 'model'):
                print(f"🏗️ 模型架构: {cfg.model}")
        
        if 'args' in checkpoint:
            print(f"⚙️ 训练参数: {checkpoint['args']}")
            
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    print("🚀 开始模型测试...")
    
    # 首先检查checkpoint信息
    check_checkpoint_info()
    
    print("\n" + "="*60)
    
    # 尝试简单方式加载
    if not test_model_simple():
        print("\n" + "="*60)
        # 如果失败，尝试hub方式
        test_model_with_hub()

if __name__ == "__main__":
    main() 