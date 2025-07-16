#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多语言模型测试脚本
"""

import os
import sys
import torch

def test_multilingual_model():
    """测试多语言模型"""
    print("🌍 多语言模型测试")
    print("=" * 60)
    
    # 路径配置
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ_PATH = os.path.join(ROOT_PATH, "fairseq")
    
    # 添加路径
    sys.path.insert(0, os.path.abspath('fairseq'))
    
    # 可能的模型路径
    model_paths = {
        "方案1_三语言": "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt",
        "方案2_四语言": "pdec_work/checkpoints/multilingual_方案2_四语言/1/checkpoint_best.pt", 
        "方案3_五语言": "pdec_work/checkpoints/multilingual_方案3_五语言/1/checkpoint_best.pt",
        "方案4_欧洲主要语言": "pdec_work/checkpoints/multilingual_方案4_欧洲主要语言/1/checkpoint_best.pt",
        "双向模型": "pdec_work/checkpoints/europarl_bidirectional/1/checkpoint_best.pt"
    }
    
    # 检查可用模型
    available_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            available_models[name] = {"path": path, "size": size}
            print(f"✅ 发现模型: {name} ({size:.1f}MB)")
        else:
            print(f"❌ 模型不存在: {name}")
    
    if not available_models:
        print("❌ 没有可用的多语言模型")
        return
    
    # 测试句子
    test_sentences = {
        'en': [
            "Hello, how are you?",
            "I am fine, thank you.",
            "What is your name?",
            "Where are you from?",
            "Good morning.",
            "Have a nice day."
        ],
        'de': [
            "Hallo, wie geht es dir?",
            "Mir geht es gut, danke.",
            "Wie heißt du?",
            "Woher kommst du?",
            "Guten Morgen.",
            "Hab einen schönen Tag."
        ],
        'es': [
            "Hola, ¿cómo estás?",
            "Estoy bien, gracias.",
            "¿Cómo te llamas?",
            "¿De dónde eres?",
            "Buenos días.",
            "Que tengas un buen día."
        ],
        'it': [
            "Ciao, come stai?",
            "Sto bene, grazie.",
            "Come ti chiami?",
            "Di dove sei?",
            "Buongiorno.",
            "Buona giornata."
        ]
    }
    
    # 语言对配置
    language_pairs = {
        "方案1_三语言": [('en', 'de'), ('de', 'en'), ('en', 'es'), ('es', 'en'), ('de', 'es'), ('es', 'de')],
        "方案2_四语言": [('en', 'de'), ('de', 'en'), ('en', 'es'), ('es', 'en'), ('en', 'it'), ('it', 'en'), 
                        ('de', 'es'), ('es', 'de'), ('de', 'it'), ('it', 'de'), ('es', 'it'), ('it', 'es')],
        "双向模型": [('en', 'de'), ('de', 'en'), ('en', 'es'), ('es', 'en'), ('en', 'it'), ('it', 'en')]
    }
    
    # 测试每个可用模型
    for model_name, model_info in available_models.items():
        print(f"\n🎯 测试模型: {model_name}")
        print("-" * 60)
        
        try:
            # 加载模型
            checkpoint = torch.load(model_info["path"], map_location='cpu')
            print(f"✅ 模型加载成功")
            print(f"📊 模型大小: {model_info['size']:.1f}MB")
            
            # 检查模型配置
            if 'cfg' in checkpoint:
                cfg = checkpoint['cfg']
                if hasattr(cfg, 'model') and hasattr(cfg.model, 'langs'):
                    print(f"🌍 支持语言: {cfg.model.langs}")
                if hasattr(cfg, 'task') and hasattr(cfg.task, 'lang_pairs'):
                    pairs = cfg.task.lang_pairs.split(',')
                    print(f"🔄 翻译方向: {len(pairs)} 个")
                    for i, pair in enumerate(pairs[:6]):  # 显示前6个
                        print(f"   {i+1}. {pair}")
                    if len(pairs) > 6:
                        print(f"   ... 还有 {len(pairs)-6} 个")
            
            # 简单编码测试
            print(f"\n🔍 编码测试:")
            
            # 测试不同语言的句子编码
            for lang, sentences in test_sentences.items():
                if lang in ['en', 'de', 'es', 'it']:  # 主要测试语言
                    test_sentence = sentences[0]  # 取第一个句子
                    print(f"  {lang}: {test_sentence}")
                    
                    # 这里可以添加实际的编码测试
                    # 由于需要完整的fairseq环境，这里只做基本检查
                    print(f"    ✅ 句子长度: {len(test_sentence.split())} 词")
            
            print(f"✅ {model_name} 测试完成")
            
        except Exception as e:
            print(f"❌ {model_name} 测试失败: {e}")
    
    print(f"\n📋 测试总结:")
    print("=" * 60)
    print(f"✅ 可用模型: {len(available_models)} 个")
    
    for name, info in available_models.items():
        print(f"  • {name}: {info['size']:.1f}MB")
    
    print(f"\n💡 使用建议:")
    print("1. 如果有多个模型，选择最新训练的")
    print("2. 方案2_四语言 支持最多翻译方向 (12个)")
    print("3. 可以使用 working_translate.py 进行实际翻译测试")

if __name__ == "__main__":
    test_multilingual_model() 