#!/usr/bin/env python3
"""
翻译质量测试脚本
测试不同模型在各语言对上的翻译效果
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def test_translation(model_path, data_bin, src_lang, tgt_lang, test_sentences):
    """测试翻译质量"""
    print(f"\n🔄 测试 {src_lang}->{tgt_lang} 翻译...")
    
    # 创建临时输入文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        for sentence in test_sentences:
            f.write(sentence + '\n')
        temp_input = f.name
    
    try:
        # 构建翻译命令
        cmd = [
            'python', 'fairseq/fairseq_cli/interactive.py',
            data_bin,
            '--path', model_path,
            '--task', 'translation_multi_simple_epoch',
            '--lang-pairs', 'en-de,de-en,en-es,es-en,en-it,it-en',
            '--source-lang', src_lang,
            '--target-lang', tgt_lang,
            '--beam', '5',
            '--remove-bpe',
            '--buffer-size', '1024',
            '--max-tokens', '4096'
        ]
        
        # 执行翻译
        with open(temp_input, 'r', encoding='utf-8') as input_file:
            result = subprocess.run(
                cmd,
                stdin=input_file,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
        
        if result.returncode == 0:
            # 解析输出
            lines = result.stdout.strip().split('\n')
            translations = []
            
            for line in lines:
                if line.startswith('H-'):
                    # 提取翻译结果
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        translation = parts[2].strip()
                        translations.append(translation)
            
            return translations
        else:
            print(f"❌ 翻译失败: {result.stderr}")
            return []
            
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)

def main():
    print("🔍 PhasedDecoder翻译质量测试")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 定义测试模型
    models = {
        "测试模型(1epoch)": "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "继续训练(5epochs)": "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    }
    
    # 数据路径
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 测试句子
    test_sentences = {
        'en': [
            "Hello, how are you today?",
            "The European Parliament is meeting today.",
            "We need to discuss the economic situation.",
            "Technology is changing our world rapidly.",
            "Climate change is a global challenge."
        ],
        'de': [
            "Guten Tag, wie geht es Ihnen?",
            "Das Europäische Parlament tagt heute.",
            "Wir müssen die Wirtschaftslage besprechen.",
            "Die Technologie verändert unsere Welt schnell.",
            "Der Klimawandel ist eine globale Herausforderung."
        ],
        'es': [
            "Hola, ¿cómo estás hoy?",
            "El Parlamento Europeo se reúne hoy.",
            "Necesitamos discutir la situación económica.",
            "La tecnología está cambiando nuestro mundo rápidamente.",
            "El cambio climático es un desafío global."
        ],
        'it': [
            "Ciao, come stai oggi?",
            "Il Parlamento Europeo si riunisce oggi.",
            "Dobbiamo discutere la situazione economica.",
            "La tecnologia sta cambiando il nostro mondo rapidamente.",
            "Il cambiamento climatico è una sfida globale."
        ]
    }
    
    # 语言对
    lang_pairs = [
        ('en', 'de'), ('de', 'en'),
        ('en', 'es'), ('es', 'en'),
        ('en', 'it'), ('it', 'en')
    ]
    
    # 检查模型可用性
    available_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            available_models[name] = path
            print(f"✅ 发现模型: {name}")
        else:
            print(f"❌ 模型不存在: {name}")
    
    if not available_models:
        print("❌ 没有可用的模型")
        return
    
    # 开始测试
    results = {}
    
    for model_name, model_path in available_models.items():
        print(f"\n🎯 测试模型: {model_name}")
        print("-" * 60)
        
        results[model_name] = {}
        
        for src_lang, tgt_lang in lang_pairs:
            lang_pair = f"{src_lang}-{tgt_lang}"
            
            # 获取测试句子
            if src_lang in test_sentences:
                sentences = test_sentences[src_lang]
                
                try:
                    translations = test_translation(
                        model_path, data_bin, src_lang, tgt_lang, sentences
                    )
                    
                    if translations:
                        results[model_name][lang_pair] = {
                            'source': sentences,
                            'translations': translations
                        }
                        
                        print(f"✅ {lang_pair}: 成功翻译 {len(translations)} 句")
                        
                        # 显示前2个翻译示例
                        for i in range(min(2, len(sentences), len(translations))):
                            print(f"   源文: {sentences[i]}")
                            print(f"   译文: {translations[i]}")
                            print()
                    else:
                        print(f"❌ {lang_pair}: 翻译失败")
                        
                except Exception as e:
                    print(f"❌ {lang_pair}: 出错 - {e}")
    
    # 保存结果
    print(f"\n📊 测试总结:")
    print("=" * 80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for lang_pair, data in model_results.items():
            success_rate = len(data['translations']) / len(data['source']) * 100
            print(f"  {lang_pair}: {success_rate:.1f}% 成功率")
    
    # 保存详细结果到文件
    output_file = f"translation_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PhasedDecoder翻译质量测试结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试时间: {datetime.now()}\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"模型: {model_name}\n")
            f.write("-" * 60 + "\n")
            
            for lang_pair, data in model_results.items():
                f.write(f"\n语言对: {lang_pair}\n")
                
                for i, (src, tgt) in enumerate(zip(data['source'], data['translations'])):
                    f.write(f"{i+1}. 源文: {src}\n")
                    f.write(f"   译文: {tgt}\n\n")
    
    print(f"\n📄 详细结果已保存到: {output_file}")
    
    print(f"\n💡 下一步建议:")
    print("1. 查看翻译质量，识别问题模式")
    print("2. 如果发现过拟合，运行优化训练脚本")
    print("3. 比较不同模型的翻译风格和准确性")

if __name__ == "__main__":
    main() 