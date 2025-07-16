#!/usr/bin/env python3
"""
测试修复后的模型
支持的翻译方向: de→en, es→en, it→en
"""

import os
import sys
import torch
from fairseq.models.transformer import TransformerModel

def test_fixed_model():
    """测试修复后的模型"""
    print("🧪 测试修复后的模型")
    print("=" * 60)
    
    # 模型路径
    checkpoint_path = "pdec_work/checkpoints/europarl_fixed/1/checkpoint_best.pt"
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请先运行训练: python europarl_fixed_training.py")
        return
    
    print(f"📂 模型路径: {checkpoint_path}")
    print(f"📂 数据路径: {data_bin}")
    
    try:
        # 加载模型
        print("\n🔄 加载模型...")
        
        # 使用fairseq-interactive进行测试
        test_sentences = {
            "de": [
                "Das ist ein Test.",
                "Wir müssen für das Volk arbeiten.",
                "Die Europäische Union ist wichtig.",
                "Deutschland und Frankreich arbeiten zusammen.",
                "Das Parlament hat eine wichtige Entscheidung getroffen."
            ],
            "es": [
                "Esta es una prueba.",
                "Debemos trabajar para el pueblo.",
                "La Unión Europea es importante.",
                "España y Francia trabajan juntos.",
                "El Parlamento ha tomado una decisión importante."
            ],
            "it": [
                "Questo è un test.",
                "Dobbiamo lavorare per il popolo.",
                "L'Unione Europea è importante.",
                "Italia e Francia lavorano insieme.",
                "Il Parlamento ha preso una decisione importante."
            ]
        }
        
        print("\n📝 测试句子:")
        for lang, sentences in test_sentences.items():
            print(f"\n🇪🇺 {lang.upper()} → EN:")
            for i, sentence in enumerate(sentences, 1):
                print(f"  {i}. {sentence}")
        
        print(f"\n💡 使用fairseq-interactive进行翻译测试:")
        print("=" * 60)
        
        for src_lang in ["de", "es", "it"]:
            print(f"\n🔧 {src_lang.upper()} → EN 翻译命令:")
            cmd = f"""fairseq-interactive {data_bin} \\
    --path {checkpoint_path} \\
    --source-lang {src_lang} --target-lang en \\
    --beam 5 --lenpen 0.6 \\
    --tokenizer moses \\
    --bpe sentencepiece"""
            
            print(cmd)
            print(f"\n📝 测试句子 ({src_lang}):")
            for sentence in test_sentences[src_lang]:
                print(f"  {sentence}")
        
        print(f"\n🎯 预期结果:")
        print("- 德语句子应该翻译成合理的英语")
        print("- 西班牙语句子应该翻译成合理的英语")
        print("- 意大利语句子应该翻译成合理的英语")
        print("- 不应该出现大量专有名词")
        print("- 不应该输出全是<unk>")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def create_interactive_test_script():
    """创建交互式测试脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
交互式测试修复后的模型
"""

import subprocess
import os

def test_translation(src_lang, sentences):
    """测试翻译"""
    print(f"\n🔄 测试 {src_lang.upper()} → EN 翻译:")
    print("=" * 40)
    
    checkpoint_path = "pdec_work/checkpoints/europarl_fixed/1/checkpoint_best.pt"
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型不存在: {checkpoint_path}")
        return
    
    # 构建命令
    cmd = [
        "fairseq-interactive", data_bin,
        "--path", checkpoint_path,
        "--source-lang", src_lang,
        "--target-lang", "en",
        "--beam", "5",
        "--lenpen", "0.6"
    ]
    
    try:
        # 创建临时输入文件
        input_file = f"temp_input_{src_lang}.txt"
        with open(input_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        
        # 运行翻译
        with open(input_file, 'r', encoding='utf-8') as f:
            result = subprocess.run(cmd, stdin=f, capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            translations = []
            
            for line in lines:
                if line.startswith('H-'):
                    # 提取翻译结果
                    translation = line.split('\t')[-1] if '\t' in line else line[2:]
                    translations.append(translation.strip())
            
            # 显示结果
            for i, (src, tgt) in enumerate(zip(sentences, translations)):
                print(f"\n📝 句子 {i+1}:")
                print(f"  🇪🇺 {src_lang.upper()}: {src}")
                print(f"  🇬🇧 EN: {tgt}")
        else:
            print(f"❌ 翻译失败: {result.stderr}")
        
        # 清理临时文件
        if os.path.exists(input_file):
            os.remove(input_file)
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")

def main():
    print("🧪 交互式模型测试")
    print("=" * 60)
    
    # 测试句子
    test_data = {
        "de": [
            "Das ist ein Test.",
            "Wir arbeiten zusammen.",
            "Die Entscheidung ist wichtig."
        ],
        "es": [
            "Esta es una prueba.",
            "Trabajamos juntos.",
            "La decisión es importante."
        ],
        "it": [
            "Questo è un test.",
            "Lavoriamo insieme.",
            "La decisione è importante."
        ]
    }
    
    # 测试每种语言
    for lang, sentences in test_data.items():
        test_translation(lang, sentences)
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()
'''
    
    script_file = "interactive_test_fixed.py"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 创建了交互式测试脚本: {script_file}")
    return script_file

def main():
    print("🧪 创建模型测试方案")
    print("=" * 60)
    
    # 基础测试
    test_fixed_model()
    
    print("\n" + "="*60)
    
    # 创建交互式测试脚本
    interactive_script = create_interactive_test_script()
    
    print(f"\n🎯 测试步骤:")
    print("1. 等待训练完成")
    print("2. 运行基础测试:")
    print(f"   python {interactive_script}")
    print("3. 手动测试翻译质量")
    
    print(f"\n💡 如果测试成功，说明:")
    print("- 数据问题已解决")
    print("- 模型训练正常")
    print("- 可以进行正常的翻译任务")

if __name__ == "__main__":
    main() 