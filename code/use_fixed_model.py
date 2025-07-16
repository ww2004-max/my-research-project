#!/usr/bin/env python3
"""
使用修复后的PhasedDecoder模型进行翻译
支持交互模式和批量翻译
"""

import os
import sys
import subprocess
import tempfile

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    sys.path.insert(0, os.path.join(FAIRSEQ, "models", "PhasedDecoder"))
    
    # 导入必要模块
    import models.transformer_pdec
    import criterions.label_smoothed_cross_entropy_instruction
    
    return ROOT_PATH, FAIRSEQ

def translate_text(text, source_lang='en', target_lang='de', model_path=None):
    """翻译单个文本"""
    if model_path is None:
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 构建翻译命令
    cmd = [
        'python', 'fairseq/fairseq_cli/interactive.py',
        data_bin,
        '--path', model_path,
        '--task', 'translation_multi_simple_epoch',
        '--lang-pairs', 'en-de,de-en,en-es,es-en,en-it,it-en',
        '--source-lang', source_lang,
        '--target-lang', target_lang,
        '--beam', '5',
        '--remove-bpe',
        '--buffer-size', '1'
    ]
    
    try:
        # 执行翻译
        result = subprocess.run(
            cmd,
            input=text + '\n',
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        
        if result.returncode == 0:
            # 解析输出，查找翻译结果
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('H-0'):  # fairseq输出格式
                    # 格式: H-0	-0.123	翻译结果
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        return parts[2].strip()
            
            # 如果没找到标准格式，返回最后一行非空内容
            for line in reversed(lines):
                if line.strip() and not line.startswith(('S-', 'T-', 'D-')):
                    return line.strip()
        
        return f"翻译失败: {result.stderr[:100]}..."
        
    except subprocess.TimeoutExpired:
        return "翻译超时"
    except Exception as e:
        return f"翻译异常: {e}"

def interactive_translation():
    """交互式翻译模式"""
    print("🌍 PhasedDecoder交互式翻译")
    print("=" * 60)
    print("支持的语言: en (英语), de (德语), es (西班牙语), it (意大利语)")
    print("支持的语言对: en-de, de-en, en-es, es-en, en-it, it-en")
    print("输入 'quit' 退出")
    print("=" * 60)
    
    while True:
        try:
            # 获取用户输入
            text = input("\n📝 请输入要翻译的文本: ").strip()
            
            if text.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
            
            if not text:
                continue
            
            # 获取语言对
            source_lang = input("🔤 源语言 (默认en): ").strip() or 'en'
            target_lang = input("🎯 目标语言 (默认de): ").strip() or 'de'
            
            # 验证语言对
            valid_pairs = ['en-de', 'de-en', 'en-es', 'es-en', 'en-it', 'it-en']
            lang_pair = f"{source_lang}-{target_lang}"
            
            if lang_pair not in valid_pairs:
                print(f"❌ 不支持的语言对: {lang_pair}")
                print(f"支持的语言对: {', '.join(valid_pairs)}")
                continue
            
            print(f"\n🔄 翻译中... ({source_lang} -> {target_lang})")
            
            # 执行翻译
            result = translate_text(text, source_lang, target_lang)
            
            print(f"✅ 翻译结果: {result}")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

def batch_translation():
    """批量翻译模式"""
    print("📄 批量翻译模式")
    print("=" * 60)
    
    # 测试句子
    test_sentences = [
        ("Hello, how are you?", "en", "de"),
        ("The meeting is scheduled for today.", "en", "de"),
        ("We need to discuss this project.", "en", "es"),
        ("Thank you for your help.", "en", "it"),
        ("Guten Tag, wie geht es Ihnen?", "de", "en"),
        ("Hola, ¿cómo estás?", "es", "en")
    ]
    
    print("🔄 开始批量翻译...")
    
    for i, (text, src, tgt) in enumerate(test_sentences, 1):
        print(f"\n{i}. {src}->{tgt}: {text}")
        result = translate_text(text, src, tgt)
        print(f"   翻译: {result}")

def main():
    print("🚀 PhasedDecoder翻译工具")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print("✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 检查修复后的模型
    model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    if not os.path.exists(model_path):
        print(f"❌ 修复后的模型不存在: {model_path}")
        print("请先运行 python fix_checkpoint_args.py 修复模型")
        return
    
    print(f"✅ 使用模型: {model_path}")
    
    # 选择模式
    print("\n请选择翻译模式:")
    print("1. 交互式翻译 (推荐)")
    print("2. 批量测试翻译")
    
    choice = input("\n请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        interactive_translation()
    elif choice == '2':
        batch_translation()
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 