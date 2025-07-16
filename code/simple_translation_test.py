#!/usr/bin/env python3
"""
简化翻译测试脚本
使用之前成功的推理方法
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

def test_simple_translation():
    """简单翻译测试"""
    print("🔍 简单翻译测试")
    print("=" * 60)
    
    # 测试句子
    test_sentences = [
        "Hello, how are you?",
        "The meeting is today.",
        "We need to discuss this."
    ]
    
    # 模型路径
    models = {
        "测试模型(1epoch)": "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "继续训练(5epochs)": "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    }
    
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在: {model_name}")
            continue
            
        print(f"\n🎯 测试模型: {model_name}")
        print("-" * 40)
        
        # 测试英语到德语翻译
        print("测试 en->de 翻译:")
        
        for sentence in test_sentences[:2]:  # 只测试前2句
            try:
                # 使用generate.py而不是interactive.py
                cmd = [
                    'python', 'fairseq/fairseq_cli/generate.py',
                    data_bin,
                    '--path', model_path,
                    '--task', 'translation_multi_simple_epoch',
                    '--lang-pairs', 'en-de,de-en,en-es,es-en,en-it,it-en',
                    '--source-lang', 'en',
                    '--target-lang', 'de',
                    '--gen-subset', 'test',
                    '--beam', '5',
                    '--max-tokens', '4096',
                    '--quiet',
                    '--remove-bpe'
                ]
                
                # 创建临时输入文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                    f.write(sentence + '\n')
                    temp_input = f.name
                
                # 执行翻译
                with open(temp_input, 'r', encoding='utf-8') as input_file:
                    result = subprocess.run(
                        cmd,
                        stdin=input_file,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        timeout=60  # 60秒超时
                    )
                
                if result.returncode == 0:
                    # 解析输出
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.startswith('H-'):
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                translation = parts[2].strip()
                                print(f"  源文: {sentence}")
                                print(f"  译文: {translation}")
                                break
                else:
                    print(f"  ❌ 翻译失败: {sentence}")
                    if result.stderr:
                        print(f"     错误: {result.stderr[:200]}...")
                
                # 清理临时文件
                if os.path.exists(temp_input):
                    os.unlink(temp_input)
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ 翻译超时: {sentence}")
            except Exception as e:
                print(f"  ❌ 翻译出错: {sentence} - {e}")

def check_model_info():
    """检查模型信息"""
    print("📋 检查模型信息")
    print("=" * 60)
    
    models = {
        "测试模型(1epoch)": "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt",
        "继续训练(5epochs)": "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    }
    
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}")
            print(f"   路径: {path}")
            print(f"   大小: {size_mb:.1f}MB")
        else:
            print(f"❌ {name}: 文件不存在")

def main():
    print("🚀 PhasedDecoder简化翻译测试")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 检查模型
    check_model_info()
    
    # 测试翻译
    test_simple_translation()
    
    print(f"\n💡 总结:")
    print("如果翻译测试失败，可能的原因:")
    print("1. 模型checkpoint损坏或不兼容")
    print("2. fairseq版本问题")
    print("3. 数据预处理问题")
    print("4. 需要重新训练模型")

if __name__ == "__main__":
    main() 