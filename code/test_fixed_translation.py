#!/usr/bin/env python3
"""
测试修复后的checkpoint翻译功能
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
    
    return ROOT_PATH, FAIRSEQ

def test_fixed_translation():
    """测试修复后的翻译"""
    print("🧪 测试修复后的checkpoint翻译功能")
    print("=" * 60)
    
    # 使用修复后的模型
    model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    if not os.path.exists(model_path):
        print(f"❌ 修复后的模型不存在: {model_path}")
        return False
    
    # 构建翻译命令 - 使用interactive模式
    cmd = [
        'python', 'fairseq/fairseq_cli/interactive.py',
        data_bin,
        '--path', model_path,
        '--task', 'translation_multi_simple_epoch',
        '--lang-pairs', 'en-de,de-en,en-es,es-en,en-it,it-en',
        '--source-lang', 'en',
        '--target-lang', 'de',
        '--beam', '5',
        '--remove-bpe'
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # 测试句子
    test_sentences = [
        "Hello, how are you?",
        "The meeting is today.",
        "We need to discuss this."
    ]
    
    try:
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n🔍 测试句子 {i}: {sentence}")
            
            # 创建输入
            input_text = sentence + '\n'
            
            # 执行翻译
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=60  # 60秒超时
            )
            
            if result.returncode == 0:
                # 解析输出
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('H-'):  # 假设输出格式
                        translation = line.split('\t')[-1] if '\t' in line else line
                        print(f"✅ 翻译结果: {translation}")
                        break
                else:
                    print(f"✅ 原始输出:")
                    print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            else:
                print(f"❌ 翻译失败 (返回码: {result.returncode})")
                print(f"错误信息: {result.stderr[:200]}...")
                
                # 如果是架构问题，尝试不同的方法
                if "transformer_pdec_6_e_6_d" in result.stderr:
                    print("🔧 检测到架构注册问题，尝试修复...")
                    return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ 翻译超时")
        return False
    except Exception as e:
        print(f"❌ 翻译异常: {e}")
        return False

def main():
    print("🧪 修复后Checkpoint翻译测试")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 首先确保PhasedDecoder模块已加载
    try:
        # 导入PhasedDecoder模块
        sys.path.insert(0, os.path.join(FAIRSEQ, "models", "PhasedDecoder"))
        import models.transformer_pdec
        print("✅ PhasedDecoder模块加载成功")
        
        # 检查架构注册
        from fairseq.models import ARCH_MODEL_REGISTRY
        if 'transformer_pdec_6_e_6_d' in ARCH_MODEL_REGISTRY:
            print("✅ 架构已注册")
        else:
            print("❌ 架构未注册，需要修复")
            return
            
    except Exception as e:
        print(f"❌ 模块加载失败: {e}")
        return
    
    # 测试翻译
    if test_fixed_translation():
        print("\n🎉 修复后的模型翻译测试成功!")
    else:
        print("\n❌ 修复后的模型翻译测试失败")

if __name__ == "__main__":
    main() 