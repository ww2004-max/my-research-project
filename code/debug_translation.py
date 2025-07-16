#!/usr/bin/env python3
"""
调试翻译问题的详细脚本
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

def test_single_translation():
    """测试单个翻译"""
    print("🔍 详细翻译调试")
    print("=" * 60)
    
    model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt"
    data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 构建翻译命令
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
        '--remove-bpe'
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # 创建测试输入
    test_sentence = "Hello, how are you?"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_sentence + '\n')
        temp_input = f.name
    
    try:
        # 执行翻译并显示完整输出
        with open(temp_input, 'r', encoding='utf-8') as input_file:
            result = subprocess.run(
                cmd,
                stdin=input_file,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
        
        print(f"返回码: {result.returncode}")
        print(f"\n标准输出:")
        print(result.stdout)
        print(f"\n标准错误:")
        print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 翻译成功!")
        else:
            print(f"\n❌ 翻译失败，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"❌ 执行异常: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_input):
            os.unlink(temp_input)

def main():
    print("🐛 PhasedDecoder翻译调试工具")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    test_single_translation()

if __name__ == "__main__":
    main() 