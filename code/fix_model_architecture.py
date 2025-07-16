#!/usr/bin/env python3
"""
修复PhasedDecoder模型架构注册问题
添加缺失的phaseddecoder_iwslt_de_en架构别名
"""

import os
import sys

def setup_environment():
    """设置环境"""
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    
    os.chdir(ROOT_PATH)
    sys.path.insert(0, FAIRSEQ)
    
    return ROOT_PATH, FAIRSEQ

def add_missing_architecture():
    """添加缺失的架构注册"""
    
    # 找到PhasedDecoder的transformer_pdec.py文件
    pdec_files = [
        "fairseq/models/PhasedDecoder/models/transformer_pdec.py",
        "fairseq/fairseq/models/PhasedDecoder/PhasedDecoder/models/transformer_pdec.py",
        "PhasedDecoder/PhasedDecoder/models/transformer_pdec.py"
    ]
    
    target_file = None
    for file_path in pdec_files:
        if os.path.exists(file_path):
            target_file = file_path
            break
    
    if not target_file:
        print("❌ 找不到PhasedDecoder的transformer_pdec.py文件")
        return False
    
    print(f"✅ 找到目标文件: {target_file}")
    
    # 读取文件内容
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经有phaseddecoder_iwslt_de_en架构
    if 'phaseddecoder_iwslt_de_en' in content:
        print("✅ phaseddecoder_iwslt_de_en架构已存在")
        return True
    
    # 添加缺失的架构注册
    additional_code = '''

# 添加缺失的架构别名
@register_model_architecture("transformer_pdec", "phaseddecoder_iwslt_de_en")
def phaseddecoder_iwslt_de_en(args):
    """PhasedDecoder IWSLT德英架构 - 别名"""
    # 使用与transformer_pdec_0_e_12_d_iwslt相同的配置
    args.encoder_layers = getattr(args, "encoder_layers", 0)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)
'''
    
    # 在文件末尾添加新的架构
    new_content = content + additional_code
    
    # 备份原文件
    backup_file = target_file + '.backup'
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已备份原文件到: {backup_file}")
    
    # 写入新内容
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 已添加phaseddecoder_iwslt_de_en架构到: {target_file}")
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    try:
        # 导入必要的模块
        from fairseq import checkpoint_utils
        from fairseq.models import build_model
        
        # 测试加载模型
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"📁 尝试加载模型: {model_path}")
        
        # 尝试加载checkpoint
        state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
        
        print("✅ 成功加载checkpoint到CPU")
        print(f"   模型架构: {state.get('args', {}).get('arch', 'unknown')}")
        print(f"   任务: {state.get('args', {}).get('task', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def main():
    print("🔧 PhasedDecoder架构修复工具")
    print("=" * 60)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 添加缺失的架构
    if add_missing_architecture():
        print("\n✅ 架构修复完成")
        
        # 测试模型加载
        if test_model_loading():
            print("\n🎉 修复成功！现在可以正常加载模型了")
            
            print(f"\n💡 下一步操作:")
            print("1. 重新运行翻译测试: python simple_translation_test.py")
            print("2. 或者运行推理脚本: python pdec_work/inference_simple.py")
            print("3. 如果还有问题，可能需要重新训练模型")
        else:
            print("\n⚠️  架构已修复，但模型加载仍有问题")
            print("可能需要检查模型文件或重新训练")
    else:
        print("\n❌ 架构修复失败")

if __name__ == "__main__":
    main() 