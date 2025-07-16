#!/usr/bin/env python3
"""
简单测试修复后的checkpoint
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

def test_model_loading():
    """测试模型加载"""
    print("🧪 测试修复后的模型加载")
    print("=" * 60)
    
    try:
        # 导入必要模块
        sys.path.insert(0, "fairseq/models/PhasedDecoder")
        import models.transformer_pdec
        import criterions.label_smoothed_cross_entropy_instruction
        
        from fairseq import checkpoint_utils
        from fairseq.models import ARCH_MODEL_REGISTRY
        
        print("✅ 模块导入成功")
        print(f"✅ 已注册架构数量: {len([k for k in ARCH_MODEL_REGISTRY.keys() if 'pdec' in k])}")
        
        # 测试加载修复后的模型
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        print(f"🔍 加载模型: {model_path}")
        
        # 尝试加载模型
        models, model_args = checkpoint_utils.load_model_ensemble([model_path])
        
        print(f"✅ 模型加载成功!")
        print(f"✅ 模型数量: {len(models)}")
        print(f"✅ 模型架构: {getattr(model_args, 'arch', 'N/A')}")
        print(f"✅ 任务: {getattr(model_args, 'task', 'N/A')}")
        
        # 检查模型结构
        model = models[0]
        print(f"✅ 编码器层数: {len(model.encoder.layers) if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers') else 'N/A'}")
        print(f"✅ 解码器层数: {len(model.decoder.layers) if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 简单模型加载测试")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print(f"✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    if test_model_loading():
        print("\n🎉 修复后的模型加载成功!")
        print("💡 现在可以尝试翻译功能了")
    else:
        print("\n❌ 修复后的模型加载失败")
        print("💡 需要进一步调试")

if __name__ == "__main__":
    main() 