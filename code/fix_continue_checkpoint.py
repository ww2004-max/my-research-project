#!/usr/bin/env python3
"""
修复继续训练checkpoint的args字段
"""

import torch
import os
from omegaconf import OmegaConf

def fix_continue_checkpoint():
    """修复继续训练的checkpoint"""
    print("🔧 修复继续训练的Checkpoint")
    print("=" * 60)
    
    checkpoint_path = "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best.pt"
    fixed_path = "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best_fixed.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 原始checkpoint不存在: {checkpoint_path}")
        return False
    
    try:
        # 加载checkpoint
        print(f"📂 加载: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu')
        
        print("✅ Checkpoint加载成功")
        print(f"📊 Keys: {list(state.keys())}")
        
        # 检查args字段
        if 'args' not in state or state['args'] is None:
            print("⚠️  args字段缺失或为空，需要修复")
            
            # 从cfg重建args
            if 'cfg' in state and state['cfg'] is not None:
                cfg = state['cfg']
                print("🔍 从cfg重建args...")
                
                # 转换cfg为args格式
                args = OmegaConf.create({})
                
                # 复制关键配置
                if hasattr(cfg, 'model'):
                    args.update(cfg.model)
                if hasattr(cfg, 'task'):
                    args.update(cfg.task)
                if hasattr(cfg, 'dataset'):
                    args.update(cfg.dataset)
                if hasattr(cfg, 'optimization'):
                    args.update(cfg.optimization)
                if hasattr(cfg, 'checkpoint'):
                    args.update(cfg.checkpoint)
                if hasattr(cfg, 'common'):
                    args.update(cfg.common)
                if hasattr(cfg, 'distributed_training'):
                    args.update(cfg.distributed_training)
                
                # 设置args
                state['args'] = args
                print("✅ args字段重建完成")
            else:
                print("❌ cfg字段也不存在，无法修复")
                return False
        else:
            print("✅ args字段存在")
        
        # 检查训练信息
        if 'extra_state' in state:
            extra = state['extra_state']
            print(f"\n📊 训练信息:")
            for key, value in extra.items():
                if isinstance(value, (int, float, str)):
                    print(f"   {key}: {value}")
        
        # 保存修复后的checkpoint
        print(f"\n💾 保存修复后的checkpoint: {fixed_path}")
        torch.save(state, fixed_path)
        print("✅ 保存成功")
        
        # 验证修复结果
        print(f"\n🔍 验证修复结果...")
        test_state = torch.load(fixed_path, map_location='cpu')
        if 'args' in test_state and test_state['args'] is not None:
            print("✅ 修复验证成功")
            return True
        else:
            print("❌ 修复验证失败")
            return False
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_test_script():
    """创建使用修复后checkpoint的测试脚本"""
    print(f"\n📝 创建使用修复checkpoint的测试脚本")
    
    script_content = '''#!/usr/bin/env python3
"""
使用修复后的继续训练checkpoint进行测试
"""

import torch
import sys
import os

# 添加fairseq路径
sys.path.insert(0, 'fairseq')

def test_fixed_continue_model():
    """测试修复后的继续训练模型"""
    print("🏛️ 测试修复后的继续训练模型")
    print("=" * 60)
    
    try:
        from fairseq import checkpoint_utils, tasks
        
        # 使用修复后的checkpoint
        checkpoint_path = "pdec_work/checkpoints/europarl_continue_5epochs/checkpoint_best_fixed.pt"
        data_path = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
        
        print(f"🔍 加载模型: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 修复后的checkpoint不存在: {checkpoint_path}")
            print("请先运行: python fix_continue_checkpoint.py")
            return
        
        # 加载checkpoint
        state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
        cfg = state['cfg']
        
        # 设置任务
        task = tasks.setup_task(cfg.task)
        task.load_dataset('train', combine=False, epoch=1)
        
        # 构建模型
        models, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task)
        model = models[0]
        model.eval()
        
        if torch.cuda.is_available():
            model.cuda()
        
        print("✅ 模型加载成功")
        
        # 测试翻译
        test_sentences = [
            "The European Parliament",
            "We must consider this proposal", 
            "This is very important",
            "The Commission has presented",
            "I would like to thank"
        ]
        
        print(f"\n🧪 翻译测试 (继续训练5轮后的模型):")
        for sentence in test_sentences:
            print(f"\n🔄 翻译: '{sentence}'")
            
            # 编码输入
            src_tokens = task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long().unsqueeze(0)
            if torch.cuda.is_available():
                src_tokens = src_tokens.cuda()
            
            # 创建sample
            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': torch.LongTensor([src_tokens.size(1)])
                }
            }
            if torch.cuda.is_available():
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
            
            # 翻译
            with torch.no_grad():
                translations = task.inference_step(model, sample, prefix_tokens=None)
            
            # 解码结果
            hypo = translations[0][0]  # 取最好的结果
            tokens = hypo['tokens'].cpu()
            score = hypo['score']
            translation = task.target_dictionary.string(tokens, bpe_symbol='@@ ')
            
            print(f"✅ 结果: '{translation}' (分数: {score:.4f})")
        
        print(f"\n🎉 测试完成!")
        print(f"\n💡 对比结果:")
        print("- 如果结果仍然是重复的专有名词，说明模型确实过拟合了")
        print("- 如果结果有改善，说明继续训练有帮助")
        print("- 可以尝试使用更早的checkpoint或调整解码参数")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_continue_model()
'''
    
    with open("test_fixed_continue_model.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 创建了 test_fixed_continue_model.py")

def main():
    print("🔧 修复继续训练的Checkpoint")
    print("=" * 60)
    
    # 修复checkpoint
    if fix_continue_checkpoint():
        print(f"\n🎉 修复成功!")
        
        # 创建测试脚本
        create_fixed_test_script()
        
        print(f"\n💡 下一步:")
        print("1. 运行: python test_fixed_continue_model.py")
        print("2. 比较两个模型的翻译结果")
        print("3. 如果仍有问题，考虑其他解决方案")
    else:
        print(f"\n❌ 修复失败")

if __name__ == "__main__":
    main() 