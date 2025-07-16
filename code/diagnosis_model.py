#!/usr/bin/env python3
"""
模型诊断脚本 - 测试不同的解码策略
"""

import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
import numpy as np
import os

def load_model_and_task():
    """加载模型和任务"""
    print("🔍 加载模型和任务...")
    
    # 设置参数
    checkpoint_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    data_path = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    
    # 加载checkpoint
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
    cfg = state['cfg']
    
    # 设置任务
    task = tasks.setup_task(cfg.task)
    
    # 加载数据
    task.load_dataset('train', combine=False, epoch=1)
    
    # 构建模型
    models, _model_args = checkpoint_utils.load_model_ensemble([checkpoint_path], task=task)
    model = models[0]
    
    # 设置为评估模式
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    return model, task, cfg

def test_different_decoding_strategies(model, task, text="The European Parliament"):
    """测试不同的解码策略"""
    print(f"\n🧪 测试不同解码策略: '{text}'")
    print("=" * 60)
    
    # 编码输入
    src_tokens = task.source_dictionary.encode_line(text, add_if_not_exist=False).long().unsqueeze(0)
    if torch.cuda.is_available():
        src_tokens = src_tokens.cuda()
    
    print(f"📝 输入tokens: {src_tokens[0].tolist()}")
    
    # 1. 贪婪解码 (beam_size=1)
    print("\n【策略1: 贪婪解码】")
    try:
        sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': torch.LongTensor([src_tokens.size(1)])}}
        if torch.cuda.is_available():
            sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
        
        with torch.no_grad():
            translations = task.inference_step(model, sample, prefix_tokens=None)
        
        for i, hypo in enumerate(translations[0][:1]):  # 只取最好的
            tokens = hypo['tokens'].cpu()
            score = hypo['score']
            translation = task.target_dictionary.string(tokens, bpe_symbol='@@ ')
            print(f"   分数: {score:.4f}")
            print(f"   结果: {translation}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 2. 测试不同的beam size
    for beam_size in [3, 5]:
        print(f"\n【策略2: Beam Search (beam_size={beam_size})】")
        try:
            # 修改配置
            task.cfg.generation.beam = beam_size
            sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': torch.LongTensor([src_tokens.size(1)])}}
            if torch.cuda.is_available():
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
            
            with torch.no_grad():
                translations = task.inference_step(model, sample, prefix_tokens=None)
            
            for i, hypo in enumerate(translations[0][:3]):  # 取前3个
                tokens = hypo['tokens'].cpu()
                score = hypo['score']
                translation = task.target_dictionary.string(tokens, bpe_symbol='@@ ')
                print(f"   候选{i+1} 分数: {score:.4f} -> {translation}")
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    # 3. 测试随机采样
    print(f"\n【策略3: 随机采样】")
    try:
        # 直接使用模型前向传播
        with torch.no_grad():
            encoder_out = model.encoder(src_tokens)
            
            # 手动解码，使用采样
            max_len = 50
            tgt_tokens = torch.LongTensor([[task.target_dictionary.eos()]]).cuda() if torch.cuda.is_available() else torch.LongTensor([[task.target_dictionary.eos()]])
            
            for step in range(max_len):
                decoder_out = model.decoder(tgt_tokens, encoder_out)
                logits = decoder_out[0][:, -1, :]  # 最后一个时间步的logits
                
                # 应用温度采样
                temperature = 0.8
                probs = F.softmax(logits / temperature, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, 1)
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                
                if next_token.item() == task.target_dictionary.eos():
                    break
            
            translation = task.target_dictionary.string(tgt_tokens[0], bpe_symbol='@@ ')
            print(f"   采样结果: {translation}")
            
    except Exception as e:
        print(f"   ❌ 错误: {e}")

def analyze_model_weights(model):
    """分析模型权重分布"""
    print("\n🔬 模型权重分析")
    print("=" * 60)
    
    # 检查输出层权重
    if hasattr(model.decoder, 'output_projection'):
        output_weights = model.decoder.output_projection.weight
        print(f"📊 输出层权重形状: {output_weights.shape}")
        print(f"📊 权重均值: {output_weights.mean().item():.6f}")
        print(f"📊 权重标准差: {output_weights.std().item():.6f}")
        print(f"📊 权重最大值: {output_weights.max().item():.6f}")
        print(f"📊 权重最小值: {output_weights.min().item():.6f}")
        
        # 检查是否有异常的权重分布
        if output_weights.std().item() < 0.01:
            print("⚠️  警告: 输出层权重标准差很小，可能存在梯度消失问题")
        
        # 检查特定token的权重
        common_tokens = [2, 3, 27]  # </s>, <unk>, .
        for token_id in common_tokens:
            if token_id < output_weights.shape[0]:
                token_weights = output_weights[token_id]
                print(f"📊 Token {token_id} 权重统计: 均值={token_weights.mean().item():.6f}, 标准差={token_weights.std().item():.6f}")

def test_vocabulary_coverage():
    """测试词汇表覆盖率"""
    print("\n📚 词汇表覆盖率测试")
    print("=" * 60)
    
    # 加载字典
    data_path = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    src_dict_path = os.path.join(data_path, "dict.en.txt")
    tgt_dict_path = os.path.join(data_path, "dict.de.txt")
    
    # 测试常见词汇
    test_words = [
        "the", "and", "of", "to", "a", "in", "is", "it", "you", "that",
        "European", "Parliament", "Commission", "report", "committee",
        "important", "must", "should", "would", "could", "will", "can"
    ]
    
    try:
        from fairseq.data import Dictionary
        src_dict = Dictionary.load(src_dict_path)
        
        print("🔍 英语词汇覆盖率:")
        covered = 0
        for word in test_words:
            token_id = src_dict.index(word)
            if token_id != src_dict.unk():
                print(f"   ✅ '{word}' -> {token_id}")
                covered += 1
            else:
                print(f"   ❌ '{word}' -> <unk>")
        
        print(f"\n📊 覆盖率: {covered}/{len(test_words)} ({covered/len(test_words)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ 无法加载词典: {e}")

def main():
    print("🔬 PhasedDecoder模型诊断")
    print("=" * 60)
    
    try:
        # 加载模型
        model, task, cfg = load_model_and_task()
        print("✅ 模型加载成功")
        
        # 测试词汇表
        test_vocabulary_coverage()
        
        # 分析模型权重
        analyze_model_weights(model)
        
        # 测试不同解码策略
        test_sentences = [
            "The European Parliament",
            "We must consider",
            "This is important",
            "The report shows"
        ]
        
        for sentence in test_sentences:
            test_different_decoding_strategies(model, task, sentence)
        
        print("\n🎉 诊断完成!")
        
    except Exception as e:
        print(f"❌ 诊断过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 