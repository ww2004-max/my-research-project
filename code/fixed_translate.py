#!/usr/bin/env python3
"""
修复版翻译脚本
解决BPE和模型输出问题
"""

import os
import sys
import torch
import argparse

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

def greedy_decode_fixed(model, src_tokens, src_lengths, tgt_dict, max_len=50):
    """修复的贪心解码"""
    device = src_tokens.device
    
    # 编码
    encoder_out = model.encoder(src_tokens, src_lengths)
    
    # 初始化解码 - 从EOS开始而不是BOS
    generated = torch.LongTensor([[tgt_dict.eos()]]).to(device)
    
    print(f"🔍 编码器输出形状: {encoder_out['encoder_out'][0].shape}")
    print(f"🔍 开始解码，初始token: {generated[0].tolist()}")
    
    for step in range(max_len):
        print(f"  步骤 {step+1}: 当前序列 {generated[0].tolist()}")
        
        # 解码当前步
        decoder_out = model.decoder(generated, encoder_out)
        
        # 获取最后一个位置的logits
        logits = decoder_out[0][:, -1, :]  # [batch_size, vocab_size]
        print(f"  Logits形状: {logits.shape}")
        
        # 获取概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 获取top-5概率最高的tokens
        top_probs, top_indices = torch.topk(probs, 5, dim=-1)
        print(f"  Top-5 tokens: {top_indices[0].tolist()}")
        print(f"  Top-5 probs: {top_probs[0].tolist()}")
        print(f"  Top-5 words: {[tgt_dict[idx.item()] for idx in top_indices[0]]}")
        
        # 选择概率最高的token
        next_token = top_indices[:, 0:1]  # 取第一个（概率最高的）
        
        # 添加到生成序列
        generated = torch.cat([generated, next_token], dim=1)
        
        print(f"  选择token: {next_token.item()} ({tgt_dict[next_token.item()]})")
        
        # 如果生成了结束token，停止
        if next_token.item() == tgt_dict.eos():
            print(f"  遇到EOS，停止生成")
            break
    
    return generated

def translate_sentence_fixed(model, src_dict, tgt_dict, sentence, device):
    """修复的翻译函数"""
    print(f"\n🔄 翻译: '{sentence}'")
    
    # 预处理句子
    sentence = sentence.lower().strip()
    
    # Token化
    tokens = src_dict.encode_line(sentence, add_if_not_exist=False)
    print(f"📝 输入tokens: {tokens.tolist()}")
    print(f"📝 输入words: {[src_dict[t.item()] for t in tokens]}")
    
    # 准备输入
    src_tokens = tokens.unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([src_tokens.size(1)]).to(device)
    
    print(f"📝 输入形状: {src_tokens.shape}")
    
    # 执行翻译
    with torch.no_grad():
        generated = greedy_decode_fixed(model, src_tokens, src_lengths, tgt_dict)
    
    # 解码结果
    generated_tokens = generated[0].tolist()
    print(f"📝 完整输出tokens: {generated_tokens}")
    
    # 手动转换为文本（避免BPE问题）
    words = []
    for token in generated_tokens:
        word = tgt_dict[token]
        if word not in ['<s>', '</s>', '<pad>']:
            words.append(word)
    
    # 简单的文本清理
    translation = ' '.join(words)
    translation = translation.replace('<unk>', '[UNK]')
    
    print(f"✅ 翻译结果: '{translation}'")
    return translation

def debug_model_info(model, src_dict, tgt_dict):
    """调试模型信息"""
    print("\n🔍 模型调试信息")
    print("=" * 40)
    
    print(f"📋 模型类型: {type(model)}")
    print(f"📋 编码器层数: {len(model.encoder.layers)}")
    print(f"📋 解码器层数: {len(model.decoder.layers)}")
    print(f"📋 词汇表大小: {len(src_dict)}")
    
    # 检查一些常见词汇
    test_words = ['hello', 'the', 'and', 'you', 'are']
    print(f"\n📝 词汇表测试:")
    for word in test_words:
        if word in src_dict.indices:
            idx = src_dict.indices[word]
            print(f"  '{word}' -> {idx}")
        else:
            print(f"  '{word}' -> 不在词汇表中")

def main():
    print("🚀 修复版翻译测试")
    print("=" * 60)
    
    try:
        # 设置环境
        ROOT_PATH, FAIRSEQ = setup_environment()
        print("✅ 环境设置完成")
        
        # 导入fairseq模块
        from fairseq import checkpoint_utils
        from fairseq.data import Dictionary
        
        print("✅ fairseq模块导入成功")
        
        # 加载模型
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
        data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
        
        print(f"🔍 加载模型: {model_path}")
        models, model_args = checkpoint_utils.load_model_ensemble([model_path])
        model = models[0]
        
        print(f"✅ 模型加载成功")
        
        # 加载字典
        dict_path = os.path.join(data_bin, "dict.en.txt")
        src_dict = Dictionary.load(dict_path)
        tgt_dict = src_dict  # 共享字典
        
        print(f"✅ 字典加载成功，大小: {len(src_dict)}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ 使用设备: {device}")
        
        # 调试模型信息
        debug_model_info(model, src_dict, tgt_dict)
        
        # 测试翻译
        test_sentences = [
            "hello",
            "thank you"
        ]
        
        print("\n" + "="*60)
        print("🌍 开始详细翻译测试")
        print("="*60)
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n【测试 {i}】")
            try:
                translation = translate_sentence_fixed(model, src_dict, tgt_dict, sentence, device)
                print(f"🎯 最终结果: {sentence} -> {translation}")
            except Exception as e:
                print(f"❌ 翻译失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 调试测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 