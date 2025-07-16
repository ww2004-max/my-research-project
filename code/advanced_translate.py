#!/usr/bin/env python3
"""
高级翻译脚本 - 使用多种解码策略
"""

import torch
import sentencepiece as spm
import os
import sys

# 添加fairseq路径
sys.path.insert(0, 'fairseq')

def load_bpe_model():
    """加载BPE模型"""
    bpe_model_path = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl.bpe.model"
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    return sp

def load_fairseq_dict():
    """加载fairseq字典"""
    from fairseq.data import Dictionary
    dict_path = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin/dict.txt"
    dictionary = Dictionary.load(dict_path)
    return dictionary

def create_simple_model():
    """创建简单的翻译模型（不依赖复杂的fairseq配置）"""
    print("🔧 创建简化模型...")
    
    # 加载checkpoint
    checkpoint_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    state = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取模型权重
    model_state = state['model']
    
    # 创建简单的transformer模型
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, model_state):
            super().__init__()
            self.model_state = model_state
            
            # 提取关键参数
            self.embed_dim = 512  # 从checkpoint分析得出
            self.vocab_size = 50005
            
            # 重建嵌入层
            self.encoder_embed = torch.nn.Embedding(self.vocab_size, self.embed_dim)
            self.decoder_embed = torch.nn.Embedding(self.vocab_size, self.embed_dim)
            
            # 输出投影层
            self.output_projection = torch.nn.Linear(self.embed_dim, self.vocab_size, bias=False)
            
            # 加载权重
            self.load_weights()
        
        def load_weights(self):
            """加载预训练权重"""
            # 加载嵌入层权重
            if 'encoder.embed_tokens.weight' in self.model_state:
                self.encoder_embed.weight.data = self.model_state['encoder.embed_tokens.weight']
            if 'decoder.embed_tokens.weight' in self.model_state:
                self.decoder_embed.weight.data = self.model_state['decoder.embed_tokens.weight']
            
            # 加载输出层权重
            if 'decoder.output_projection.weight' in self.model_state:
                self.output_projection.weight.data = self.model_state['decoder.output_projection.weight']
        
        def simple_forward(self, src_tokens, max_len=50):
            """简单的前向传播（仅用于测试）"""
            batch_size = src_tokens.size(0)
            device = src_tokens.device
            
            # 简化的编码（仅使用嵌入）
            src_embed = self.encoder_embed(src_tokens)
            # 这里应该有transformer层，但为了简化，我们直接使用嵌入的平均值作为上下文
            context = src_embed.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]
            
            # 解码
            tgt_tokens = torch.LongTensor([[2]]).to(device)  # 从</s>开始
            
            for step in range(max_len):
                tgt_embed = self.decoder_embed(tgt_tokens)
                # 简化的解码器（实际应该有attention机制）
                decoder_out = tgt_embed + context  # 简单相加
                
                # 输出投影
                logits = self.output_projection(decoder_out[:, -1, :])  # 最后一个位置
                
                # 获取下一个token
                next_token = logits.argmax(dim=-1, keepdim=True)
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
                
                if next_token.item() == 2:  # </s>
                    break
            
            return tgt_tokens
    
    model = SimpleTransformer(model_state)
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    return model

def translate_with_different_strategies(text, src_lang="en", tgt_lang="de"):
    """使用不同策略翻译"""
    print(f"🌍 翻译: '{text}' ({src_lang} -> {tgt_lang})")
    print("=" * 60)
    
    # 加载BPE模型
    sp = load_bpe_model()
    
    # 编码输入
    bpe_pieces = sp.encode_as_pieces(text)
    bpe_ids = sp.encode_as_ids(text)
    
    print(f"📝 BPE编码: {bpe_pieces}")
    print(f"📝 Token IDs: {bpe_ids}")
    
    # 转换为tensor
    src_tokens = torch.LongTensor([bpe_ids + [2]])  # 添加</s>
    if torch.cuda.is_available():
        src_tokens = src_tokens.cuda()
    
    # 加载模型
    try:
        model = create_simple_model()
        
        print(f"\n🎯 策略1: 简单解码")
        with torch.no_grad():
            output_tokens = model.simple_forward(src_tokens, max_len=30)
            output_ids = output_tokens[0].cpu().tolist()
            
            # 解码输出
            # 移除特殊token
            clean_ids = [id for id in output_ids if id not in [0, 1, 2, 3]]  # 移除pad, bos, eos, unk
            
            if clean_ids:
                output_text = sp.decode_ids(clean_ids)
                print(f"   结果: '{output_text}'")
            else:
                print(f"   结果: [空输出]")
            
            print(f"   原始IDs: {output_ids}")
        
        # 策略2: 随机采样
        print(f"\n🎯 策略2: 随机采样解码")
        with torch.no_grad():
            # 这里可以实现温度采样等更复杂的解码策略
            # 由于模型结构简化，我们先用基本方法
            pass
        
    except Exception as e:
        print(f"❌ 翻译失败: {e}")
        import traceback
        traceback.print_exc()

def test_direct_output_analysis():
    """直接分析模型输出分布"""
    print(f"\n🔬 直接分析模型输出")
    print("=" * 60)
    
    try:
        # 加载checkpoint
        checkpoint_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
        state = torch.load(checkpoint_path, map_location='cpu')
        model_state = state['model']
        
        # 分析输出层权重
        if 'decoder.output_projection.weight' in model_state:
            output_weights = model_state['decoder.output_projection.weight']
            
            # 找到权重最大的几个token
            print("📊 输出层权重分析:")
            
            # 计算每个token的权重范数
            token_norms = torch.norm(output_weights, dim=1)
            top_tokens = torch.topk(token_norms, 20)
            
            print("权重范数最大的20个token:")
            sp = load_bpe_model()
            for i, (norm, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
                try:
                    token_text = sp.id_to_piece(token_id.item())
                    print(f"  {i+1:2d}: Token {token_id.item():5d} '{token_text}' (范数: {norm.item():.4f})")
                except:
                    print(f"  {i+1:2d}: Token {token_id.item():5d} [无法解码] (范数: {norm.item():.4f})")
            
            # 检查特定token的权重
            special_tokens = [2, 3, 27]  # </s>, <unk>, 可能是句号
            print(f"\n特殊token权重分析:")
            for token_id in special_tokens:
                if token_id < output_weights.shape[0]:
                    weight_norm = torch.norm(output_weights[token_id]).item()
                    try:
                        token_text = sp.id_to_piece(token_id)
                        print(f"  Token {token_id} '{token_text}': 权重范数 {weight_norm:.4f}")
                    except:
                        print(f"  Token {token_id} [无法解码]: 权重范数 {weight_norm:.4f}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    print("🚀 高级翻译测试")
    print("=" * 60)
    
    # 测试句子
    test_sentences = [
        "The European Parliament",
        "We must consider this proposal",
        "This is very important"
    ]
    
    # 直接分析模型输出
    test_direct_output_analysis()
    
    # 翻译测试
    for sentence in test_sentences:
        translate_with_different_strategies(sentence)
        print()

if __name__ == "__main__":
    main() 