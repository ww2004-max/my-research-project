#!/usr/bin/env python3
"""
Europarl风格翻译测试
使用模型训练域内的句子进行测试
"""

import os
import sys
import torch

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

def greedy_decode(model, src_tokens, src_lengths, tgt_dict, max_len=50):
    """贪心解码"""
    device = src_tokens.device
    
    # 编码
    encoder_out = model.encoder(src_tokens, src_lengths)
    
    # 初始化解码
    generated = torch.LongTensor([[tgt_dict.eos()]]).to(device)
    
    for step in range(max_len):
        # 解码当前步
        decoder_out = model.decoder(generated, encoder_out)
        
        # 获取最后一个位置的logits
        logits = decoder_out[0][:, -1, :]
        
        # 获取概率分布
        probs = torch.softmax(logits, dim=-1)
        
        # 选择概率最高的token
        next_token = probs.argmax(dim=-1, keepdim=True)
        
        # 添加到生成序列
        generated = torch.cat([generated, next_token], dim=1)
        
        # 如果生成了结束token，停止
        if next_token.item() == tgt_dict.eos():
            break
    
    return generated

def translate_europarl_sentence(model, src_dict, tgt_dict, sentence, device):
    """翻译Europarl风格句子"""
    print(f"\n🔄 翻译: '{sentence}'")
    
    # Token化
    tokens = src_dict.encode_line(sentence, add_if_not_exist=False)
    print(f"📝 输入tokens: {tokens.tolist()}")
    
    # 检查未知词
    unknown_count = (tokens == src_dict.unk()).sum().item()
    if unknown_count > 0:
        print(f"⚠️  包含 {unknown_count} 个未知词")
    
    # 显示token对应的词
    words = [src_dict[t.item()] for t in tokens]
    print(f"📝 输入words: {words}")
    
    # 准备输入
    src_tokens = tokens.unsqueeze(0).to(device)
    src_lengths = torch.LongTensor([src_tokens.size(1)]).to(device)
    
    # 执行翻译
    with torch.no_grad():
        generated = greedy_decode(model, src_tokens, src_lengths, tgt_dict)
    
    # 解码结果
    generated_tokens = generated[0].tolist()
    print(f"📝 输出tokens: {generated_tokens}")
    
    # 转换为文本
    output_words = [tgt_dict[token] for token in generated_tokens]
    print(f"📝 输出words: {output_words}")
    
    # 清理输出
    clean_words = []
    for word in output_words:
        if word not in ['<s>', '</s>', '<pad>']:
            # 处理subword
            if word.startswith('▁'):
                clean_words.append(word[1:])  # 去掉▁前缀
            else:
                clean_words.append(word)
    
    translation = ' '.join(clean_words)
    print(f"✅ 翻译结果: '{translation}'")
    return translation

def main():
    print("🏛️ Europarl风格翻译测试")
    print("=" * 60)
    
    try:
        # 设置环境
        ROOT_PATH, FAIRSEQ = setup_environment()
        print("✅ 环境设置完成")
        
        # 导入fairseq模块
        from fairseq import checkpoint_utils
        from fairseq.data import Dictionary
        
        # 加载模型
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
        data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
        
        print(f"🔍 加载模型: {model_path}")
        models, model_args = checkpoint_utils.load_model_ensemble([model_path])
        model = models[0]
        
        # 加载字典
        dict_path = os.path.join(data_bin, "dict.en.txt")
        src_dict = Dictionary.load(dict_path)
        tgt_dict = src_dict
        
        print(f"✅ 模型和字典加载成功")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Europarl风格的测试句子（政治/议会语言）
        europarl_sentences = [
            "The European Parliament",
            "We must consider this proposal",
            "The Commission has presented",
            "I would like to thank",
            "This is very important",
            "The report shows that",
            "We need to discuss",
            "The committee has decided"
        ]
        
        print("\n" + "="*60)
        print("🏛️ Europarl风格翻译测试 (英语 -> 德语)")
        print("="*60)
        
        for i, sentence in enumerate(europarl_sentences, 1):
            print(f"\n【测试 {i}】")
            try:
                translation = translate_europarl_sentence(model, src_dict, tgt_dict, sentence, device)
                print(f"🎯 {sentence} -> {translation}")
            except Exception as e:
                print(f"❌ 翻译失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 Europarl风格测试完成!")
        print("\n💡 总结:")
        print("- 你的模型是在欧洲议会数据上训练的")
        print("- 它专门用于翻译政治/议会文档")
        print("- 对于日常对话可能效果不好")
        print("- 但在其专业领域内应该表现良好")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 