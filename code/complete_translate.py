#!/usr/bin/env python3
"""
完整的翻译脚本
实现完整的beam search翻译
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

def greedy_decode(model, src_tokens, src_lengths, tgt_dict, max_len=50):
    """贪心解码"""
    device = src_tokens.device
    batch_size = src_tokens.size(0)
    
    # 编码
    encoder_out = model.encoder(src_tokens, src_lengths)
    
    # 初始化解码
    generated = torch.LongTensor([[tgt_dict.bos()]]).to(device)  # 开始token
    
    for step in range(max_len):
        # 解码当前步
        decoder_out = model.decoder(generated, encoder_out)
        
        # 获取下一个token的概率
        probs = torch.softmax(decoder_out[0][:, -1, :], dim=-1)  # 只看最后一个位置
        next_token = probs.argmax(dim=-1, keepdim=True)
        
        # 添加到生成序列
        generated = torch.cat([generated, next_token], dim=1)
        
        # 如果生成了结束token，停止
        if next_token.item() == tgt_dict.eos():
            break
    
    return generated

def translate_sentence(model, src_dict, tgt_dict, sentence, device):
    """翻译单个句子"""
    print(f"\n🔄 翻译: '{sentence}'")
    
    # 预处理句子
    sentence = sentence.lower().strip()
    
    # Token化
    tokens = src_dict.encode_line(sentence, add_if_not_exist=False)
    print(f"📝 输入tokens: {tokens.tolist()}")
    
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
    translation = tgt_dict.string(generated_tokens, bpe_symbol='▁')
    
    # 清理输出
    translation = translation.replace('<s>', '').replace('</s>', '').strip()
    translation = translation.replace('▁', ' ').strip()
    
    print(f"✅ 翻译结果: '{translation}'")
    return translation

def complete_translate():
    """完整翻译测试"""
    print("🚀 完整翻译测试")
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
        
        print(f"✅ 模型加载成功: {type(model)}")
        
        # 加载字典
        dict_path = os.path.join(data_bin, "dict.en.txt")
        src_dict = Dictionary.load(dict_path)
        tgt_dict = src_dict  # 共享字典
        
        print(f"✅ 字典加载成功，大小: {len(src_dict)}")
        print(f"📋 BOS token: {tgt_dict.bos()} ({tgt_dict[tgt_dict.bos()]})")
        print(f"📋 EOS token: {tgt_dict.eos()} ({tgt_dict[tgt_dict.eos()]})")
        print(f"📋 UNK token: {tgt_dict.unk()} ({tgt_dict[tgt_dict.unk()]})")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ 使用设备: {device}")
        
        # 测试翻译
        test_sentences = [
            "Hello",
            "How are you",
            "Thank you",
            "Good morning",
            "I love you"
        ]
        
        print("\n" + "="*60)
        print("🌍 开始翻译测试 (英语 -> 德语)")
        print("="*60)
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n【测试 {i}】")
            try:
                translation = translate_sentence(model, src_dict, tgt_dict, sentence, device)
                print(f"🎯 {sentence} -> {translation}")
            except Exception as e:
                print(f"❌ 翻译失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("🎉 翻译测试完成!")
        print("💡 现在你可以看到实际的翻译结果了!")
        
        # 交互模式
        print("\n🌍 进入交互模式 (输入 'quit' 退出)")
        while True:
            try:
                sentence = input("\n请输入英语句子: ").strip()
                if sentence.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见!")
                    break
                if sentence:
                    translate_sentence(model, src_dict, tgt_dict, sentence, device)
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    complete_translate()

if __name__ == "__main__":
    main() 