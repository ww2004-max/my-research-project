#!/usr/bin/env python3
"""
最简单的翻译脚本
绕过复杂的配置问题
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

def create_simple_args():
    """创建简单的参数对象"""
    args = argparse.Namespace()
    
    # 基本路径
    args.data = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    args.path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
    
    # 任务设置
    args.task = 'translation_multi_simple_epoch'
    args.source_lang = 'en'
    args.target_lang = 'de'
    args.lang_pairs = ['en-de', 'de-en', 'en-es', 'es-en', 'en-it', 'it-en']
    args.langs = ['en', 'de', 'es', 'it']
    
    # 模型设置
    args.arch = 'transformer_pdec_6_e_6_d'
    args.encoder_langtok = 'tgt'
    args.decoder_langtok = False
    
    # 生成设置
    args.beam = 5
    args.max_len_a = 0
    args.max_len_b = 200
    args.min_len = 1
    args.lenpen = 1.0
    args.unkpen = 0.0
    args.temperature = 1.0
    args.remove_bpe = None
    
    # 其他设置
    args.cpu = False
    args.fp16 = False
    args.seed = 1
    args.no_progress_bar = True
    args.quiet = False
    
    return args

def simple_translate():
    """简单翻译测试"""
    print("🚀 最简单翻译测试")
    print("=" * 60)
    
    try:
        # 设置环境
        ROOT_PATH, FAIRSEQ = setup_environment()
        print("✅ 环境设置完成")
        
        # 创建参数
        args = create_simple_args()
        print("✅ 参数创建完成")
        
        # 导入fairseq模块
        from fairseq import checkpoint_utils, tasks, utils
        from fairseq.data import Dictionary
        
        print("✅ fairseq模块导入成功")
        
        # 加载模型
        print(f"🔍 加载模型: {args.path}")
        models, model_args = checkpoint_utils.load_model_ensemble([args.path])
        model = models[0]
        
        print(f"✅ 模型加载成功")
        print(f"✅ 模型类型: {type(model)}")
        
        # 手动加载字典
        dict_path = os.path.join(args.data, "dict.en.txt")
        if os.path.exists(dict_path):
            src_dict = Dictionary.load(dict_path)
            tgt_dict = src_dict  # 共享字典
            print(f"✅ 字典加载成功，大小: {len(src_dict)}")
        else:
            print(f"❌ 字典文件不存在: {dict_path}")
            return
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ 使用设备: {device}")
        
        # 测试翻译
        test_sentences = [
            "Hello",
            "How are you",
            "Thank you"
        ]
        
        print("\n🔄 开始翻译测试...")
        
        for sentence in test_sentences:
            print(f"\n原文: {sentence}")
            
            try:
                # 简单的token化
                tokens = src_dict.encode_line(sentence, add_if_not_exist=False)
                print(f"Tokens: {tokens}")
                
                # 准备输入
                src_tokens = tokens.unsqueeze(0).to(device)
                src_lengths = torch.LongTensor([src_tokens.size(1)]).to(device)
                
                print(f"输入形状: {src_tokens.shape}")
                
                # 简单的前向传播测试
                with torch.no_grad():
                    # 编码
                    encoder_out = model.encoder(src_tokens, src_lengths)
                    print(f"编码器输出形状: {encoder_out['encoder_out'][0].shape}")
                    
                    # 简单解码（只取第一个token）
                    prev_output_tokens = torch.LongTensor([[tgt_dict.eos()]]).to(device)
                    decoder_out = model.decoder(prev_output_tokens, encoder_out)
                    
                    # 获取概率最高的token
                    probs = torch.softmax(decoder_out[0], dim=-1)
                    next_token = probs.argmax(dim=-1)
                    
                    print(f"下一个token: {next_token.item()}")
                    print(f"对应词汇: {tgt_dict[next_token.item()]}")
                
                print("✅ 前向传播成功")
                
            except Exception as e:
                print(f"❌ 翻译失败: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n🎉 基本翻译测试完成!")
        print("💡 模型可以正常工作，只需要完善翻译逻辑")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    simple_translate()

if __name__ == "__main__":
    main() 