#!/usr/bin/env python3
"""
直接使用fairseq API进行翻译
避免subprocess调用问题
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

def load_model_and_task():
    """加载模型和任务"""
    try:
        from fairseq import checkpoint_utils, tasks
        from fairseq.data import encoders
        
        # 模型和数据路径
        model_path = "pdec_work/checkpoints/europarl_test/1/checkpoint_best_fixed.pt"
        data_bin = "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
        
        print(f"🔍 加载模型: {model_path}")
        
        # 加载模型
        models, model_args = checkpoint_utils.load_model_ensemble([model_path])
        model = models[0]
        
        print(f"✅ 模型加载成功")
        print(f"✅ 架构: {getattr(model_args, 'arch', 'N/A')}")
        
        # 设置任务参数
        model_args.data = data_bin
        model_args.source_lang = 'en'
        model_args.target_lang = 'de'
        
        # 创建任务
        task = tasks.setup_task(model_args)
        
        print(f"✅ 任务创建成功: {task.__class__.__name__}")
        
        # 加载字典
        task.load_dataset('test')  # 加载测试数据集以初始化字典
        
        print(f"✅ 字典加载成功")
        print(f"✅ 源语言字典大小: {len(task.source_dictionary)}")
        print(f"✅ 目标语言字典大小: {len(task.target_dictionary)}")
        
        return model, task, model_args
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def translate_sentence(model, task, sentence, source_lang='en', target_lang='de'):
    """翻译单个句子"""
    try:
        from fairseq import utils
        from fairseq.data import data_utils
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"🔄 翻译: '{sentence}' ({source_lang} -> {target_lang})")
        
        # 编码输入句子
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary
        
        # 分词并转换为索引
        src_tokens = src_dict.encode_line(sentence, add_if_not_exist=False).long()
        
        # 添加语言标记（如果需要）
        # 根据PhasedDecoder的设置，可能需要添加语言标记
        
        # 准备输入
        src_tokens = src_tokens.unsqueeze(0).to(device)  # 添加batch维度
        src_lengths = torch.LongTensor([src_tokens.size(1)]).to(device)
        
        print(f"📝 输入tokens: {src_tokens.shape}")
        
        # 执行翻译
        with torch.no_grad():
            # 使用beam search
            from fairseq.sequence_generator import SequenceGenerator
            
            generator = SequenceGenerator(
                models=[model],
                tgt_dict=tgt_dict,
                beam_size=5,
                max_len_a=0,
                max_len_b=200,
                min_len=1,
                normalize_scores=True,
                len_penalty=1.0,
                unk_penalty=0.0,
                temperature=1.0,
                match_source_len=False,
                no_repeat_ngram_size=0,
            )
            
            # 准备样本
            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
                'target': None,
            }
            
            # 生成翻译
            translations = generator.generate([model], sample)
            
            # 解码结果
            translation = translations[0][0]  # 第一个样本的最佳翻译
            translated_tokens = translation['tokens']
            
            # 转换回文本
            translated_sentence = tgt_dict.string(translated_tokens, bpe_symbol='@@ ')
            
            print(f"✅ 翻译结果: {translated_sentence}")
            return translated_sentence
            
    except Exception as e:
        print(f"❌ 翻译失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🚀 直接API翻译测试")
    print("=" * 80)
    
    try:
        ROOT_PATH, FAIRSEQ = setup_environment()
        print("✅ 环境设置完成")
    except Exception as e:
        print(f"❌ 环境设置失败: {e}")
        return
    
    # 加载模型
    model, task, model_args = load_model_and_task()
    
    if model is None:
        print("❌ 无法加载模型，退出")
        return
    
    # 测试翻译
    test_sentences = [
        "Hello, how are you?",
        "The meeting is today.",
        "Thank you very much."
    ]
    
    print("\n🔄 开始翻译测试...")
    
    for sentence in test_sentences:
        result = translate_sentence(model, task, sentence)
        if result:
            print(f"原文: {sentence}")
            print(f"译文: {result}")
        print("-" * 40)

if __name__ == "__main__":
    main() 