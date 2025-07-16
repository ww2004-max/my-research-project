#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向Europarl训练脚本 - 支持6个语言对的双向翻译
基于成功的europarl_simple_no_userdir.py，但添加双向语言对
"""

import os
import sys
from datetime import datetime

def main():
    print("🚀 双向Europarl训练（6个语言对）")
    print("============================================================")
    print("📋 使用的语言对: en-de, de-en, en-es, es-en, en-it, it-en")
    print("🎯 支持翻译方向: 英语↔德语, 英语↔西语, 英语↔意语 (双向)")
    print("============================================================")
    
    # 路径配置
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ_PATH = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ_PATH, "models", "PhasedDecoder")
    
    # 添加路径到 Python 路径
    print(f"🔧 添加到Python路径: {FAIRSEQ_PATH}")
    print(f"🔧 添加到Python路径: {PHASEDDECODER_PATH}")
    sys.path.insert(0, FAIRSEQ_PATH)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    # 重要：导入PhasedDecoder模块来注册架构
    try:
        print("🔧 导入PhasedDecoder模块...")
        import models.transformer_pdec
        import criterions.label_smoothed_cross_entropy_instruction
        print("✅ PhasedDecoder模块和criterion加载成功")
    except Exception as e:
        print(f"❌ PhasedDecoder模块加载失败: {e}")
        return
    
    # 验证架构注册
    try:
        from fairseq.models import ARCH_MODEL_REGISTRY
        if 'transformer_pdec_6_e_6_d' not in ARCH_MODEL_REGISTRY:
            print("❌ transformer_pdec_6_e_6_d 架构未注册")
            return
        print("✅ transformer_pdec_6_e_6_d 架构已正确注册")
    except Exception as e:
        print(f"❌ 检查架构注册失败: {e}")
        return
    
    # 验证criterion注册  
    try:
        from fairseq.criterions import CRITERION_REGISTRY
        if 'label_smoothed_cross_entropy_instruction' not in CRITERION_REGISTRY:
            print("❌ criterion未正确注册")
            return
        print("✅ criterion已正确注册")
    except Exception as e:
        print(f"❌ 检查criterion注册失败: {e}")
        return
    
    # 路径配置
    DATA_BIN = os.path.join(ROOT_PATH, "fairseq", "models", "ZeroTrans", "europarl_scripts", "build_data", "europarl_15-bin")
    SAVE_DIR = os.path.join(ROOT_PATH, "pdec_work", "checkpoints", "europarl_bidirectional", "1")
    LOG_DIR = os.path.join(ROOT_PATH, "pdec_work", "logs", "europarl_bidirectional")
    
    # 创建目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"📂 检查点目录: {SAVE_DIR}")
    print(f"📝 日志目录: {LOG_DIR}")
    print(f"📁 工作目录: {FAIRSEQ_PATH}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PHASEDDECODER_PATH};"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    
    # 构建训练命令 - 双向语言对配置
    cmd = [
        sys.executable, "-m", "fairseq_cli.train",
        DATA_BIN,
        "--seed", "0",
        "--fp16",
        "--arch", "transformer_pdec_6_e_6_d",
        "--task", "translation_multi_simple_epoch",
        "--sampling-method", "temperature", 
        "--sampling-temperature", "5",
        "--attention-position-bias", "1",
        "--adaption-flag", "True",
        "--adaption-inner-size", "2048",
        "--adaption-dropout", "0.1",
        "--contrastive-flag", "True",
        "--contrastive-type", "enc",
        "--dim", "512",
        "--mode", "1",
        "--cl-position", "6",
        "--temperature", "1.0",
        # 双向语言配置 - 6个语言对
        "--langs", "en,de,es,it",  # 4种语言
        "--lang-pairs", "en-de,de-en,en-es,es-en,en-it,it-en",  # 6个双向语言对
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy_instruction",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0005",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "500",
        "--max-epoch", "3",  # 训练3个epoch
        "--max-tokens", "1800",  # 稍微减少batch size以适应更多语言对
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "1",
        "--log-interval", "50",
        "--log-format", "simple",
        "--save-dir", SAVE_DIR
    ]
    
    print("\n🚀 开始双向训练...")
    print("⏰ 预计训练时间: 4-5小时 (双倍数据量)")
    print("📊 训练数据: 6个语言对，约111万训练样本")
    
    # 日志文件
    log_file = os.path.join(LOG_DIR, "1.log")
    print(f"📝 日志文件: {log_file}")
    
    # 写入日志头
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== 双向Europarl训练日志（6个语言对） ===\n")
        f.write("语言对: en-de, de-en, en-es, es-en, en-it, it-en\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PhasedDecoder路径: {PHASEDDECODER_PATH}\n")
        f.write("训练数据: 6个双向语言对，约111万训练样本\n")
        f.write("==================================================\n")
        f.flush()
    
    # 直接调用fairseq训练 - 就像原来成功的代码那样
    try:
        from fairseq_cli.train import cli_main
        print("🔧 使用直接API调用方式...")
        
        # 设置sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ["train.py"] + cmd[3:]  # 跳过python -m fairseq_cli.train部分
        
        # 切换到fairseq目录
        original_dir = os.getcwd()
        os.chdir(FAIRSEQ_PATH)
        
        # 直接调用训练
        cli_main()
        print("✅ 双向训练完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return_code = 1
    else:
        return_code = 0
    finally:
        # 恢复环境
        sys.argv = original_argv
        os.chdir(original_dir)
    
    if return_code == 0:
        print("\n🎉 双向训练成功完成!")
        print("\n🎯 现在模型支持以下翻译:")
        print("  • 英语 → 德语/西语/意语")
        print("  • 德语/西语/意语 → 英语")
        print("\n🎯 测试训练完成后可以运行:")
        print("python test_bidirectional_model.py")
    else:
        print(f"\n❌ 训练失败，返回码: {return_code}")
        print(f"📝 请检查日志: {log_file}")

if __name__ == "__main__":
    main() 