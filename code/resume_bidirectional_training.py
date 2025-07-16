#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恢复双向训练 - 从99.9%的地方继续完成最后的保存
"""

import os
import sys
from datetime import datetime

def main():
    print("🔄 恢复双向Europarl训练")
    print("============================================================")
    print("📋 继续训练: en-de, de-en, en-es, es-en, en-it, it-en")
    print("🎯 从第1个epoch的27930步继续")
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
    
    # 路径配置 - 现在在D盘！
    DATA_BIN = os.path.join(ROOT_PATH, "fairseq", "models", "ZeroTrans", "europarl_scripts", "build_data", "europarl_15-bin")
    SAVE_DIR = os.path.join(ROOT_PATH, "pdec_work", "checkpoints", "europarl_bidirectional", "1")  # junction会自动指向D盘
    LOG_DIR = os.path.join(ROOT_PATH, "pdec_work", "logs", "europarl_bidirectional")
    
    # 确保目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"📂 检查点目录: {SAVE_DIR} (实际在D盘)")
    print(f"📝 日志目录: {LOG_DIR}")
    print(f"📁 工作目录: {FAIRSEQ_PATH}")
    
    # 检查是否存在临时checkpoint文件
    temp_checkpoint = os.path.join(SAVE_DIR, "checkpoint_last.pt")
    if os.path.exists(temp_checkpoint):
        print(f"✅ 找到临时checkpoint: {temp_checkpoint}")
    else:
        print("⚠️  没有找到临时checkpoint，将从头开始")
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PHASEDDECODER_PATH};"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    
    # 构建训练命令 - 恢复训练
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
        "--max-epoch", "3",  # 完成剩余的epoch
        "--max-tokens", "1800",  # 与之前一致
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "1",
        "--log-interval", "50",
        "--log-format", "simple",
        "--save-dir", SAVE_DIR,
        # 恢复训练的关键参数
        "--restore-file", "checkpoint_last.pt" if os.path.exists(temp_checkpoint) else "checkpoint_best.pt"
    ]
    
    print("\n🚀 恢复双向训练...")
    print("⏰ 预计完成时间: 10-30分钟（只需完成最后的保存和剩余步骤）")
    
    # 日志文件
    log_file = os.path.join(LOG_DIR, "resume.log")
    print(f"📝 恢复日志: {log_file}")
    
    # 写入恢复日志头
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== 恢复双向Europarl训练日志 ===\n")
        f.write("恢复时间: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("之前进度: 第1个epoch 27930/27933步 (99.9%)\n")
        f.write("验证损失: 5.447\n")
        f.write("checkpoint位置: D盘 (通过junction链接)\n")
        f.write("==================================================\n")
        f.flush()
    
    # 直接调用fairseq训练 - 恢复模式
    try:
        from fairseq_cli.train import cli_main
        print("🔧 使用直接API调用方式恢复训练...")
        
        # 设置sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ["train.py"] + cmd[3:]  # 跳过python -m fairseq_cli.train部分
        
        # 切换到fairseq目录
        original_dir = os.getcwd()
        os.chdir(FAIRSEQ_PATH)
        
        # 直接调用训练
        cli_main()
        print("✅ 双向训练恢复完成!")
        
    except Exception as e:
        print(f"❌ 恢复训练失败: {e}")
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
        print("\n🎯 现在您有了完整的双向翻译模型!")
        print("支持的翻译方向:")
        print("  • 英语 ↔ 德语")
        print("  • 英语 ↔ 西班牙语") 
        print("  • 英语 ↔ 意大利语")
        print("\n📂 模型位置: D:/machine_checkpoints/checkpoints/europarl_bidirectional/1/")
    else:
        print(f"\n❌ 恢复失败，返回码: {return_code}")
        print(f"📝 请检查日志: {log_file}")

if __name__ == "__main__":
    main() 