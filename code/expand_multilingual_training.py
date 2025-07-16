#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展多语言训练脚本 - 基于成功的双向训练模式
支持更多语言对的多语言翻译
"""

import os
import sys
from datetime import datetime

def main():
    print("🌍 扩展多语言训练（基于成功的双向模式）")
    print("============================================================")
    
    # 可选的语言配置方案
    language_configs = {
        "方案1_三语言": {
            "langs": "en,de,es",
            "lang_pairs": "en-de,de-en,en-es,es-en",
            "description": "英语↔德语, 英语↔西班牙语 (4个方向)"
        },
        "方案2_四语言": {
            "langs": "en,de,es,it", 
            "lang_pairs": "en-de,de-en,en-es,es-en,en-it,it-en",
            "description": "英语↔德语, 英语↔西班牙语, 英语↔意大利语 (6个方向)"
        },
        "方案3_五语言": {
            "langs": "en,de,es,it,fr",
            "lang_pairs": "en-de,de-en,en-es,es-en,en-it,it-en,en-fr,fr-en",
            "description": "英德西意法，以英语为中心 (8个方向)"
        },
        "方案4_欧洲主要语言": {
            "langs": "en,de,es,it,fr,pt,nl",
            "lang_pairs": "en-de,de-en,en-es,es-en,en-it,it-en,en-fr,fr-en,en-pt,pt-en,en-nl,nl-en",
            "description": "欧洲主要语言，以英语为中心 (12个方向)"
        }
    }
    
    print("📋 可选语言配置方案:")
    for key, config in language_configs.items():
        lang_count = len(config["langs"].split(","))
        pair_count = len(config["lang_pairs"].split(","))
        print(f"  {key}: {config['description']}")
        print(f"    语言数: {lang_count}, 翻译方向: {pair_count}")
    
    # 选择方案 (可以修改这里选择不同方案)
    selected_config = "方案1_三语言"  # 选择英德西三语言方案
    config = language_configs[selected_config]
    
    print(f"\n🎯 选择方案: {selected_config}")
    print(f"📝 描述: {config['description']}")
    print(f"🌍 语言: {config['langs']}")
    print(f"🔄 翻译对: {config['lang_pairs']}")
    
    # 路径配置 - 使用你现有的成功路径
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ_PATH = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ_PATH, "models", "PhasedDecoder")
    
    # 添加路径到 Python 路径
    print(f"\n🔧 添加到Python路径: {FAIRSEQ_PATH}")
    print(f"🔧 添加到Python路径: {PHASEDDECODER_PATH}")
    sys.path.insert(0, FAIRSEQ_PATH)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    # 导入PhasedDecoder模块
    try:
        print("🔧 导入PhasedDecoder模块...")
        import models.transformer_pdec
        import criterions.label_smoothed_cross_entropy_instruction
        print("✅ PhasedDecoder模块和criterion加载成功")
    except Exception as e:
        print(f"❌ PhasedDecoder模块加载失败: {e}")
        return
    
    # 验证架构和criterion注册
    try:
        from fairseq.models import ARCH_MODEL_REGISTRY
        from fairseq.criterions import CRITERION_REGISTRY
        
        if 'transformer_pdec_6_e_6_d' not in ARCH_MODEL_REGISTRY:
            print("❌ transformer_pdec_6_e_6_d 架构未注册")
            return
        if 'label_smoothed_cross_entropy_instruction' not in CRITERION_REGISTRY:
            print("❌ criterion未正确注册")
            return
            
        print("✅ 架构和criterion已正确注册")
    except Exception as e:
        print(f"❌ 检查注册失败: {e}")
        return
    
    # 路径配置 - 使用现有的europarl数据
    DATA_BIN = os.path.join(ROOT_PATH, "fairseq", "models", "ZeroTrans", "europarl_scripts", "build_data", "europarl_15-bin")
    SAVE_DIR = os.path.join(ROOT_PATH, "pdec_work", "checkpoints", f"multilingual_{selected_config}", "1")
    LOG_DIR = os.path.join(ROOT_PATH, "pdec_work", "logs", f"multilingual_{selected_config}")
    
    # 创建目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"\n📂 检查点目录: {SAVE_DIR}")
    print(f"📝 日志目录: {LOG_DIR}")
    print(f"📁 数据目录: {DATA_BIN}")
    
    # 检查数据是否存在
    if not os.path.exists(DATA_BIN):
        print(f"❌ 数据目录不存在: {DATA_BIN}")
        return
    
    # 构建训练命令 - 基于你成功的配置
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
        # 多语言配置 - 使用选择的方案
        "--langs", config["langs"],
        "--lang-pairs", config["lang_pairs"],
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy_instruction",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0005",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "500",
        "--max-epoch", "3",
        "--max-tokens", "1600",  # 根据语言对数量调整
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "1",
        "--log-interval", "50",
        "--log-format", "simple",
        "--save-dir", SAVE_DIR
    ]
    
    # 估算训练规模
    lang_count = len(config["langs"].split(","))
    pair_count = len(config["lang_pairs"].split(","))
    estimated_samples = pair_count * 185000  # 每个语言对约18.5万样本
    estimated_hours = pair_count * 0.8  # 每个语言对约0.8小时
    
    print(f"\n🚀 开始多语言训练...")
    print(f"📊 语言数量: {lang_count}")
    print(f"📊 翻译方向: {pair_count}")
    print(f"📊 预估样本: {estimated_samples:,}")
    print(f"⏰ 预估时间: {estimated_hours:.1f}小时")
    
    # 日志文件
    log_file = os.path.join(LOG_DIR, "1.log")
    print(f"📝 日志文件: {log_file}")
    
    # 写入日志头
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== 多语言训练日志（{selected_config}） ===\n")
        f.write(f"语言: {config['langs']}\n")
        f.write(f"语言对: {config['lang_pairs']}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"预估样本: {estimated_samples:,}\n")
        f.write(f"预估时间: {estimated_hours:.1f}小时\n")
        f.write("==================================================\n")
        f.flush()
    
    # 询问用户是否继续
    print(f"\n❓ 是否开始训练？这将需要约 {estimated_hours:.1f} 小时")
    print("💡 提示: 你可以修改脚本中的 selected_config 来选择不同方案")
    
    # 直接调用fairseq训练 - 使用你成功的方式
    try:
        from fairseq_cli.train import cli_main
        print("🔧 使用直接API调用方式...")
        
        # 设置sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ["train.py"] + cmd[3:]
        
        # 切换到fairseq目录
        original_dir = os.getcwd()
        os.chdir(FAIRSEQ_PATH)
        
        # 直接调用训练
        cli_main()
        print("✅ 多语言训练完成!")
        
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
        print(f"\n🎉 多语言训练成功完成!")
        print(f"\n🎯 现在模型支持 {lang_count} 种语言，{pair_count} 个翻译方向")
        print(f"📁 模型位置: {SAVE_DIR}")
        print("\n🎯 测试模型:")
        print("python test_multilingual_model.py")
    else:
        print(f"\n❌ 训练失败，返回码: {return_code}")
        print(f"📝 请检查日志: {log_file}")

if __name__ == "__main__":
    main() 