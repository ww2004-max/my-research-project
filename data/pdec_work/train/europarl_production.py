#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境Europarl训练脚本 - 完整训练配置
"""

import os
import sys

def main():
    print("开始Europarl训练流程 (生产环境)...")
    
    # 设置路径
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ, "models", "PhasedDecoder")
    
    # 添加路径到sys.path
    sys.path.insert(0, FAIRSEQ)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    print(f"添加路径: {FAIRSEQ}")
    print(f"添加路径: {PHASEDDECODER_PATH}")
    
    # 导入PhasedDecoder模块
    try:
        import models.transformer_pdec
        import criterions.label_smoothed_cross_entropy_instruction
        print("[SUCCESS] PhasedDecoder模块和criterion加载成功")
    except Exception as e:
        print(f"[ERROR] PhasedDecoder模块加载失败: {e}")
        return
    
    # 验证criterion注册
    try:
        from fairseq.criterions import CRITERION_REGISTRY
        if 'label_smoothed_cross_entropy_instruction' not in CRITERION_REGISTRY:
            print("[ERROR] criterion未正确注册")
            return
        print("[SUCCESS] criterion已正确注册")
    except Exception as e:
        print(f"[ERROR] 检查criterion注册失败: {e}")
        return
    
    # 检查CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("[ERROR] PyTorch未安装")
        return
    
    # 导入fairseq训练模块
    try:
        from fairseq_cli.train import cli_main
        print("[SUCCESS] fairseq训练模块导入成功")
    except Exception as e:
        print(f"[ERROR] fairseq训练模块导入失败: {e}")
        return
    
    # 设置训练参数
    DATA_BIN = r"C:\Users\33491\PycharmProjects\machine\fairseq\models\ZeroTrans\europarl_scripts\build_data\europarl_15-bin"
    SAVE_DIR = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_production"
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"保存目录: {SAVE_DIR}")
    
    # 生产环境训练参数 - 更大的batch size和更多epochs
    sys.argv = [
        'train.py', DATA_BIN,
        '--seed', '0',
        '--fp16',
        '--arch', 'transformer_pdec_6_e_6_d',
        '--task', 'translation_multi_simple_epoch',
        '--sampling-method', 'temperature',
        '--sampling-temperature', '5',
        '--attention-position-bias', '1',
        '--adaption-flag', 'True',
        '--adaption-inner-size', '2048',
        '--adaption-dropout', '0.1',
        '--contrastive-flag', 'True',
        '--contrastive-type', 'enc',
        '--dim', '512',
        '--mode', '1',
        '--cl-position', '6',
        '--temperature', '1.0',
        '--langs', 'en,de,es,it',
        '--lang-pairs', 'en-de,de-en,en-es,es-en,en-it,it-en',
        '--encoder-langtok', 'tgt',
        '--criterion', 'label_smoothed_cross_entropy_instruction',
        '--label-smoothing', '0.1',
        '--optimizer', 'adam',
        '--adam-betas', '(0.9,0.98)',
        '--lr', '0.0005',
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '4000',    # 标准warmup
        '--max-epoch', '30',           # 完整训练30个epoch
        '--max-tokens', '4096',        # 更大的batch size
        '--update-freq', '4',          # 梯度累积
        '--share-all-embeddings',
        '--weight-decay', '0.0001',
        '--clip-norm', '1.0',          # 梯度裁剪
        '--dropout', '0.1',
        '--attention-dropout', '0.1',
        '--save-interval', '1',        # 每个epoch保存
        '--keep-best-checkpoints', '5',
        '--patience', '10',            # 早停耐心值
        '--log-interval', '100',       # 每100步记录一次
        '--log-format', 'simple',
        '--validate-interval', '1',    # 每个epoch验证
        '--best-checkpoint-metric', 'loss',
        '--save-dir', SAVE_DIR
    ]
    
    print(f"训练命令参数: {len(sys.argv)}个")
    print("=" * 60)
    print("生产环境训练配置:")
    print(f"  - Epochs: 30")
    print(f"  - Max tokens per batch: 4096")
    print(f"  - Update frequency: 4 (梯度累积)")
    print(f"  - Warmup updates: 4000")
    print(f"  - Learning rate: 0.0005")
    print(f"  - 预计总时间: 15-25小时 (取决于GPU性能)")
    print("=" * 60)
    
    response = input("是否开始生产环境训练? (y/n): ")
    if response.lower() != 'y':
        print("训练已取消")
        return
    
    print("开始生产环境训练...")
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 切换到fairseq目录
    original_dir = os.getcwd()
    os.chdir(FAIRSEQ)
    print(f"切换到目录: {os.getcwd()}")
    
    try:
        # 直接调用训练函数
        cli_main()
        print("[SUCCESS] 训练完成!")
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_dir)
        print(f"恢复目录: {os.getcwd()}")

if __name__ == "__main__":
    main() 