#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于之前checkpoint继续训练5个epoch
"""

import os
import sys

def main():
    print("基于之前模型继续训练5个epoch...")
    
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
    SAVE_DIR = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_continue_5epochs"
    RESTORE_FILE = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_test\1\checkpoint_best.pt"
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"保存目录: {SAVE_DIR}")
    print(f"恢复模型: {RESTORE_FILE}")
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(RESTORE_FILE):
        print(f"[ERROR] Checkpoint文件不存在: {RESTORE_FILE}")
        return
    
    print(f"[SUCCESS] 找到checkpoint文件: {os.path.getsize(RESTORE_FILE) / 1024**2:.1f} MB")
    
    # 基于之前模型继续训练5个epoch
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
        '--lr', '0.0003',               # 稍微降低学习率，因为是继续训练
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '1000',     # 减少warmup，因为已经预训练过
        '--max-epoch', '5',             # 继续训练5个epoch
        '--max-tokens', '3072',         # 稍微增加batch size
        '--update-freq', '2',           # 梯度累积
        '--share-all-embeddings',
        '--weight-decay', '0.0001',
        '--clip-norm', '1.0',
        '--dropout', '0.1',
        '--attention-dropout', '0.1',
        '--save-interval', '1',         # 每个epoch保存
        '--keep-best-checkpoints', '5', # 保留更多最佳checkpoint
        '--patience', '10',             # 早停
        '--log-interval', '50',         # 每50步记录
        '--log-format', 'simple',
        '--validate-interval', '1',     # 每个epoch验证
        '--best-checkpoint-metric', 'loss',
        '--save-dir', SAVE_DIR,
        '--restore-file', RESTORE_FILE, # 关键：从之前的checkpoint恢复
        '--reset-optimizer',            # 重置优化器状态
        '--reset-lr-scheduler',         # 重置学习率调度器
        '--reset-meters'                # 重置统计指标
    ]
    
    print(f"训练命令参数: {len(sys.argv)}个")
    print("=" * 60)
    print("基于之前模型继续训练配置:")
    print(f"  - 基础模型: checkpoint_best.pt (损失: 5.9260)")
    print(f"  - 继续训练: 5个epoch")
    print(f"  - Max tokens per batch: 3072")
    print(f"  - Update frequency: 2")
    print(f"  - Warmup updates: 1000 (减少)")
    print(f"  - Learning rate: 0.0003 (降低)")
    print(f"  - 预计训练时间: 4-6小时")
    print(f"  - 每个epoch约1小时")
    print("=" * 60)
    
    print("开始基于之前模型的继续训练...")
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 切换到fairseq目录
    original_dir = os.getcwd()
    os.chdir(FAIRSEQ)
    print(f"切换到目录: {os.getcwd()}")
    
    try:
        # 直接调用训练函数
        cli_main()
        print("[SUCCESS] 基于之前模型的5个epoch训练完成!")
    except Exception as e:
        print(f"[ERROR] 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_dir)
        print(f"恢复目录: {os.getcwd()}")

if __name__ == "__main__":
    main() 