#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Europarl训练脚本 - 避免编码和子进程问题
"""

import os
import sys

def main():
    print("开始Europarl训练流程 (简化版)...")
    
    # 设置路径
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ, "models", "PhasedDecoder")
    
    # 添加路径到sys.path
    sys.path.insert(0, FAIRSEQ)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    print(f"添加路径: {FAIRSEQ}")
    print(f"添加路径: {PHASEDDECODER_PATH}")
    
    # 导入PhasedDecoder模块 - 必须在设置sys.argv之前完成
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
    
    # 检查架构注册
    try:
        from fairseq.models import ARCH_MODEL_REGISTRY
        pdec_archs = [arch for arch in ARCH_MODEL_REGISTRY.keys() if 'pdec' in arch.lower()]
        print(f"已注册的PhasedDecoder架构: {len(pdec_archs)}个")
        for arch in sorted(pdec_archs)[:5]:  # 显示前5个
            print(f"  - {arch}")
    except Exception as e:
        print(f"[ERROR] 检查架构注册失败: {e}")
        return
    
    # 检查CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
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
    SAVE_DIR = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_test\1"
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"保存目录: {SAVE_DIR}")
    
    # 设置命令行参数 - 在所有模块导入完成后设置
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
        '--warmup-updates', '500',  # 减少warmup
        '--max-epoch', '1',         # 只训练1个epoch
        '--max-tokens', '500',      # 减少tokens
        '--share-all-embeddings',
        '--weight-decay', '0.0001',
        '--no-epoch-checkpoints',
        '--no-progress-bar',
        '--keep-best-checkpoints', '1',
        '--log-interval', '10',     # 更频繁的日志
        '--log-format', 'simple',
        '--save-dir', SAVE_DIR
    ]
    
    print(f"训练命令参数: {len(sys.argv)}个")
    print("开始训练...")
    
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