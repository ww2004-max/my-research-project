#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于之前checkpoint继续训练5个epoch - GPU内存优化版
"""

import os
import sys
import gc
import torch
import threading
import time

def setup_gpu_optimization():
    """设置GPU内存优化"""
    if torch.cuda.is_available():
        # 启用内存池
        torch.cuda.empty_cache()
        
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # 启用cudnn benchmark（对固定输入大小有效）
        torch.backends.cudnn.benchmark = True
        
        print("[GPU优化] GPU内存优化设置完成")
        return True
    return False

def gpu_memory_monitor(interval=300):
    """GPU内存监控线程"""
    while True:
        try:
            if torch.cuda.is_available():
                # 获取GPU内存信息
                allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(0) / 1024**3      # GB
                
                print(f"[GPU监控] 已分配: {allocated:.2f}GB, 缓存: {cached:.2f}GB")
                
                # 如果缓存内存过多，进行清理
                if cached > allocated * 1.5:  # 缓存超过分配内存的1.5倍
                    print("[GPU清理] 检测到过多缓存，开始清理...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 清理后再次检查
                    allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                    cached_after = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"[GPU清理] 清理后 - 已分配: {allocated_after:.2f}GB, 缓存: {cached_after:.2f}GB")
            
            time.sleep(interval)
        except Exception as e:
            print(f"[GPU监控] 监控出错: {e}")
            time.sleep(interval)

def main():
    print("基于之前模型继续训练5个epoch (GPU内存优化版)...")
    
    # 设置路径
    ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
    FAIRSEQ = os.path.join(ROOT_PATH, "fairseq")
    PHASEDDECODER_PATH = os.path.join(FAIRSEQ, "models", "PhasedDecoder")
    
    # 添加路径到sys.path
    sys.path.insert(0, FAIRSEQ)
    sys.path.insert(0, PHASEDDECODER_PATH)
    
    print(f"添加路径: {FAIRSEQ}")
    print(f"添加路径: {PHASEDDECODER_PATH}")
    
    # 设置GPU优化
    gpu_available = setup_gpu_optimization()
    
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
            
            # 显示当前GPU内存使用
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"当前GPU使用: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
    except ImportError:
        print("[ERROR] PyTorch未安装")
        return
    
    # 启动GPU内存监控线程
    if gpu_available:
        monitor_thread = threading.Thread(target=gpu_memory_monitor, args=(300,), daemon=True)
        monitor_thread.start()
        print("[GPU监控] GPU内存监控线程已启动 (每5分钟检查一次)")
    
    # 导入fairseq训练模块
    try:
        from fairseq_cli.train import cli_main
        print("[SUCCESS] fairseq训练模块导入成功")
    except Exception as e:
        print(f"[ERROR] fairseq训练模块导入失败: {e}")
        return
    
    # 设置训练参数
    DATA_BIN = r"C:\Users\33491\PycharmProjects\machine\fairseq\models\ZeroTrans\europarl_scripts\build_data\europarl_15-bin"
    SAVE_DIR = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_optimized"
    RESTORE_FILE = r"C:\Users\33491\PycharmProjects\machine\pdec_work\checkpoints\europarl_continue_5epochs\checkpoint_best.pt"
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"保存目录: {SAVE_DIR}")
    print(f"恢复模型: {RESTORE_FILE}")
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(RESTORE_FILE):
        print(f"[ERROR] Checkpoint文件不存在: {RESTORE_FILE}")
        return
    
    print(f"[SUCCESS] 找到checkpoint文件: {os.path.getsize(RESTORE_FILE) / 1024**2:.1f} MB")
    
    # 基于之前模型继续训练5个epoch - GPU内存优化版
    sys.argv = [
        'train.py', DATA_BIN,
        '--seed', '0',
        '--fp16',                       # 使用混合精度训练节省内存
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
        '--lr', '0.0005',               # 使用与原训练相同的学习率
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '2000',     # 使用与原训练相同的warmup
        '--max-epoch', '5',             # 继续训练5个epoch
        '--max-tokens', '2048',         # 使用适中的batch size平衡速度和内存
        '--update-freq', '2',           # 梯度累积
        '--share-all-embeddings',
        '--weight-decay', '0.0001',
        '--clip-norm', '1.0',
        '--dropout', '0.1',
        '--attention-dropout', '0.1',
        '--save-interval', '1',         # 每个epoch保存
        '--keep-best-checkpoints', '3', # 减少保存的checkpoint数量以节省空间
        '--patience', '10',             # 早停
        '--log-interval', '50',         # 每50步记录
        '--log-format', 'simple',
        '--validate-interval', '1',     # 每个epoch验证
        '--best-checkpoint-metric', 'loss',
        '--save-dir', SAVE_DIR,
        '--restore-file', RESTORE_FILE, # 从之前的checkpoint恢复
        '--empty-cache-freq', '100',    # 每100步清理一次GPU缓存
        '--all-gather-list-size', '16384',  # 优化分布式训练内存使用
        '--checkpoint-activations',     # 用计算换内存，减少激活内存占用
        '--ddp-backend', 'no_c10d',     # 优化内存使用
        # 注意：保持优化器状态，不使用reset参数
    ]
    
    print(f"训练命令参数: {len(sys.argv)}个")
    print("=" * 60)
    print("基于之前模型继续训练配置 (GPU内存优化版):")
    print(f"  - 基础模型: checkpoint_best.pt (损失: 5.9260)")
    print(f"  - 继续训练: 5个epoch")
    print(f"  - Max tokens per batch: 2048 (与原训练一致)")
    print(f"  - Update frequency: 2")
    print(f"  - Warmup updates: 2000 (与原训练一致)")
    print(f"  - Learning rate: 0.0005 (与原训练一致)")
    print(f"  - 保持优化器状态: 是")
    print(f"  - GPU内存优化: 是")
    print(f"  - 自动内存清理: 每100步")
    print(f"  - 内存监控: 每5分钟")
    print(f"  - 预计训练时间: 4-6小时")
    print("=" * 60)
    
    # 训练前清理一次GPU内存
    if gpu_available:
        print("[GPU清理] 训练前清理GPU内存...")
        torch.cuda.empty_cache()
        gc.collect()
    
    print("开始基于之前模型的继续训练 (GPU内存优化版)...")
    
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
        # 训练结束后清理GPU内存
        if gpu_available:
            print("[GPU清理] 训练结束，清理GPU内存...")
            torch.cuda.empty_cache()
            gc.collect()
        
        os.chdir(original_dir)
        print(f"恢复目录: {os.getcwd()}")

if __name__ == "__main__":
    main() 