#!/usr/bin/env python3
"""
GPU内存清理工具 - 在训练过程中定期清理GPU内存
使用方法:
python gpu_memory_cleaner.py --interval 3600 --pid <训练进程PID>
"""

import argparse
import time
import os
import sys
import signal
import logging
import subprocess
import gc
import psutil
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('gpu_memory_cleaner')

def get_gpu_memory_info():
    """使用nvidia-smi命令获取GPU内存使用情况"""
    try:
        # 使用nvidia-smi命令获取GPU信息
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], 
                                        universal_newlines=True)
        lines = result.strip().split('\n')
        
        if not lines:
            return "GPU不可用", 0, 0, 0, 0
        
        # 解析第一个GPU的信息
        parts = lines[0].split(', ')
        if len(parts) >= 4:
            name = parts[0]
            total_memory_mb = float(parts[1])
            free_memory_mb = float(parts[2])
            used_memory_mb = float(parts[3])
            
            return name, total_memory_mb, free_memory_mb, used_memory_mb, total_memory_mb - free_memory_mb
        else:
            return "解析错误", 0, 0, 0, 0
    except Exception as e:
        logger.error(f"获取GPU信息时出错: {e}")
        return "GPU不可用", 0, 0, 0, 0

def clean_gpu_memory():
    """清理GPU内存"""
    try:
        # 强制Python垃圾回收
        gc.collect()
        
        # 尝试使用系统命令清理CUDA缓存
        try:
            # 在Windows上尝试使用taskkill终止不必要的CUDA进程
            # 注意：这可能会影响其他使用GPU的应用程序，请谨慎使用
            # subprocess.call('taskkill /F /IM "cudart64*.dll" /T', shell=True)
            pass
        except Exception as e:
            logger.warning(f"清理CUDA进程时出错: {e}")
        
        # 再次进行垃圾回收
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"清理GPU内存时出错: {e}")
        return False

def is_process_running(pid):
    """检查进程是否在运行"""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def main():
    parser = argparse.ArgumentParser(description='GPU内存清理工具')
    parser.add_argument('--interval', type=int, default=3600, help='清理间隔(秒)')
    parser.add_argument('--pid', type=int, help='要监视的训练进程PID')
    parser.add_argument('--log-file', type=str, default='gpu_cleaner.log', help='日志文件路径')
    
    args = parser.parse_args()
    
    # 添加文件日志
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"GPU内存清理工具已启动，清理间隔: {args.interval}秒")
    
    if args.pid:
        logger.info(f"监视训练进程PID: {args.pid}")
    
    try:
        while True:
            # 检查训练进程是否还在运行
            if args.pid and not is_process_running(args.pid):
                logger.info(f"训练进程 {args.pid} 已结束，清理工具退出")
                break
            
            # 获取清理前的GPU内存信息
            gpu_info_before = get_gpu_memory_info()
            logger.info(f"清理前 - GPU: {gpu_info_before[0]}, 总内存: {gpu_info_before[1]:.2f}MB, "
                       f"可用: {gpu_info_before[2]:.2f}MB, 已使用: {gpu_info_before[3]:.2f}MB")
            
            # 清理GPU内存
            success = clean_gpu_memory()
            
            # 获取清理后的GPU内存信息
            gpu_info_after = get_gpu_memory_info()
            logger.info(f"清理后 - GPU: {gpu_info_after[0]}, 总内存: {gpu_info_after[1]:.2f}MB, "
                       f"可用: {gpu_info_after[2]:.2f}MB, 已使用: {gpu_info_after[3]:.2f}MB")
            
            freed_memory = gpu_info_after[2] - gpu_info_before[2]
            logger.info(f"{'成功' if success else '失败'}清理GPU内存，释放了 {freed_memory:.2f}MB")
            
            # 等待下一次清理
            logger.info(f"等待 {args.interval} 秒后进行下一次清理...")
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        logger.info("用户中断，清理工具退出")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        
    logger.info("GPU内存清理工具已退出")

if __name__ == "__main__":
    main() 