#!/usr/bin/env python3
import time
import subprocess
import re
import os

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(gpu_util),
                'mem_used': int(mem_used),
                'mem_total': int(mem_total),
                'temp': int(temp)
            }
    except:
        pass
    return None

def get_training_process():
    """获取训练进程信息"""
    try:
        result = subprocess.run(['powershell', '-Command', 
                               'Get-Process python | Where-Object {$_.CPU -gt 10} | Select-Object Id, CPU, WorkingSet'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[2:]:  # 跳过标题行
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            pid = int(parts[0])
                            cpu = float(parts[1])
                            memory = int(parts[2])
                            return {'pid': pid, 'cpu': cpu, 'memory': memory}
                        except:
                            continue
    except:
        pass
    return None

def monitor_training():
    """监控训练状态"""
    print("🔍 训练性能监控器启动...")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 获取GPU信息
            gpu_info = get_gpu_info()
            
            # 获取训练进程信息
            process_info = get_training_process()
            
            # 显示信息
            print(f"\n⏰ 运行时间: {elapsed/60:.1f}分钟")
            
            if gpu_info:
                print(f"🎮 GPU: {gpu_info['gpu_util']}% | 内存: {gpu_info['mem_used']}/{gpu_info['mem_total']}MB | 温度: {gpu_info['temp']}°C")
            
            if process_info:
                print(f"💻 进程: PID {process_info['pid']} | CPU: {process_info['cpu']:.1f}s | 内存: {process_info['memory']/1024/1024:.1f}GB")
            
            # 检查是否有新的日志输出
            try:
                # 查找最新的日志文件
                log_files = []
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.endswith('.log'):
                            full_path = os.path.join(root, file)
                            mtime = os.path.getmtime(full_path)
                            if current_time - mtime < 300:  # 5分钟内修改的文件
                                log_files.append((full_path, mtime))
                
                if log_files:
                    # 找到最新的日志文件
                    latest_log = max(log_files, key=lambda x: x[1])[0]
                    print(f"📝 最新日志: {latest_log}")
                    
                    # 读取最后几行
                    try:
                        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                if 'train_inner' in last_line:
                                    print(f"📊 最新训练: {last_line[-100:]}")
                    except:
                        pass
            except:
                pass
            
            print("-" * 60)
            time.sleep(30)  # 每30秒更新一次
            
        except KeyboardInterrupt:
            print("\n👋 监控结束")
            break
        except Exception as e:
            print(f"❌ 监控出错: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training() 