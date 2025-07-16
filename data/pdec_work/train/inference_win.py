# 创建文件：pdec_work/train/inference_win.py
import os
import sys
import subprocess

# 配置
METHOD = "ted_pdec_mini"
ID = "1"
ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
SAVE_PATH = os.path.join(WORK_PATH, "results")
DATA_BIN = r"D:\europarl_15-bin"
PYTHON_PATH = r"D:\conda_envs\YOLOV11\python.exe"
BASH_PATH = r"C:\Program Files\Git\bin\bash.exe"  # Git Bash路径

# 确保结果目录存在
os.makedirs(os.path.join(SAVE_PATH, METHOD, ID), exist_ok=True)

# 语言对
languages = ["en", "de", "es", "it"]
lang_pairs = []
for src in languages:
    for tgt in languages:
        if src != tgt:
            lang_pairs.append((src, tgt))

print(f"将处理 {len(lang_pairs)} 个语言对...")

# 执行推理
for i, (src, tgt) in enumerate(lang_pairs):
    print(f"处理语言对 {i + 1}/{len(lang_pairs)}: {src}-{tgt}")
    
    # 调用单语言对推理脚本
    script_path = os.path.join(WORK_PATH, "ted_evaluation", "single_inference_mini.sh")
    
    # 设置环境变量以确保UTF-8编码
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    
    # 构建命令
    cmd = [
        BASH_PATH,
        script_path,
        METHOD,          # $1: 方法名
        ID,              # $2: ID
        src,             # $3: 源语言
        tgt,             # $4: 目标语言
        "0",             # $5: GPU ID (Windows上不使用)
        ROOT_PATH,       # $6: 根路径
        DATA_BIN         # $7: 数据路径
    ]
    
    # 执行命令
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=my_env, check=True, text=True, encoding='utf-8')
        print(f"语言对 {src}-{tgt} 处理完成，返回代码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"处理语言对 {src}-{tgt} 时出错: {e}")
        continue

print("所有语言对处理完成")