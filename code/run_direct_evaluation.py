#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
import torch
import re

# 配置
METHOD = "ted_pdec_mini"
ID = "1"
ROOT_PATH = r"C:\Users\33491\PycharmProjects\machine"
WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
SAVE_PATH = os.path.join(WORK_PATH, "results")
DATA_BIN = os.path.join(ROOT_PATH, "fairseq", "models", "ZeroTrans", "europarl_scripts", "build_data", "europarl_15-bin")
CHECKPOINT_FILE = os.path.join(WORK_PATH, "checkpoints", METHOD, ID, "checkpoint_averaged.pt")
DETOKENIZER = os.path.join(ROOT_PATH, "moses", "scripts", "tokenizer", "detokenizer.perl")

# 检查路径是否存在
def check_path(path, description):
    if not os.path.exists(path):
        print(f"错误: {description}路径不存在: {path}")
        return False
    return True

# 检查所有必要的路径
paths_ok = True
paths_ok &= check_path(DATA_BIN, "数据二进制文件")
paths_ok &= check_path(os.path.join(WORK_PATH, "models", "PhasedDecoder"), "PhasedDecoder模型目录")
paths_ok &= check_path(CHECKPOINT_FILE, "检查点文件")
paths_ok &= check_path(DETOKENIZER, "Detokenizer脚本")

if not paths_ok:
    print("由于路径问题，无法继续执行。请修复上述路径问题。")
    sys.exit(1)

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
    
    # 1. 运行命令行工具生成翻译
    output_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.raw.txt")
    
    # 使用subprocess直接调用命令行
    cmd = [
        "D:\\conda_envs\\YOLOV11\\python.exe",
        "-m", "fairseq.scripts.generate",
        DATA_BIN,
        "--user-dir", os.path.join(WORK_PATH, "models", "PhasedDecoder"),
        "-s", src, "-t", tgt,
        "--langs", ",".join(languages),
        "--lang-pairs", ",".join([f"{s}-{t}" for s in languages for t in languages if s != t]),
        "--path", CHECKPOINT_FILE,
        "--remove-bpe", "sentencepiece",
        "--required-batch-size-multiple", "1",
        "--task", "translation_multi_simple_epoch",
        "--encoder-langtok", "tgt",
        "--beam", "4"
    ]
    
    print("运行命令:", " ".join(cmd))
    
    try:
        # 先切换到项目根目录
        os.chdir(ROOT_PATH)
        with open(output_file, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)
        print(f"命令执行成功，输出保存到 {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        print(f"错误输出: {e.stderr}")
        continue
    except Exception as e:
        print(f"发生错误: {e}")
        continue
    
    # 2. 提取假设、参考和源文本
    h_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.h")
    r_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.r")
    s_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.s")
    
    try:
        # 解析生成的输出文件
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 提取假设(H)、参考(T)和源文本(S)
        h_lines = re.findall(r'^H-\d+\t-?\d+\.\d+\t(.+)$', content, re.MULTILINE)
        t_lines = re.findall(r'^T-\d+\t(.+)$', content, re.MULTILINE)
        s_lines = re.findall(r'^S-\d+\t(.+)$', content, re.MULTILINE)
        
        # 写入提取的文本
        with open(h_file, "w", encoding="utf-8") as f:
            f.write("\n".join(h_lines))
        with open(r_file, "w", encoding="utf-8") as f:
            f.write("\n".join(t_lines))
        with open(s_file, "w", encoding="utf-8") as f:
            f.write("\n".join(s_lines))
        
        print(f"已提取并保存假设、参考和源文本")
    except Exception as e:
        print(f"处理输出文件时出错: {e}")
        continue
    
    # 3. 使用sacrebleu计算BLEU分数
    try:
        bleu_cmd = [
            "D:\\conda_envs\\YOLOV11\\python.exe",
            "-m", "sacrebleu",
            r_file,
            "-i", h_file,
            "--tokenize", "none",
            "--width", "2"
        ]
        
        print("计算BLEU分数...")
        bleu_result = subprocess.run(bleu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        bleu_score = bleu_result.stdout.strip()
        
        # 保存BLEU分数
        bleu_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.bleu")
        with open(bleu_file, "w", encoding="utf-8") as f:
            f.write(bleu_score)
        
        print(f"BLEU分数: {bleu_score}")
    except subprocess.CalledProcessError as e:
        print(f"计算BLEU分数失败: {e}")
        print(f"错误输出: {e.stderr}")
    except Exception as e:
        print(f"计算BLEU分数时出错: {e}")

print("评估完成!") 