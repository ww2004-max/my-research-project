import os
import subprocess
import sys
from pathlib import Path

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
    
    # 1. 运行fairseq-generate生成翻译
    output_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.raw.txt")
    
    # 使用Python -m fairseq_cli.generate方式运行
    cmd = [
        sys.executable,
        "-m", "fairseq_cli.generate",
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
        # 提取假设(H)
        with open(output_file, "r", encoding="utf-8") as in_f, open(h_file, "w", encoding="utf-8") as out_f:
            for line in in_f:
                if line.startswith("H-"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        out_f.write(parts[2] + "\n")
        
        # 提取参考(T)
        with open(output_file, "r", encoding="utf-8") as in_f, open(r_file, "w", encoding="utf-8") as out_f:
            for line in in_f:
                if line.startswith("T-"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        out_f.write(parts[1] + "\n")
        
        # 提取源文本(S)
        with open(output_file, "r", encoding="utf-8") as in_f, open(s_file, "w", encoding="utf-8") as out_f:
            for line in in_f:
                if line.startswith("S-"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        text = parts[1]
                        # 移除语言标记
                        text = " ".join([t for t in text.split() if not t.startswith("__") or not t.endswith("__")])
                        out_f.write(text + "\n")
    except Exception as e:
        print(f"提取文本时发生错误: {e}")
        continue
    
    # 3. 进行detokenize处理
    detok_h_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.detok.h")
    detok_r_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.detok.r")
    detok_s_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.detok.s")
    
    try:
        # Detokenize假设
        subprocess.run(["perl", DETOKENIZER, "-l", tgt, "-threads", "4"], 
                      stdin=open(h_file, "r", encoding="utf-8"),
                      stdout=open(detok_h_file, "w", encoding="utf-8"))
        
        # Detokenize参考
        subprocess.run(["perl", DETOKENIZER, "-l", tgt, "-threads", "4"], 
                      stdin=open(r_file, "r", encoding="utf-8"),
                      stdout=open(detok_r_file, "w", encoding="utf-8"))
        
        # Detokenize源文本
        subprocess.run(["perl", DETOKENIZER, "-l", src, "-threads", "4"], 
                      stdin=open(s_file, "r", encoding="utf-8"),
                      stdout=open(detok_s_file, "w", encoding="utf-8"))
    except Exception as e:
        print(f"Detokenize处理时发生错误: {e}")
        continue
    
    # 4. 计算BLEU分数
    bleu_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.bleu")
    sacrebleu_file = os.path.join(SAVE_PATH, METHOD, ID, f"{ID}.sacrebleu")
    
    tok = "13a"
    if tgt == "zh":
        tok = "zh"
    elif tgt == "ja":
        tok = "ja-mecab"
    elif tgt == "ko":
        tok = "ko-mecab"
    
    try:
        # 计算BLEU
        bleu_cmd = [
            sys.executable, "-m", "sacrebleu",
            detok_h_file, "-w", "4", "-tok", tok,
        ]
        
        with open(detok_r_file, "r", encoding="utf-8") as ref_f, open(bleu_file, "w", encoding="utf-8") as out_f:
            subprocess.run(bleu_cmd, stdin=ref_f, stdout=out_f)
        
        # 添加到总结文件
        with open(sacrebleu_file, "a", encoding="utf-8") as f:
            f.write(f"{src}-{tgt}\n")
            with open(bleu_file, "r", encoding="utf-8") as bleu_f:
                f.write(bleu_f.read() + "\n")
    except Exception as e:
        print(f"计算BLEU分数时发生错误: {e}")
    
    # 5. 计算CHRF分数
    chrf_file = os.path.join(SAVE_PATH, METHOD, ID, f"{src}-{tgt}.chrf")
    chrf_summary_file = os.path.join(SAVE_PATH, METHOD, ID, f"{ID}.chrf")
    
    try:
        # 计算CHRF
        chrf_cmd = [
            sys.executable, "-m", "sacrebleu",
            detok_h_file, "-w", "4", "-m", "chrf", "--chrf-word-order", "2",
        ]
        
        with open(detok_r_file, "r", encoding="utf-8") as ref_f, open(chrf_file, "w", encoding="utf-8") as out_f:
            subprocess.run(chrf_cmd, stdin=ref_f, stdout=out_f)
        
        # 添加到总结文件
        with open(chrf_summary_file, "a", encoding="utf-8") as f:
            f.write(f"{src}-{tgt}\n")
            with open(chrf_file, "r", encoding="utf-8") as chrf_f:
                f.write(chrf_f.read() + "\n")
    except Exception as e:
        print(f"计算CHRF分数时发生错误: {e}")
    
    # 删除中间文件
    try:
        os.remove(output_file)
        os.remove(h_file)
        os.remove(r_file)
        os.remove(s_file)
    except Exception as e:
        print(f"删除中间文件时发生错误: {e}")
    
    print(f"语言对 {src}-{tgt} 处理完成")

print("所有语言对评估完成!") 