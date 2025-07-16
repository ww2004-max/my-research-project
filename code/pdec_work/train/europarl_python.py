 #!/usr/bin/env python3
"""
Europarl training script (Python version)
等价于 europarl.sh 的Python实现
"""

import os
import sys
import subprocess
import time

def main():
    # 设置路径和参数
    DATA_BIN = "C:/Users/33491/PycharmProjects/machine/fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin"
    ROOT_PATH = "C:/Users/33491/PycharmProjects/machine"
    num_gpus = 8
    FAIR_PATH = os.path.join(ROOT_PATH, "fairseq")
    WORK_PATH = os.path.join(ROOT_PATH, "pdec_work")
    
    METHOD = "europarl_pdec"
    ID = "1"
    
    # 模型参数
    ENC = 6
    DEC = 6
    BIAS = 1
    ADAPTION = 'True'
    DROP = 0.1
    INNER = 2048
    CONTRASTIVE = 'True'
    POSITION = 6
    TYPE = 'enc'
    T = 1.0
    DIM = 512
    MODE = 1
    SEED = 0
    
    print("开始Europarl训练流程...")
    
    # 创建必要目录
    dirs_to_create = [
        os.path.join(WORK_PATH, "checkpoints"),
        os.path.join(WORK_PATH, "checkpoints", METHOD),
        os.path.join(WORK_PATH, "checkpoints", METHOD, ID),
        os.path.join(WORK_PATH, "logs"),
        os.path.join(WORK_PATH, "logs", METHOD),
        os.path.join(WORK_PATH, "results"),
        os.path.join(WORK_PATH, "results", METHOD),
        os.path.join(WORK_PATH, "results", METHOD, ID),
        os.path.join(WORK_PATH, "europarl_evaluation"),
        os.path.join(WORK_PATH, "excel")
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 切换到fairseq目录
    original_dir = os.getcwd()
    os.chdir(FAIR_PATH)
    print(f"切换到目录: {FAIR_PATH}")
    
    # 构建训练命令
    train_cmd = [
        "fairseq-train", DATA_BIN,
        "--user-dir", "models/PhasedDecoder-main/PhasedDecoder/",
        "--seed", str(SEED),
        "--fp16",
        "--ddp-backend=no_c10d",
        "--arch", f"transformer_pdec_{ENC}_e_{DEC}_d",
        "--task", "translation_multi_simple_epoch",
        "--sampling-method", "temperature",
        "--sampling-temperature", "5",
        "--attention-position-bias", str(BIAS),
        "--adaption-flag", ADAPTION,
        "--adaption-inner-size", str(INNER),
        "--adaption-dropout", str(DROP),
        "--contrastive-flag", CONTRASTIVE,
        "--contrastive-type", TYPE,
        "--dim", str(DIM),
        "--mode", str(MODE),
        "--cl-position", str(POSITION),
        "--temperature", str(T),
        "--langs", "en,de,es,it",
        "--lang-pairs", "en-de,de-en,en-es,es-en,en-it,it-en",
        "--encoder-langtok", "tgt",
        "--criterion", "label_smoothed_cross_entropy_instruction",
        "--label-smoothing", "0.1",
        "--optimizer", "adam",
        "--adam-betas", "(0.9,0.98)",
        "--lr", "0.0005",
        "--lr-scheduler", "inverse_sqrt",
        "--warmup-updates", "4000",
        "--max-epoch", "30",
        "--max-tokens", "4000",
        "--share-all-embeddings",
        "--weight-decay", "0.0001",
        "--no-epoch-checkpoints",
        "--no-progress-bar",
        "--keep-best-checkpoints", "5",
        "--log-interval", "1000",
        "--log-format", "simple",
        "--save-dir", os.path.join(WORK_PATH, "checkpoints", METHOD, ID)
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # 开始训练
    print("开始模型训练...")
    log_file = os.path.join(WORK_PATH, "logs", METHOD, f"{ID}.log")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.run(train_cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        
        if process.returncode == 0:
            print("训练完成!")
        else:
            print(f"训练失败，返回码: {process.returncode}")
            print(f"请检查日志文件: {log_file}")
            return
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return
    
    # 平均检查点
    print("开始平均检查点...")
    checkpoint_dir = os.path.join(WORK_PATH, "checkpoints", METHOD, ID)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint.best_loss_")]
    
    if checkpoint_files:
        avg_cmd = [
            "python3", "scripts/average_checkpoints.py",
            "--inputs"] + [os.path.join(checkpoint_dir, f) for f in checkpoint_files] + [
            "--output", os.path.join(checkpoint_dir, "checkpoint_averaged.pt")
        ]
        
        try:
            subprocess.run(avg_cmd, check=True)
            print("检查点平均完成!")
        except subprocess.CalledProcessError as e:
            print(f"检查点平均失败: {e}")
    
    # 切换回工作目录
    os.chdir(os.path.join(original_dir, WORK_PATH))
    
    # 运行推理
    print("开始推理...")
    inference_cmd = [
        "bash", "europarl_evaluation/batch_inference.sh",
        METHOD, ID, ROOT_PATH, str(num_gpus), DATA_BIN,
        "en", "de", "es", "it"
    ]
    
    try:
        subprocess.run(inference_cmd, check=True)
        print("推理完成!")
    except subprocess.CalledProcessError as e:
        print(f"推理失败: {e}")
    
    # 生成结果表格
    print("生成结果表格...")
    table_cmd = [
        "python", "europarl_evaluation/make_table.py",
        METHOD, ID, ROOT_PATH, "en", "de", "es", "it"
    ]
    
    try:
        subprocess.run(table_cmd, check=True)
        print("结果表格生成完成!")
    except subprocess.CalledProcessError as e:
        print(f"结果表格生成失败: {e}")
    
    print("\n训练和评估完成!")
    print(f"结果保存在: {os.path.join(WORK_PATH, 'checkpoints', METHOD, ID)}")
    print(f"日志保存在: {log_file}")
    print(f"推理结果在: {os.path.join(WORK_PATH, 'results', METHOD, ID)}")

if __name__ == "__main__":
    main()