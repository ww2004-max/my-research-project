@echo off
setlocal enabledelayedexpansion

echo 开始在YOLOV11环境中安装fairseq...

set PYTHON_PATH=D:\conda_envs\YOLOV11\python.exe
set FAIRSEQ_DIR=C:\Users\33491\PycharmProjects\machine\fairseq

cd %FAIRSEQ_DIR%

echo 使用Python: %PYTHON_PATH%
echo 安装fairseq目录: %FAIRSEQ_DIR%

echo 1. 安装依赖包...
%PYTHON_PATH% -m pip install pip==23.0
%PYTHON_PATH% -m pip install torch numpy tqdm regex sacremoses sacrebleu sentencepiece
%PYTHON_PATH% -m pip install omegaconf==2.0.0
%PYTHON_PATH% -m pip install hydra-core==1.0.7

echo 2. 以开发模式安装fairseq...
%PYTHON_PATH% -m pip install -e .

echo 3. 验证安装...
%PYTHON_PATH% -c "try: import fairseq; print('fairseq version:', fairseq.__version__); except ImportError as e: print('Import error:', e)"
%PYTHON_PATH% -c "try: import fairseq_cli; print('fairseq_cli module available'); except ImportError as e: print('Import error:', e)"

echo 安装完成!
pause