@echo off
setlocal enabledelayedexpansion

echo 开始在YOLOV11环境中运行Europarl训练...

set PYTHON_PATH=D:\conda_envs\YOLOV11\python.exe
set PROJECT_ROOT=C:\Users\33491\PycharmProjects\machine

cd %PROJECT_ROOT%

echo 使用Python: %PYTHON_PATH%
echo 项目根目录: %PROJECT_ROOT%

echo 检查YOLOV11环境是否可用...
%PYTHON_PATH% -c "import sys; print('Python version:', sys.version)"

echo 检查fairseq是否安装...
%PYTHON_PATH% -c "try: import fairseq; print('fairseq version:', fairseq.__version__); except ImportError as e: print('需要先安装fairseq:', e)"

echo 检查必要的Python包...
%PYTHON_PATH% -c "
try:
    import torch
    import numpy
    import pandas
    print('所有必要包已安装')
except ImportError as e:
    print('缺少包:', e)
"

echo 切换到pdec_work目录...
cd pdec_work

echo 开始运行Europarl训练脚本...
echo 注意：这个过程可能需要几个小时到几天的时间

REM 在Windows上运行bash脚本需要Git Bash或WSL
REM 如果您没有bash环境，我们需要转换为Python脚本

echo 如果您看到此消息，请确保：
echo 1. 已安装Git Bash或WSL
echo 2. fairseq已在YOLOV11环境中正确安装
echo 3. 有足够的GPU内存和磁盘空间

REM 尝试运行bash脚本
bash train/europarl.sh

if errorlevel 1 (
    echo 训练过程中出现错误，请检查日志文件
    echo 日志位置: pdec_work/logs/europarl_pdec/1.log
) else (
    echo 训练完成！
    echo 结果保存在: pdec_work/results/europarl_pdec/1/
)

echo 按任意键退出...
pause