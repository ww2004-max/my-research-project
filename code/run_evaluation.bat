@echo off
setlocal enabledelayedexpansion

echo 开始评估过程...

set PYTHON_PATH=D:\conda_envs\YOLOV11\python.exe
set SCRIPT_PATH=C:\Users\33491\PycharmProjects\machine\run_direct_evaluation.py

cd C:\Users\33491\PycharmProjects\machine

echo 使用Python: %PYTHON_PATH%
echo 使用脚本: %SCRIPT_PATH%

%PYTHON_PATH% %SCRIPT_PATH%

echo 评估完成!
pause 