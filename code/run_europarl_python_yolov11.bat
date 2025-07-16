 @echo off
setlocal enabledelayedexpansion

echo 使用YOLOV11环境运行Europarl Python训练脚本...

set PYTHON_PATH=D:\conda_envs\YOLOV11\python.exe
set PROJECT_ROOT=C:\Users\33491\PycharmProjects\machine

cd %PROJECT_ROOT%

echo 使用Python: %PYTHON_PATH%
echo 项目根目录: %PROJECT_ROOT%

echo 检查环境和依赖...
%PYTHON_PATH% -c "
import sys
print('Python version:', sys.version)
try:
    import fairseq
    print('fairseq version:', fairseq.__version__)
except ImportError as e:
    print('fairseq未安装:', e)
    sys.exit(1)

try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count())
except ImportError as e:
    print('PyTorch未安装:', e)
    sys.exit(1)

try:
    import pandas
    print('pandas version:', pandas.__version__)
except ImportError:
    print('pandas未安装，正在安装...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas', 'openpyxl'])
"

if errorlevel 1 (
    echo 环境检查失败，请先安装必要的依赖
    pause
    exit /b 1
)

echo 切换到pdec_work目录...
cd pdec_work

echo 开始运行Python训练脚本...
echo 注意：这个过程可能需要几个小时到几天的时间

%PYTHON_PATH% train/europarl_python.py

if errorlevel 1 (
    echo 训练过程中出现错误
    echo 请检查日志文件: pdec_work/logs/europarl_pdec/1.log
) else (
    echo 训练和评估完成！
    echo 检查点: pdec_work/checkpoints/europarl_pdec/1/
    echo 结果: pdec_work/results/europarl_pdec/1/
    echo Excel表格: pdec_work/excel/
)

echo 按任意键退出...
pause