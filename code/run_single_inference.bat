@echo off
setlocal enabledelayedexpansion

echo 开始单语言对评估...

set METHOD=ted_pdec_mini
set ID=1
set SRC=en
set TGT=de
set ROOT_PATH=C:\Users\33491\PycharmProjects\machine
set DATA_BIN=%ROOT_PATH%\fairseq\models\ZeroTrans\europarl_scripts\build_data\europarl_15-bin
set WORK_PATH=%ROOT_PATH%\pdec_work
set SAVE_PATH=%WORK_PATH%\results
set DETOKENIZER=%ROOT_PATH%\moses\scripts\tokenizer\detokenizer.perl
set PYTHON_CMD=D:\conda_envs\YOLOV11\python.exe

echo 方法: %METHOD%
echo ID: %ID%
echo 源语言: %SRC%
echo 目标语言: %TGT%
echo 根目录: %ROOT_PATH%
echo 数据目录: %DATA_BIN%
echo Python: %PYTHON_CMD%

REM 确保结果目录存在
if not exist %SAVE_PATH%\%METHOD%\%ID% (
    mkdir %SAVE_PATH%\%METHOD%\%ID%
)

cd %ROOT_PATH%\fairseq

REM 设置环境变量
set PYTHONIOENCODING=utf-8

set DIR=--user-dir %ROOT_PATH%\pdec_work\models\PhasedDecoder
set TGT_FILE=%SRC%-%TGT%.raw.txt

echo 正在处理语言对 %SRC%-%TGT%...

REM 检查检查点文件是否存在
set CHECKPOINT_FILE=%WORK_PATH%\checkpoints\%METHOD%\%ID%\checkpoint_averaged.pt
if not exist %CHECKPOINT_FILE% (
    echo 错误: 检查点文件不存在: %CHECKPOINT_FILE%
    exit /b 1
)

REM 执行推理命令
echo 执行推理命令...
%PYTHON_CMD% -m fairseq_cli.generate %DATA_BIN% ^
%DIR% ^
-s %SRC% -t %TGT% ^
--langs "en,de,es,it" ^
--lang-pairs "en-de,de-en,en-es,es-en,en-it,it-en" ^
--path %CHECKPOINT_FILE% ^
--remove-bpe sentencepiece ^
--required-batch-size-multiple 1 ^
--task translation_multi_simple_epoch ^
--encoder-langtok tgt ^
--beam 4 > %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE%

REM 检查推理结果是否生成
if not exist %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE% (
    echo 错误: 推理结果文件未生成
    exit /b 1
)

echo 提取假设、参考和源文本...
REM 在Windows下使用PowerShell处理文本
powershell -Command "Get-Content %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE% | Select-String -Pattern '^H-' | ForEach-Object { $_.Line.Split('\t')[2] } | Out-File -Encoding utf8 %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.h"
powershell -Command "Get-Content %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE% | Select-String -Pattern '^T-' | ForEach-Object { $_.Line.Split('\t')[1] } | Out-File -Encoding utf8 %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.r"
powershell -Command "Get-Content %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE% | Select-String -Pattern '^S-' | ForEach-Object { $_.Line.Split('\t')[1].Replace('__tgt_en__ ', '').Replace('__tgt_de__ ', '').Replace('__tgt_es__ ', '').Replace('__tgt_it__ ', '') } | Out-File -Encoding utf8 %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.s"

REM 删除原始文件
del %SAVE_PATH%\%METHOD%\%ID%\%TGT_FILE%

echo 进行detokenize处理...
perl %DETOKENIZER% -l %TGT% < %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.h > %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.h
perl %DETOKENIZER% -l %TGT% < %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.r > %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.r
perl %DETOKENIZER% -l %SRC% < %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.s > %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.s

REM 删除中间文件
del %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.h
del %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.r
del %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.s

set TOK=13a
if "%TGT%"=="zh" set TOK=zh
if "%TGT%"=="ja" set TOK=ja-mecab
if "%TGT%"=="ko" set TOK=ko-mecab

echo 计算评估指标...
%PYTHON_CMD% -m sacrebleu %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.h -w 4 -tok %TOK% < %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.r > %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.bleu
echo %SRC%-%TGT% > %SAVE_PATH%\%METHOD%\%ID%\%ID%.sacrebleu
type %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.bleu >> %SAVE_PATH%\%METHOD%\%ID%\%ID%.sacrebleu

%PYTHON_CMD% -m sacrebleu %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.h -w 4 -m chrf --chrf-word-order 2 < %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.detok.r > %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.chrf
echo %SRC%-%TGT% > %SAVE_PATH%\%METHOD%\%ID%\%ID%.chrf
type %SAVE_PATH%\%METHOD%\%ID%\%SRC%-%TGT%.chrf >> %SAVE_PATH%\%METHOD%\%ID%\%ID%.chrf

echo 语言对 %SRC%-%TGT% 处理完成
pause 