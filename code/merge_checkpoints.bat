@echo off
echo 开始合并检查点...

REM 激活YOLOV11环境
call D:\conda_envs\YOLOV11\Scripts\activate.bat

REM 执行合并检查点脚本
D:/conda_envs/YOLOV11/python.exe C:\Users\33491\PycharmProjects\machine\direct_merge.py

echo 合并检查点操作完成。
pause 