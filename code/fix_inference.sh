#!/bin/bash

# 备份原始文件
cp pdec_work/ted_evaluation/single_inference_mini.sh pdec_work/ted_evaluation/single_inference_mini.sh.bak

# 修改文件，将DIR变量设置为空
sed -i 's/DIR="--user-dir models\/PhasedDecoder\/ "/DIR=""/' pdec_work/ted_evaluation/single_inference_mini.sh

echo "已修复single_inference_mini.sh文件，移除了--user-dir参数" 