 # 在YOLOV11环境中运行Europarl训练

## 环境要求

1. **YOLOV11环境**：`D:\conda_envs\YOLOV11\`
2. **必要的Python包**：
   - fairseq
   - torch
   - pandas
   - numpy
   - sacrebleu

## 使用方法

### 方法一：使用Python脚本（推荐）

```bash
# 双击运行
run_europarl_python_yolov11.bat
```

或者手动运行：
```bash
D:\conda_envs\YOLOV11\python.exe pdec_work/train/europarl_python.py
```

### 方法二：使用Bash脚本（需要Git Bash或WSL）

```bash
# 双击运行
run_europarl_with_yolov11.bat
```

## 训练流程

1. **环境检查**：验证Python、fairseq、PyTorch等是否安装
2. **模型训练**：30个epoch的多语言翻译模型训练
3. **检查点平均**：提高模型性能
4. **推理评估**：在测试集上进行翻译
5. **结果整理**：生成BLEU分数表格

## 输出文件

- **检查点**：`pdec_work/checkpoints/europarl_pdec/1/`
- **训练日志**：`pdec_work/logs/europarl_pdec/1.log`
- **推理结果**：`pdec_work/results/europarl_pdec/1/`
- **Excel表格**：`pdec_work/excel/europarl_europarl_pdec_1_results.xlsx`

## 语言对

支持以下6个语言对：
- en-de (英语-德语)
- de-en (德语-英语)
- en-es (英语-西班牙语)
- es-en (西班牙语-英语)
- en-it (英语-意大利语)
- it-en (意大利语-英语)

## 注意事项

1. **GPU要求**：默认使用8个GPU，如果GPU不足请修改脚本
2. **时间要求**：完整训练可能需要几小时到几天
3. **存储空间**：确保有足够空间存储检查点（可能需要几GB）
4. **内存要求**：建议至少16GB RAM

## 故障排除

### 如果fairseq未安装：
```bash
install_fairseq_yolov11.bat
```

### 如果缺少pandas：
```bash
D:\conda_envs\YOLOV11\python.exe -m pip install pandas openpyxl
```

### 如果训练失败：
1. 检查日志文件：`pdec_work/logs/europarl_pdec/1.log`
2. 确认数据集路径正确
3. 检查GPU内存是否足够

## 监控训练进度

可以通过以下方式监控：
1. 查看日志文件实时更新
2. 检查检查点目录中的文件
3. 观察GPU使用情况

训练完成后，结果会自动保存并生成评估报告。