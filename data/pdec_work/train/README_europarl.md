# Europarl训练脚本使用说明

这个脚本用于训练Europarl数据集上的PhasedDecoder模型。

## 支持的语言对
- en-de (英语-德语)
- de-en (德语-英语)  
- en-es (英语-西班牙语)
- es-en (西班牙语-英语)
- en-it (英语-意大利语)
- it-en (意大利语-英语)

## 使用方法

1. 确保数据集已准备好：
   ```
   C:\Users\33491\PycharmProjects\machine\fairseq\models\ZeroTrans\europarl_scripts\build_data\europarl_15-bin
   ```

2. 运行训练脚本：
   ```bash
   cd pdec_work
   bash train/europarl.sh
   ```

## 脚本功能

1. **模型训练**：使用PhasedDecoder架构训练多语言翻译模型
2. **检查点平均**：对最佳检查点进行平均以提高性能
3. **推理**：在测试集上进行翻译
4. **结果评估**：计算BLEU分数并生成结果表格

## 输出文件

- **检查点**：`pdec_work/checkpoints/europarl_pdec/1/`
- **训练日志**：`pdec_work/logs/europarl_pdec/1.log`
- **推理结果**：`pdec_work/results/europarl_pdec/1/`
- **评估表格**：`pdec_work/excel/europarl_europarl_pdec_1_results.xlsx`

## 注意事项

- 脚本使用8个GPU进行训练（CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7）
- 如果GPU数量不足，请修改脚本中的`num_gpus`参数
- 训练大约需要30个epoch完成
- 确保有足够的磁盘空间存储检查点和结果文件 