#!/bin/bash

# 先进知识蒸馏训练脚本
# 基于您的三语言模型进行性能提升蒸馏

echo "🚀 开始先进知识蒸馏训练"
echo "教师模型: multilingual_方案1_三语言"
echo "目标: 性能提升 + 模型压缩"

# 阶段1: 多教师蒸馏
echo "📚 阶段1: 多教师知识融合..."
python fairseq_cli/train.py \
    fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin \
    --user-dir fairseq/models/PhasedDecoder \
    --task translation_multi_simple_epoch \
    --arch transformer_pdec_4_e_4_d \
    --teacher-model pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt \
    --distillation-alpha 0.7 \
    --distillation-temperature 4.0 \
    --criterion label_smoothed_cross_entropy_with_distillation \
    --optimizer adam \
    --lr 0.0003 \
    --max-epoch 5 \
    --save-dir pdec_work/checkpoints/distilled_enhanced_phase1

# 阶段2: 特征对齐
echo "🎯 阶段2: 特征对齐优化..."
python fairseq_cli/train.py \
    fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin \
    --user-dir fairseq/models/PhasedDecoder \
    --restore-file pdec_work/checkpoints/distilled_enhanced_phase1/checkpoint_best.pt \
    --feature-alignment-loss \
    --attention-alignment-weight 0.3 \
    --hidden-alignment-weight 0.2 \
    --lr 0.0001 \
    --max-epoch 3 \
    --save-dir pdec_work/checkpoints/distilled_enhanced_phase2

# 阶段3: 精细调优
echo "✨ 阶段3: 精细调优..."
python fairseq_cli/train.py \
    fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin \
    --user-dir fairseq/models/PhasedDecoder \
    --restore-file pdec_work/checkpoints/distilled_enhanced_phase2/checkpoint_best.pt \
    --lr 0.00005 \
    --max-epoch 2 \
    --save-dir pdec_work/checkpoints/distilled_enhanced_final

echo "🎉 蒸馏训练完成!"
echo "📊 开始性能对比评估..."
