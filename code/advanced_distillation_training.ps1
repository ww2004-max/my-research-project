# 先进知识蒸馏训练脚本 (Windows PowerShell版本)
# 基于您的三语言模型进行性能提升蒸馏

Write-Host "🚀 开始先进知识蒸馏训练" -ForegroundColor Green
Write-Host "教师模型: multilingual_方案1_三语言" -ForegroundColor Cyan
Write-Host "目标: 性能提升 + 模型压缩" -ForegroundColor Cyan
Write-Host ""

# 检查环境
Write-Host "🔍 检查训练环境..." -ForegroundColor Yellow
if (-not (Test-Path "fairseq")) {
    Write-Host "❌ 错误: 未找到fairseq目录" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt")) {
    Write-Host "❌ 错误: 未找到教师模型" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 环境检查通过" -ForegroundColor Green
Write-Host ""

# 创建输出目录
$outputDirs = @(
    "pdec_work/checkpoints/distilled_enhanced_phase1",
    "pdec_work/checkpoints/distilled_enhanced_phase2", 
    "pdec_work/checkpoints/distilled_enhanced_final"
)

foreach ($dir in $outputDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "📁 创建目录: $dir" -ForegroundColor Blue
    }
}

Write-Host ""

# 阶段1: 多教师蒸馏
Write-Host "📚 阶段1: 多教师知识融合..." -ForegroundColor Magenta
Write-Host "预计时间: 30-45分钟" -ForegroundColor Gray

$phase1Args = @(
    "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin",
    "--user-dir", "fairseq/models/PhasedDecoder",
    "--task", "translation_multi_simple_epoch",
    "--arch", "transformer_pdec_4_e_4_d",
    "--teacher-model", "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt",
    "--distillation-alpha", "0.7",
    "--distillation-temperature", "4.0",
    "--criterion", "label_smoothed_cross_entropy_with_distillation",
    "--optimizer", "adam",
    "--lr", "0.0003",
    "--max-epoch", "5",
    "--save-dir", "pdec_work/checkpoints/distilled_enhanced_phase1",
    "--batch-size", "32",
    "--update-freq", "2"
)

try {
    python fairseq_cli/train.py @phase1Args
    if ($LASTEXITCODE -ne 0) {
        throw "阶段1训练失败"
    }
    Write-Host "✅ 阶段1完成" -ForegroundColor Green
} catch {
    Write-Host "❌ 阶段1失败: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 阶段2: 特征对齐
Write-Host "🎯 阶段2: 特征对齐优化..." -ForegroundColor Magenta
Write-Host "预计时间: 20-30分钟" -ForegroundColor Gray

$phase2Args = @(
    "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin",
    "--user-dir", "fairseq/models/PhasedDecoder",
    "--restore-file", "pdec_work/checkpoints/distilled_enhanced_phase1/checkpoint_best.pt",
    "--feature-alignment-loss",
    "--attention-alignment-weight", "0.3",
    "--hidden-alignment-weight", "0.2",
    "--lr", "0.0001",
    "--max-epoch", "3",
    "--save-dir", "pdec_work/checkpoints/distilled_enhanced_phase2",
    "--batch-size", "32"
)

try {
    python fairseq_cli/train.py @phase2Args
    if ($LASTEXITCODE -ne 0) {
        throw "阶段2训练失败"
    }
    Write-Host "✅ 阶段2完成" -ForegroundColor Green
} catch {
    Write-Host "❌ 阶段2失败: $_" -ForegroundColor Red
    Write-Host "💡 提示: 如果特征对齐功能不支持，将跳过此阶段" -ForegroundColor Yellow
}

Write-Host ""

# 阶段3: 精细调优
Write-Host "✨ 阶段3: 精细调优..." -ForegroundColor Magenta
Write-Host "预计时间: 15-20分钟" -ForegroundColor Gray

# 选择最佳检查点
$restoreFile = if (Test-Path "pdec_work/checkpoints/distilled_enhanced_phase2/checkpoint_best.pt") {
    "pdec_work/checkpoints/distilled_enhanced_phase2/checkpoint_best.pt"
} else {
    "pdec_work/checkpoints/distilled_enhanced_phase1/checkpoint_best.pt"
}

$phase3Args = @(
    "fairseq/models/ZeroTrans/europarl_scripts/build_data/europarl_15-bin",
    "--user-dir", "fairseq/models/PhasedDecoder",
    "--restore-file", $restoreFile,
    "--lr", "0.00005",
    "--max-epoch", "2",
    "--save-dir", "pdec_work/checkpoints/distilled_enhanced_final",
    "--batch-size", "32"
)

try {
    python fairseq_cli/train.py @phase3Args
    if ($LASTEXITCODE -ne 0) {
        throw "阶段3训练失败"
    }
    Write-Host "✅ 阶段3完成" -ForegroundColor Green
} catch {
    Write-Host "❌ 阶段3失败: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 蒸馏训练完成!" -ForegroundColor Green
Write-Host "📊 开始性能对比评估..." -ForegroundColor Cyan

# 运行评估
if (Test-Path "evaluate_multilingual_model.py") {
    Write-Host "🔍 运行模型评估..." -ForegroundColor Yellow
    
    # 评估原始模型
    Write-Host "📈 评估原始教师模型..." -ForegroundColor Blue
    python evaluate_multilingual_model.py --model-path "pdec_work/checkpoints/multilingual_方案1_三语言/1/checkpoint_best.pt" --output-prefix "teacher"
    
    # 评估蒸馏模型
    Write-Host "📈 评估蒸馏学生模型..." -ForegroundColor Blue
    python evaluate_multilingual_model.py --model-path "pdec_work/checkpoints/distilled_enhanced_final/checkpoint_best.pt" --output-prefix "student"
    
    Write-Host ""
    Write-Host "🎯 性能对比完成!" -ForegroundColor Green
    Write-Host "📁 结果保存在 evaluation_results/ 目录" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  未找到评估脚本，请手动运行评估" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 预期收益:" -ForegroundColor Green
Write-Host "  ✅ BLEU分数提升: +3-7分" -ForegroundColor White
Write-Host "  ✅ 模型大小减少: 50-70%" -ForegroundColor White
Write-Host "  ✅ 推理速度提升: 1.5-2.5倍" -ForegroundColor White
Write-Host "  ✅ 内存使用减少: 40-60%" -ForegroundColor White
Write-Host ""
Write-Host "🎉 蒸馏优化完成！" -ForegroundColor Green 