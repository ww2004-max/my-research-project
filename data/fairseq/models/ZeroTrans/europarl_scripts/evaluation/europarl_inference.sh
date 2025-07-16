#!/usr/bin/env bash

METHOD=$1
ID=$2

# 设置绝对路径
FAIRSEQ_ROOT="/c/Users/33491/PycharmProjects/machine/fairseq"
PROJECT_ROOT="$FAIRSEQ_ROOT/models/ZeroTrans"
DATA_PATH="$PROJECT_ROOT/europarl_15-bin"
SCRIPTS_PATH="$PROJECT_ROOT/europarl_scripts"
MOSES_DIR="$FAIRSEQ_ROOT/moses"

SAVE_PATH="$SCRIPTS_PATH/checkpoints/$METHOD/$ID"
RESULTS_PATH="$SCRIPTS_PATH/results/$METHOD/$ID"

mkdir -p "$RESULTS_PATH"

# 进入 fairseq 主目录
cd "$FAIRSEQ_ROOT" || exit 1

# 检查是否有 checkpoint
if [ ! -f "$SAVE_PATH/checkpoint.best_loss_0.pt" ]; then
    echo "Checkpoint not found at $SAVE_PATH/checkpoint.best_loss_*.pt"
    exit 1
fi

# 平均多个 checkpoint
checkpoints=$(ls "$SAVE_PATH"/checkpoint.best_loss_* | tr '\n' ' ')
python3 "$FAIRSEQ_ROOT/scripts/average_checkpoints.py" \
  --inputs $checkpoints \
  --output "$SAVE_PATH/checkpoint_averaged.pt"

# 设置模型参数
DIR=""
TASK="translation_multi_simple_epoch"
if [ "$METHOD" == "zero" ]; then
  DIR="--user-dir models/ZeroTrans"
  TASK="translation_multi_simple_epoch_zero"
fi

# 支持的语言列表
langs=("en" "de" "fi" "pt" "bg" "sl" "it" "pl" "hu" "ro" "es" "da" "nl" "et" "cs")
lang_pairs="de-en,en-de,nl-en,en-nl,da-en,en-da,es-en,en-es,pt-en,en-pt,ro-en,en-ro,it-en,en-it,sl-en,en-sl,pl-en,en-pl,cs-en,en-cs,bg-en,en-bg,fi-en,en-fi,hu-en,en-hu,et-en,en-et"

# 开始推理
for i in "${!langs[@]}"; do
  src=${langs[$i]}
  for j in "${!langs[@]}"; do
    tgt=${langs[$j]}
    if [[ "$src" != "$tgt" ]]; then
      echo "Generating: $src-$tgt"

      RAW_OUTPUT="$RESULTS_PATH/${src}-${tgt}.raw.txt"
      HYP_FILE="$RESULTS_PATH/${src}-${tgt}.h"
      REF_FILE="$RESULTS_PATH/${src}-${tgt}.r"
      DETOK_HYP="$RESULTS_PATH/${src}-${tgt}.detok.h"
      DETOK_REF="$RESULTS_PATH/${src}-${tgt}.detok.r"
      BLEU_LOG="$RESULTS_PATH/${ID}.sacrebleu"

      # 生成翻译
      CUDA_VISIBLE_DEVICES=0 fairseq-generate "$DATA_PATH" \
        --gen-subset test \
        $DIR \
        -s "$src" -t "$tgt" \
        --langs "en,de,nl,da,es,pt,ro,it,sl,pl,cs,bg,fi,hu,et" \
        --lang-pairs "$lang_pairs" \
        --path "$SAVE_PATH/checkpoint_averaged.pt" \
        --remove-bpe sentencepiece \
        --task "$TASK" \
        --encoder-langtok tgt \
        --beam 4 > "$RAW_OUTPUT"

      # 提取 Hypothesis 和 Reference
      grep ^H "$RAW_OUTPUT" | cut -f3 > "$HYP_FILE"
      grep ^T "$RAW_OUTPUT" | cut -f3 > "$REF_FILE"

      # 去 BPE
      sed 's/@@ //g' "$HYP_FILE" > "$DETOK_HYP"
      sed 's/@@ //g' "$REF_FILE" > "$DETOK_REF"

      # 分词（注意使用 Windows 下 perl 脚本）
      perl "$MOSES_DIR/scripts/tokenizer/detokenizer.perl" -threads 32 -l "$tgt" < "$DETOK_HYP" > "${DETOK_HYP}_final"
      perl "$MOSES_DIR/scripts/tokenizer/detokenizer.perl" -threads 32 -l "$tgt" < "$DETOK_REF" > "${DETOK_REF}_final"

      mv "${DETOK_HYP}_final" "$DETOK_HYP"
      mv "${DETOK_REF}_final" "$DETOK_REF"

      rm "$HYP_FILE" "$REF_FILE" "$DETOK_HYP" "$DETOK_REF"

      # 计算 BLEU
      sacrebleu "$DETOK_REF" < "$DETOK_HYP" >> "$BLEU_LOG"

      echo "Finished $src-$tgt"
    fi
  done
done

# 计算 BERTScore
python "$SCRIPTS_PATH/evaluation/europarl_bertscore.py" "$METHOD" "$ID" 0

echo "All evaluations completed."