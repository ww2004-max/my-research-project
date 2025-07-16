#!/bin/bash

# Step 0: 设置全局变量（所有步骤都依赖）
ROOT_PATH="/c/Users/33491/PycharmProjects/machine/fairseq/models/ZeroTrans"
ROW_DATA_PATH="$ROOT_PATH/europarl_scripts/mmcr4nlp/europarl"
MONO_PATH="$ROOT_PATH/europarl_scripts/build_data/mono"
TOKENIZED_PATH="$ROOT_PATH/europarl_scripts/build_data/tokenized"
TOKENIZER="/c/Users/33491/PycharmProjects/machine/fairseq/moses/scripts/tokenizer/tokenizer.perl"
FAIRSEQ_SCRIPTS="/c/Users/33491/PycharmProjects/machine/fairseq/scripts"
SPM_TRAIN="$FAIRSEQ_SCRIPTS/spm_train.py"
SPM_ENCODE="$FAIRSEQ_SCRIPTS/spm_encode.py"
BPESIZE=50000
TRAIN_FILES="$ROOT_PATH/europarl_scripts/build_data/bpe.input-output"
BPE_MODEL_PREFIX="$ROOT_PATH/europarl_scripts/build_data/europarl.bpe"
BPE_MONO_PATH="$ROOT_PATH/europarl_scripts/build_data/bpe_mono"
BPE_PATH="$ROOT_PATH/europarl_scripts/build_data/bpe"
BINARY_PATH=$(cygpath -aw "$ROOT_PATH/europarl_scripts/build_data/europarl_15-bin")

LANGS="en de fi pt bg sl it pl hu ro es da nl et cs"

# Step 1: 创建必要目录
if [ "$1" == "step1" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 1: Creating directories..."
    mkdir -p "$ROOT_PATH/europarl_scripts/logs"
    mkdir -p "$ROOT_PATH/europarl_scripts/checkpoints"
    mkdir -p "$ROOT_PATH/europarl_scripts/results"
    mkdir -p "$MONO_PATH"
    mkdir -p "$TOKENIZED_PATH"
    mkdir -p "$BPE_MONO_PATH"
    mkdir -p "$BPE_PATH"
    mkdir -p "$(cygpath -au "$BINARY_PATH")"
fi

# Step 2: 提取单语数据
if [ "$1" == "step2" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 2: Extracting monolingual data..."
    python "$ROOT_PATH/europarl_scripts/build_data/mono_reader.py" "$ROW_DATA_PATH" "$MONO_PATH"
fi

# Step 3: 分词处理
if [ "$1" == "step3" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 3: Tokenizing monolingual data..."
    for lang in $LANGS; do
      for split in train valid test; do
        file_name="$MONO_PATH/$split.$lang"
        if [ -f "$file_name" ]; then
          cat "$file_name" | perl "$TOKENIZER" -threads 8 -l "$lang" > "$TOKENIZED_PATH/$split.$lang"
        else
          echo "[$(date)] Missing file: $file_name"
        fi
      done
    done
fi

# Step 4: 合并数据并训练 BPE 模型
if [ "$1" == "step4" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 4: Merging data and training BPE model..."
    rm -f "$TRAIN_FILES"
    for lang in $LANGS; do
      filename="$TOKENIZED_PATH/train.$lang"
      if [ -f "$filename" ]; then
        cat "$filename" >> "$TRAIN_FILES"
      else
        echo "[$(date)] Skipping missing file: $filename"
      fi
    done
    python "$SPM_TRAIN" \
        --input="$TRAIN_FILES" \
        --model_prefix="$BPE_MODEL_PREFIX" \
        --vocab_size=$BPESIZE \
        --character_coverage=1.0 \
        --model_type=bpe
fi

# Step 5: 应用 BPE 编码到单语数据
if [ "$1" == "step5" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 5: Applying BPE encoding..."
    for lang in $LANGS; do
      for split in train valid test; do
        input_file="$TOKENIZED_PATH/$split.$lang"
        output_file="$BPE_MONO_PATH/$split.$lang"
        if [ -f "$input_file" ]; then
          python "$SPM_ENCODE" \
              --model "${BPE_MODEL_PREFIX}.model" \
              --output_format=piece \
              --inputs "$input_file" \
              --outputs "$output_file"
        else
          echo "[$(date)] Skipping missing file: $input_file"
        fi
      done
    done
fi

# Step 6: 双语配对
if [ "$1" == "step6" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 6: Pairing bilingual data..."
    python "$ROOT_PATH/europarl_scripts/build_data/pairing.py" "$BPE_MONO_PATH" "$BPE_PATH"
fi
# Step 7: 创建字典 & 二值化数据
if [ "$1" == "step7" ] || [ "$1" == "all" ]; then
    echo "[$(date)] Step 7: Creating dictionary and binarizing data..."
    cut -f 1 "${BPE_MODEL_PREFIX}.vocab" | tail -n +4 | sed 's/$/ 1/g' > "$BINARY_PATH/dict.txt"

    for src in $LANGS; do
      for tgt in $LANGS; do
        if [ "$src" == "$tgt" ]; then continue; fi
        PAIR="${src}_${tgt}"
        TRAIN_PREF="${BPE_PATH}/train.${PAIR}"
        VALID_PREF="${BPE_PATH}/valid.${PAIR}"
        TEST_PREF="${BPE_PATH}/test.${PAIR}"

        # 检查是否有源语言文件存在
        if [ -f "${TRAIN_PREF}.${src}" ] || [ -f "${VALID_PREF}.${src}" ] || [ -f "${TEST_PREF}.${src}" ]; then
          echo "[$(date)] Processing $src->$tgt"
          fairseq-preprocess --task translation \
            --source-lang "$src" --target-lang "$tgt" \
            --trainpref "$TRAIN_PREF" \
            --validpref "$VALID_PREF" \
            --testpref "$TEST_PREF" \
            --destdir "$BINARY_PATH" \
            --padding-factor 1 \
            --workers 4 \
            --srcdict "$BINARY_PATH/dict.txt" \
            --tgtdict "$BINARY_PATH/dict.txt" \
            &> "$BINARY_PATH/preprocess_${src}_${tgt}.log"
          echo "[$(date)] Finished $src->$tgt, log saved to $BINARY_PATH/preprocess_${src}_${tgt}.log"
        else
          echo "[$(date)] Skipping $src->$tgt: no files found"
        fi
      done
    done
fi