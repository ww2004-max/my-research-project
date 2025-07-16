#!/bin/bash

# 设置路径
BPEROOT=../../subword-nmt/subword_nmt
RAW_DATA=raw
BPE_DATA=bpe
BPE_MODEL=model

mkdir -p $BPE_DATA $BPE_MODEL

BPESIZE=32000

# 学习 BPE 模型
cat $RAW_DATA/train.all | python $BPEROOT/learn_bpe.py -s $BPESIZE > $BPE_MODEL/bpe.codes

# 应用 BPE 到训练、验证、测试数据
for lang in en de fi pt bg sl it pl hu ro es da nl et cs; do
    for split in train valid test; do
        if [ -f "$RAW_DATA/$split.$lang" ]; then
            python $BPEROOT/apply_bpe.py -c $BPE_MODEL/bpe.codes < $RAW_DATA/$split.$lang > $BPE_DATA/$split.$lang
        fi
    done
done

# 创建语言对文件（train.en_de, valid.de_en 等）
for src in en de fi pt bg sl it pl hu ro es da nl et cs; do
  for tgt in en de fi pt bg sl it pl hu ro es da nl et cs; do
    if [ "$src" != "$tgt" ]; then
      echo "Creating data for $src -> $tgt"
      paste -d "\t" $BPE_DATA/train.$src $BPE_DATA/train.$tgt | awk -F'\t' '{print $1} |||| {print $2}' > $BPE_DATA/train.${src}_${tgt}
      paste -d "\t" $BPE_DATA/valid.$src $BPE_DATA/valid.$tgt | awk -F'\t' '{print $1} |||| {print $2}' > $BPE_DATA/valid.${src}_${tgt}
      paste -d "\t" $BPE_DATA/test.$src $BPE_DATA/test.$tgt | awk -F'\t' '{print $1} |||| {print $2}' > $BPE_DATA/test.${src}_${tgt}
    fi
  done
done

echo "✅ BPE processing completed!"