#! /bin/bash

# This script takes 3 arguments: input_corpus, dev set size and test set size
# This then splits the corpus into 3 parts: train, dev and test

corpus=$1
dev_size=$2
test_size=$3

total_lines=`wc -l $corpus | cut -d ' ' -f1`

dev_test_total=`echo $dev_size+$test_size | bc`

train_size=`echo $total_lines-$dev_test_total | bc`

echo "Total lines: " $total_lines
echo "Train size: " $train_size " from line 1 to line " $train_size
echo "Development set size: " $dev_size " from line "`echo $train_size+1 | bc` " to line " `echo $train_size+$dev_size | bc`
echo "Test set size: " $test_size " from line "`echo $train_size+1+$dev_size | bc` " to line " `echo $train_size+$dev_size+$test_size | bc`
### Cut test from end of corpus

tail -$test_size $corpus > test.$corpus

### Cut dev from the corpus

tail -$dev_test_total $corpus | head -$dev_size > dev.$corpus

### Cut train from the corpus

head -$train_size $corpus > train.$corpus