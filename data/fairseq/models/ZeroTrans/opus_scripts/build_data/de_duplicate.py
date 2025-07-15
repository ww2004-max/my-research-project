import sys
import os
import random


if __name__ == "__main__":
    random.seed(0)
    data_dir = sys.argv[1]
    orig_data = os.path.join(data_dir, "opus-100-corpus", "v1.0")
    orig_supervised = os.path.join(orig_data, "supervised")
    orig_zeroshot = os.path.join(orig_data, "zero-shot")

    # de-duplicate supervised data
    supervised_lpairs = os.listdir(orig_supervised)
    excluded_lang = {'an', 'dz', 'hy', 'mn', 'yo'}
    for lpair in supervised_lpairs:
        lpair_dir = os.path.join(orig_supervised, lpair)
        src, tgt = lpair.split('-')
        if src in excluded_lang or tgt in excluded_lang:
            continue
        if os.path.exists(os.path.join(lpair_dir, f"opus.{lpair}-train-rebuilt.{tgt}")):
            continue

        test_src =  os.path.join(lpair_dir, f"opus.{lpair}-test.{src}")
        test_tgt =  os.path.join(lpair_dir, f"opus.{lpair}-test.{tgt}")
        with open(test_src) as src_file, open(test_tgt) as tgt_file:
            test_data = list(zip(src_file, tgt_file))
            test_length = len(test_data)
            test_data = set(test_data)

        valid_src =  os.path.join(lpair_dir, f"opus.{lpair}-dev.{src}")
        valid_tgt =  os.path.join(lpair_dir, f"opus.{lpair}-dev.{tgt}")
        with open(valid_src) as src_file, open(valid_tgt) as tgt_file:
            valid_data = list(zip(src_file, tgt_file))
            valid_length = len(valid_data)
            valid_data = list(set(valid_data) - test_data)

        train_src =  os.path.join(lpair_dir, f"opus.{lpair}-train.{src}")
        train_tgt =  os.path.join(lpair_dir, f"opus.{lpair}-train.{tgt}")
        with open(train_src) as src_file, open(train_tgt) as tgt_file:
            tmp_data = set(zip(src_file, tgt_file))
            train_data = list(tmp_data - test_data)

        # supplement dev set from training set
        supply_size = valid_length - len(valid_data)
        random.shuffle(train_data)
        if supply_size > 0:
            valid_data += train_data[-supply_size:]
            train_data = train_data[:-supply_size]

        valid_src =  os.path.join(lpair_dir, f"opus.{lpair}-dev-rebuilt.{src}")
        valid_tgt =  os.path.join(lpair_dir, f"opus.{lpair}-dev-rebuilt.{tgt}")
        with open(valid_src, 'w') as src_file, open(valid_tgt, 'w') as tgt_file:
            src_data, tgt_data = list(zip(*valid_data))
            src_file.writelines(src_data)
            tgt_file.writelines(tgt_data)

        train_src =  os.path.join(lpair_dir, f"opus.{lpair}-train-rebuilt.{src}")
        train_tgt =  os.path.join(lpair_dir, f"opus.{lpair}-train-rebuilt.{tgt}")
        with open(train_src, 'w') as src_file, open(train_tgt, 'w') as tgt_file:
            src_data, tgt_data = list(zip(*train_data))
            src_file.writelines(src_data)
            tgt_file.writelines(tgt_data)