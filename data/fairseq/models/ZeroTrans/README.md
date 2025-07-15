# ZeroTrans

This is the repository for the paper:  
**Languages Transferred Within the Encoder: On Representation Transfer in Zero-Shot Multilingual Translation**

---

Prerequisite:
1. install [Fairseq](https://github.com/facebookresearch/fairseq)
2. download [Moses](https://github.com/moses-smt/mosesdecoder)
3. run commands
```bash
mkdir fairseq/models
mv ZeroTrans fairseq/models/ZeroTrans
mkdir moses
mv mosesdecoder/scripts moses
```
---
An example for running experiments:
```bash
cd europal_scripts
# please update the root_path (or ROOT_PATH) in those files
# build dataset
bash build_data/preprocess.sh
# run
bash train/vanilla.sh
# inference
# bash evaluation/europarl_inference.sh # integrated in the training bash
# sort results
python excel/make_table.py
```


