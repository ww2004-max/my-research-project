from bert_score import BERTScorer
import sys
import os

root_path="../europarl_scripts"
method = sys.argv[1]
model_id = sys.argv[2]
cuda_id = sys.argv[3]
language_sequence = ["en", "de", "nl", "da", "es", "pt", "ro", "it", "sl", "pl", "cs", "bg", "fi", "hu", "et"]


def _read_txt_strip_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]


writing_list = []
for i, tgt in enumerate(language_sequence):
    tmp_score = BERTScorer(lang=tgt, device="cuda:{}".format(cuda_id))
    for j, src in enumerate(language_sequence):
        if i == j: continue
        ref = _read_txt_strip_(os.path.join(root_path, "results", method, str(model_id), f"{src}-{tgt}.detok.r"))
        hypo = _read_txt_strip_(os.path.join(root_path, "results", method, str(model_id), f"{src}-{tgt}.detok.h"))
        P, R, F = tmp_score.score(hypo, ref, batch_size=100)
        P, R, F = round(P.mean().item() * 100, 2), round(R.mean().item() * 100, 2), round(F.mean().item() * 100, 2)
        print("{}-{}".format(src, tgt))
        print("P: {} R: {} F: {}".format(P, R, F))
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))


file = open(os.path.join(root_path, "results", method, str(model_id), f"{str(model_id)}.bertscore"), 'w', encoding='utf-8')
file.writelines(writing_list)
file.close()




