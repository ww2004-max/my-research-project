from bert_score import BERTScorer
import sys
import os
root_path = ".."
method = sys.argv[1]
model_id = sys.argv[2]
cuda_id = sys.argv[3]

language_sequence = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "fr", "pl", "ro", "th", "fa", "hr", "cs", "de"]


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
        path_ref = os.path.join(root_path, "ted_scripts", "results", method, str(model_id), "{}-{}.r".format(src, tgt))
        path_hypo = os.path.join(root_path, "ted_scripts", "results", method, str(model_id), "{}-{}.h".format(src, tgt))
        ref, hypo = _read_txt_strip_(path_ref), _read_txt_strip_(path_hypo)
        P, R, F = tmp_score.score(hypo, ref, batch_size=100)
        P, R, F = round(P.mean().item() * 100, 2), round(R.mean().item() * 100, 2), round(F.mean().item() * 100, 2)
        print("{}-{}".format(src, tgt))
        print("P: {} R: {} F: {}".format(P, R, F))
        writing_list.append("{}-{}\n".format(src, tgt))
        writing_list.append("P: {} R: {} F: {} \n".format(P, R, F))


file = open(os.path.join(root_path, "ted_scripts", "results", str(method), str(model_id), "{}.bertscore".format(str(model_id))), 'w', encoding='utf-8')
file.writelines(writing_list)
file.close()




