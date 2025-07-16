import sys

language_sequence = ["en", "de", "fi", "pt", "bg", 'sl', "it",  "pl", "hu", "ro", "es", "da", "nl", "et", "cs"]
other_language = ["de", "fi", "pt", "bg", 'sl', "it",  "pl", "hu", "ro", "es", "da", "nl", "et", "cs"]
language_dict = {'en': 1, 'de': 2, 'nl': 3, 'da': 4, 'es': 5, 'pt': 6, 'ro': 7,
                 'it': 8, 'sl': 9, 'pl': 10, 'cs': 11, 'bg': 12, "fi": 13, "hu": 14, "et": 15}

europarl_path = sys.argv[1]
save_path = sys.argv[2]

def _read_txt_(url):
    file = open(url, 'r', encoding='utf-8')
    seg_list = []
    lines = file.readlines()
    for i in range(len(lines)):
        seg_list.append(lines[i])
        pass
    file.close()
    return seg_list


train_mono_list = [_read_txt_("{}/21lingual/train.21langmultiway.europarl-v7.{}-en.{}".format(europarl_path, i, i)) for i in other_language]
valid_mono_list = [_read_txt_("{}/21lingual/dev.21langmultiway.europarl-v7.{}-en.{}".format(europarl_path, i, i )) for i in other_language]
test_mono_list = [_read_txt_("{}/21lingual/test.21langmultiway.europarl-v7.{}-en.{}".format(europarl_path, i, i )) for i in other_language]

for i in range(len(other_language)):
    tmp = open("{}/train.{}".format(save_path, other_language[i]), 'w', encoding='utf-8')
    tmp.writelines(train_mono_list[i])
    tmp.close()

    tmp = open("{}/valid.{}".format(save_path, other_language[i]), 'w', encoding='utf-8')
    tmp.writelines(valid_mono_list[i])
    tmp.close()

    tmp = open("{}/test.{}".format(save_path, other_language[i]), 'w', encoding='utf-8')
    tmp.writelines(test_mono_list[i])
    tmp.close()

tmp = open("{}/train.en".format(save_path), 'w', encoding='utf-8')
tmp.writelines(_read_txt_("{}/21lingual/train.21langmultiway.europarl-v7.de-en.en".format(europarl_path)))
tmp.close()

tmp = open("{}/valid.en".format(save_path), 'w', encoding='utf-8')
tmp.writelines(_read_txt_("{}/21lingual/dev.21langmultiway.europarl-v7.de-en.en".format(europarl_path)))
tmp.close()

tmp = open("{}/test.en".format(save_path), 'w', encoding='utf-8')
tmp.writelines(_read_txt_("{}/21lingual/test.21langmultiway.europarl-v7.de-en.en".format(europarl_path)))
tmp.close()