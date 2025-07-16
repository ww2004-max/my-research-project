import sys

language_sequence = ["en", "de", "fi", "pt", "bg", 'sl', "it",  "pl", "hu", "ro", "es", "da", "nl", "et", "cs"]
other_language = ["de", "fi", "pt", "bg", 'sl', "it",  "pl", "hu", "ro", "es", "da", "nl", "et", "cs"]
language_dict = {'en': 1, 'de': 2, 'nl': 3, 'da': 4, 'es': 5, 'pt': 6, 'ro': 7,
                 'it': 8, 'sl': 9, 'pl': 10, 'cs': 11, 'bg': 12, "fi": 13, "hu": 14, "et": 15}


bpe_mono_path = sys.argv[1]
save_path = sys.argv[2]

def _read_text_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [lines[i] for i in range(len(lines))]


def _write_text_(url, data):
    file = open(url, 'w', encoding='utf-8')
    file.writelines(data)
    file.close()


test_mono_list = [_read_text_("{}/test.{}".format(bpe_mono_path, i, i)) for i in language_sequence]
for i in range(len(language_sequence)):
    for j in range(len(language_sequence)):
        if i == j: continue
        src, tgt = language_sequence[i], language_sequence[j]
        _write_text_(url="{}/test.{}_{}.{}".format(save_path, src, tgt, src), data=test_mono_list[i])
        _write_text_(url="{}/test.{}_{}.{}".format(save_path, src, tgt, tgt), data=test_mono_list[j])


train_mono_list = [_read_text_("{}/train.{}".format(bpe_mono_path, i, i)) for i in language_sequence]
valid_mono_list = [_read_text_("{}/valid.{}".format(bpe_mono_path, i, i)) for i in language_sequence]
for i in range(len(language_sequence)):
    if i == 0: continue
    # en 2 many
    src, tgt = language_sequence[0], language_sequence[i]
    _write_text_(url="{}/train.{}_{}.{}".format(save_path, src, tgt, src), data=train_mono_list[0])
    _write_text_(url="{}/train.{}_{}.{}".format(save_path, src, tgt, tgt), data=train_mono_list[i])
    _write_text_(url="{}/valid.{}_{}.{}".format(save_path, src, tgt, src), data=valid_mono_list[0])
    _write_text_(url="{}/valid.{}_{}.{}".format(save_path, src, tgt, tgt), data=valid_mono_list[i])

    # many 2 en
    tgt, src = language_sequence[0], language_sequence[i]
    _write_text_(url="{}/train.{}_{}.{}".format(save_path, src, tgt, src), data=train_mono_list[i])
    _write_text_(url="{}/train.{}_{}.{}".format(save_path, src, tgt, tgt), data=train_mono_list[0])
    _write_text_(url="{}/valid.{}_{}.{}".format(save_path, src, tgt, src), data=valid_mono_list[0])
    _write_text_(url="{}/valid.{}_{}.{}".format(save_path, src, tgt, tgt), data=valid_mono_list[i])
