import os

def find_common_texts_and_positions(align_text_dict):
    positions = {}
    anchor_list = align_text_dict[list(align_text_dict.keys())[0]]
    for i, text in enumerate(anchor_list):
        is_common = all(text in align_text_dict[key] for key in align_text_dict)
        if is_common:
            positions[text] = [i] + [align_text_dict[key].index(text) for key in align_text_dict if key != list(align_text_dict.keys())[0]]
    return positions
def create_aligned_lists_no_placeholders(align_text_dict, positions):
    num_common_texts = len(positions)
    aligned_lists = {key: [None] * num_common_texts for key in align_text_dict}
    for idx, (text, pos_list) in enumerate(positions.items()):
        for key, pos in zip(align_text_dict, pos_list):
            aligned_lists[key][idx] = align_text_dict[key][pos]
    return aligned_lists

lang_list = ["en", "ar", "he", "ru", "ko", "it", "ja", "zh", "es", "nl", "vi", "tr", "fr", "pl", "ro", "fa", "hr", "cs", "de"]

def _read_txt_(url):
    file = open(url, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line for line in lines]
def _write_text_(url, data):
    file = open(url, 'w', encoding='utf-8')
    for i in data:
        file.write(i)
    file.close()

root_path = ".."
align_lang_list = ["ar", "he", "zh", "hr", "vi", "ja"]
align_src_dict = {
    "ar": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_ar/test.en")),
    "he": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_he/test.en")),
    "zh": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_zh/test.en")),
    "hr": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_hr/test.en")),
    "vi": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_vi/test.en")),
    "ja": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_ja/test.en")),
}
align_tgt_dict = {
    "ar": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_ar/test.ar")),
    "he": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_he/test.he")),
    "zh": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_zh/test.zh")),
    "hr": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_hr/test.hr")),
    "vi": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_vi/test.vi")),
    "ja": _read_txt_(os.path.join(root_path, "ted_scripts", "build_data","bpe", "en_ja/test.ja")),
}

positions = find_common_texts_and_positions(align_src_dict)
common_texts = list(positions.keys())
_write_text_(os.path.join(root_path, "ted_scripts", "build_data","mono", "en.txt"), common_texts)
aligned_lists = create_aligned_lists_no_placeholders(align_tgt_dict, positions)
for lang in align_lang_list:
    _write_text_(os.path.join(root_path, "ted_scripts", "build_data","mono", f"{lang}.txt"), aligned_lists[lang])