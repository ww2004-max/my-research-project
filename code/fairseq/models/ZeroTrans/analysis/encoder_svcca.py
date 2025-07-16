import customize_utils as utils
import sys, os, math

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens

DICT_LENGTH = 50000

def get_representation(model, dictionary, src_lang, src_lang_id, batch_size, layer_wise, layer_num=0):
    texts = utils.read_txt(os.path.join(dir_path, "mono", f"{src_lang}.txt"))
    
    batch_num = math.ceil(len(texts) / batch_size)
    
    outputs = None if not layer_wise else []
    
    for i in range(batch_num):

        start, end = i * batch_size, min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_src = [torch.cat([torch.LongTensor([(src_lang_id + DICT_LENGTH)]), dictionary.encode_line(text, add_if_not_exist=False, append_eos=True)], dim=0) for text in batch_texts]
        src = collate_tokens(batch_src, pad_idx=1, left_pad=True).cuda()

        tmp_output = model.encoder(src, src_lengths=src.shape[0], return_all_hiddens=True)
        mask = ~tmp_output["encoder_padding_mask"][0].cpu()
        encoder_states = tmp_output["encoder_states"]
        
        if layer_wise:
            for k, state in enumerate(encoder_states[1:], 1):  # Skip the first layer as it is input embeddings
                # shape of batch_size * model_dim
                sentences = utils.sentence_mean_pooling(state.cpu(), mask)
                if len(outputs) < k:
                    outputs.append(sentences)
                else:
                    outputs[k-1] = torch.cat([outputs[k-1], sentences], dim=0)
        else:
            x = encoder_states[layer_num + 1].cpu()
            sentences = utils.sentence_mean_pooling(x, mask)
            outputs = sentences if outputs is None else torch.cat([outputs, sentences], dim=0)
    return outputs




lang_1, lang_1_id = sys.argv[1], sys.argv[2]
lang_2, lang_2_id = sys.argv[3], sys.argv[4]
batch_size = sys.argv[5]
dir_path = sys.argv[6]
data_path = sys.argv[7]
checkpoint = sys.argv[8]
layer_num = 0 if int(sys.argv) <= 9 else int(sys.argv[9])
layer_wise = layer_num == 0

os.makedirs(os.path.join(dir_path, "pkl"), exist_ok=True)
if layer_wise:
    pkl_filename = f"encoder_{lang_1}_{lang_2}.pkl"
else:
    pkl_filename = f"encoder_layer_{layer_num}_{lang_1}_{lang_2}.pkl"
pkl_path = os.path.join(dir_path, "pkl", pkl_filename)

if os.path.exists(pkl_path):
    raise ValueError(f"file {pkl_path} is existed.")

dictionary = Dictionary.load(os.path.join(data_path, "dict.en.txt"))
model = TransformerModel.from_pretrained(checkpoint, checkpoint_file="checkpoint.pt", data_name_or_path=data_path).models[0]
model.eval().cuda()


real = get_representation(model, dictionary, lang_1, lang_2_id, batch_size, layer_wise, layer_num)
identity_1 = get_representation(model, dictionary, lang_1, lang_1_id, batch_size, layer_wise, layer_num)
identity_2 = get_representation(model, dictionary, lang_2, lang_2_id, batch_size, layer_wise, layer_num)


results_dict = {}
for idx, (first, second) in enumerate([(real, identity_1), (real, identity_2), (identity_1, identity_2)], 1):
    averaged_score, mean, std_err = utils.compute_svcca_scores(first, second, need_std=True)
    results_dict[idx] = [averaged_score, mean, std_err]

utils.save_dict_or_list(path=pkl_path, contents=results_dict)

