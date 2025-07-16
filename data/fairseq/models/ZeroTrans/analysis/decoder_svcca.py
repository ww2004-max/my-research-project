import customize_utils as utils
import sys, os, math

import torch
from fairseq.data.dictionary import Dictionary
from fairseq.models.transformer import TransformerModel
from fairseq.data.data_utils import collate_tokens


DICT_LENGTH = 50000

def get_representation(model, dictionary, src_lang, src_lang_id, tgt_lang, batch_size, layer_wise, layer_num=0):
    src_texts = utils.read_txt(os.path.join(dir_path, "mono", f"{src_lang}.txt"))
    tgt_texts = utils.read_txt(os.path.join(dir_path, "mono", f"{tgt_lang}.txt"))
    
    assert len(src_texts) == len(tgt_texts), "src and tgt are not aligned."
    batch_num = math.ceil(len(src_texts) / batch_size)
    
    outputs = None if not layer_wise else []
    
    for i in range(batch_num):

        start, end = i * batch_size, min((i + 1) * batch_size, len(src_texts))
        batch_src_texts = src_texts[start:end]

        batch_src_texts = [torch.cat([torch.LongTensor([(src_lang_id + DICT_LENGTH)]), dictionary.encode_line(text, add_if_not_exist=False, append_eos=True)], dim=0) for text in batch_src_texts]
        batch_src_texts = collate_tokens(batch_src_texts, pad_idx=1, left_pad=True).cuda()
        
        batch_tgt_texts = tgt_texts[start:end]
        batch_tgt_texts = [dictionary.encode_line(text, add_if_not_exist=False, append_eos=True) for text in batch_tgt_texts]
        batch_tgt_texts = collate_tokens(batch_tgt_texts, pad_idx=1, left_pad=False, move_eos_to_beginning=True)


        encoder_out = model.encoder(batch_src_texts, src_lengths=batch_src_texts.shape[0], return_all_hiddens=False)
        decoder_out = model.decoder(batch_tgt_texts, encoder_out=encoder_out, src_lengths=batch_src_texts.shape[0], return_all_hiddens=True,)
        
        decoder_states = decoder_out[1]["inner_states"]
        mask = (batch_tgt_texts != 1)
        if layer_wise:
            for k, state in enumerate(decoder_states[1:], 1):  # Skip the first layer as it is input embeddings
                # shape of batch_size * model_dim
                sentences = utils.sentence_mean_pooling(state.cpu(), mask)
                if len(outputs) < k:
                    outputs.append(sentences)
                else:
                    outputs[k-1] = torch.cat([outputs[k-1], sentences], dim=0)
        else:
            x = decoder_states[layer_num + 1].cpu()
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
    pkl_filename = f"decoder_{lang_1}_{lang_2}.pkl"
else:
    pkl_filename = f"decoder_layer_{layer_num}_{lang_1}_{lang_2}.pkl"
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

