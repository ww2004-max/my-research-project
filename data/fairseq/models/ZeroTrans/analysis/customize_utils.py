import cca_core
import torch
import pickle
import numpy as np

def read_txt(path):
    file = open(path, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    return [line.strip() for line in lines]

def write_txt(path, contents, line_wise: bool = False, appending: bool = False, encoding_mode: str = 'utf-8'):
    mode = 'a' if appending else 'w'
    if line_wise:
        contents = [item + "\n" for item in contents]
    with open(path, mode, encoding=encoding_mode) as file:
        file.writelines(contents)

def save_dict_or_list(path, contents):
    with open(path, "wb") as f:
        pickle.dump(contents, f)

def load_dict_or_list(path):
    with open(path, "wb") as f:
        tmp = pickle.load(f)
    return tmp

def compute_svcca_scores(x: torch.Tensor,
                         y: torch.Tensor,
                         need_std: bool = False):
    assert x.shape == y.shape, "Inputs do not have the same shape."
    assert x.dim() <= 2, "Inputs should be of shape (n) or (n, m)."
    
    if x.dim() == 1:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    
    scores = []
    for i in range(x.shape[0]):
        tmp_x, tmp_y = x[i].unsqueeze(0).numpy(), y[i].unsqueeze(0).numpy()
        tmp_score = cca_core.get_cca_similarity(tmp_x, tmp_y, verbose=False)["cca_coef1"][0]
        scores.append(tmp_score)
        
    average = np.mean(scores)
    if not need_std:
        return scores, average
    else:
        std_dev = np.std(scores)
        return scores, average, std_dev

def sentence_mean_pooling(x: torch.Tensor, mask: torch.Tensor):
    return (x.transpose(0, 1) * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)
