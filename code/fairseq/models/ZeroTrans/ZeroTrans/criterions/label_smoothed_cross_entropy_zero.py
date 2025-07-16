from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils
import math
import torch.nn as nn
from typing import List, Optional
import torch
import copy
import random

random.seed(0)

@register_criterion("label_smoothed_cross_entropy_zero")
class LabelSmoothedCrossEntropyCriterionZero(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 contrastive_lambda=1.0, temperature=1.0, dec_dim=0, negative_sampling_number=5, contrastive_learning=False, contrastive_position=6):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.negative_samples_number = negative_sampling_number
        self.contrastive_lambda = contrastive_lambda
        self.temperature = temperature
        self.dec_dim = dec_dim
        self.contrastive_learning = contrastive_learning
        self.contrastive_position = contrastive_position

        self.similarity = nn.CosineSimilarity(dim=-1)

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-lambda", type=float,
                            default=1.0,
                            help="The contrastive loss weight")
        parser.add_argument("--temperature", type=float,
                            default=1.0, )
        parser.add_argument("--dec-dim", type=int, default=0, )
        parser.add_argument("--negative-sampling-number", type=int, default=5, )
        parser.add_argument("--contrastive-learning", type=bool, default=False, )
        parser.add_argument("--contrastive-position", type=int, default=6, )

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        output_decoder_per_layer: List[Optional[torch.Tensor]] = net_output[1]["inner_states"]
        prev_output_tokens = net_output[1]["prev_output_tokens"]
        language_num = net_output[1]["language_num"]
        tgt_direction = net_output[1]["tgt_direction"]
        if self.contrastive_learning:
            positive_indices, negative_indices, k_to_language = optimal_sampling(language_num, self.negative_samples_number, tgt_direction)
            mask = (prev_output_tokens != self.padding_idx)
            # times mask, then divide by token_num for each sentence
            anchors = (output_decoder_per_layer[self.contrastive_position].transpose(0, 1) * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)
            # if decoder_normalize_before = True
            # anchors = torch.nn.functional.normalize(anchors, p=2, dim=-1)
            contrastive_loss = self.compute_contrastive_loss(anchors, positive_indices, negative_indices, k_to_language, tgt_direction, self.negative_samples_number, self.dec_dim)
        else:
            contrastive_loss = torch.Tensor([0]).cuda()

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        all_loss = loss + contrastive_loss * self.contrastive_lambda
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "contrastive_loss": contrastive_loss.data,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return all_loss, sample_size, logging_output

    def compute_contrastive_loss(self, anchors, positive_indices, negative_indices, k_to_language, tgt_direction, k,
                                 dim):
        positives = torch.index_select(anchors, 0, positive_indices)
        negatives = torch.index_select(anchors, 0, negative_indices)

        if dim != 0:
            anchors, positives, negatives = anchors[:, :dim], positives[:, :dim], negatives[:, :dim]
            anchors = torch.nn.functional.normalize(anchors, dim=1)
            positives = torch.nn.functional.normalize(positives, dim=1)
            negatives = torch.nn.functional.normalize(negatives, dim=1)

        positive_similarity = self.similarity(anchors, positives)
        positive_similarity = positive_similarity.unsqueeze(0)
        if len(set(k_to_language)) == 1 and k_to_language[0] == k:
            tmp_k = k
            negative_similarity = self.similarity(anchors.expand(tmp_k, anchors.shape[0], anchors.shape[1]),
                                                  negatives.view(tmp_k, int(negatives.shape[0] / tmp_k),
                                                                 anchors.shape[1]))
            similarity = torch.cat([positive_similarity, negative_similarity], dim=0)
            loss = -nn.LogSoftmax(0)(torch.div(similarity, self.temperature))[0].sum()
            return loss
        elif len(set(k_to_language)) == 1 and k_to_language[0] == 0:
            return torch.Tensor([0]).cuda()
        else:
            negatives = negatives.reshape(k, int(negatives.shape[0] / k), anchors.shape[1])
            language_with_dynamic_k = []
            for i in range(len(k_to_language)):
                if k_to_language[i] != k:
                    language_with_dynamic_k += [i + 1]
            isin = torch.isin(tgt_direction, torch.tensor(language_with_dynamic_k).cuda())
            valid_indices = torch.where(isin == 0)[0]
            valid_loss = torch.tensor([0]).cuda()
            if valid_indices.shape[0] != 0:
                valid_negatives = torch.index_select(negatives, 1, valid_indices)
                valid_anchors = torch.index_select(anchors, 0, valid_indices)
                valid_positive_similarity = torch.index_select(positive_similarity, 1, valid_indices)
                tmp_k = k
                valid_negative_similarity = self.similarity(valid_anchors.expand(tmp_k, valid_anchors.shape[0],
                                                                                 valid_anchors.shape[1]),
                                                            valid_negatives)
                valid_similarity = torch.cat([valid_positive_similarity, valid_negative_similarity], dim=0)
                valid_loss = -nn.LogSoftmax(0)(torch.div(valid_similarity, self.temperature))[0].sum()
            dynamic_loss = None
            for language in language_with_dynamic_k:
                tmp_k = k_to_language[language - 1]
                if tmp_k == 0:
                    continue
                else:
                    tmp_indices = torch.nonzero((tgt_direction == language)).reshape(-1)
                    tmp_negatives = torch.index_select(negatives[:tmp_k, :, :], 1, tmp_indices)
                    tmp_anchors = torch.index_select(anchors, 0, tmp_indices)
                    tmp_positive_similarity = torch.index_select(positive_similarity, 1, tmp_indices)
                    tmp_negative_similarity = self.similarity(tmp_anchors.expand(tmp_k, tmp_anchors.shape[0],
                                                                                 tmp_anchors.shape[1]),
                                                              tmp_negatives)
                    tmp_similarity = torch.cat([tmp_positive_similarity, tmp_negative_similarity], dim=0)
                    tmp_loss = -nn.LogSoftmax(0)(torch.div(tmp_similarity, self.temperature))[0].sum()
                    if dynamic_loss is None:
                        dynamic_loss = tmp_loss
                    else:
                        dynamic_loss += tmp_loss
            if dynamic_loss is not None:
                return valid_loss + dynamic_loss
            else:
                return valid_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )


def shift_derange(arr):
    if len(arr) > 2:
        step = random.randint(1, len(arr) - 1)
        arr[:] = arr[step:] + arr[:step]
    if len(arr) == 2:
        arr[0], arr[1] = arr[1], arr[0]


def negative_sample(number, indices, negative_samples_number):
    tmp = []
    for _ in range(number):
        random.shuffle(indices)
        tmp += [indices[0:negative_samples_number]]
    return tmp


def optimal_sampling(language_num, k, tgt_direction):
    indices_each_lang, positive_indices, negative_indices = [], [], []
    optional_negative = []
    k_to_language = []
    for i in range(1, language_num + 1):
        # get optional indices for selecting positive samples for each language
        positives = torch.nonzero(tgt_direction == i).reshape(-1).tolist()
        indices_each_lang += positives
        copy_positives = copy.deepcopy(positives)
        # derange algorithm to completely shuffle the list

        shift_derange(copy_positives)
        positive_indices += copy_positives
        tmp_negative = torch.nonzero((tgt_direction != i)).reshape(-1).tolist()
        if len(tmp_negative) < k:
            k_to_language.append(len(tmp_negative))
        else:
            k_to_language.append(k)
        optional_negative.append(tmp_negative)

    tgt_direction_list = tgt_direction.tolist()

    for element in tgt_direction_list:
        indices = optional_negative[element - 1]
        tmp_k = len(indices)
        if tmp_k >= k:
            random.shuffle(indices)
            negative_indices += optional_negative[element - 1][0:k]
        elif 0 < tmp_k < k:
            negative_indices += optional_negative[element - 1][0:tmp_k]
            negative_indices += [0] * (k - tmp_k)
        else:
            negative_indices += [0] * k

    # sampling is based on the sequence of languages, reorder it to make it back to the sequence of batch
    reorder = torch.tensor([indices_each_lang.index(i) for i in range(len(indices_each_lang))]).cuda()
    positive_indices = torch.tensor(positive_indices).cuda()
    positive_indices = torch.index_select(positive_indices, 0, reorder)
    negative_indices = torch.tensor(negative_indices).cuda()

    indices_for_reset_negatives = []
    for i in range(k):
        indices_for_reset_negatives += [i + j * k for j in range(len(tgt_direction))]
    indices_for_reset_negatives = torch.LongTensor(indices_for_reset_negatives).cuda()
    negative_indices = torch.index_select(negative_indices, 0, indices_for_reset_negatives)

    return positive_indices, negative_indices, k_to_language
