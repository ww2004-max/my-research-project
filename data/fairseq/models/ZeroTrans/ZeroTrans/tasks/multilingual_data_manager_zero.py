import logging
import torch
from fairseq.data import RawLabelDataset
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager
from ZeroTrans.data import LanguagePairDatasetZero


logger = logging.getLogger(__name__)

SRC_DICT_NAME = 'src'
TGT_DICT_NAME = 'tgt'


class MultilingualDatasetManagerZero(MultilingualDatasetManager):

    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return MultilingualDatasetManagerZero(
            args, lang_pairs, langs, dicts, sampling_method
        )

    def load_langpair_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        src_dataset_transform_func=lambda dataset: dataset,
        tgt_dataset_transform_func=lambda dataset: dataset,
        src_lang_id=None,
        tgt_lang_id=None,
        langpairs_sharing_datasets=None,
    ):
        norm_direction = "-".join(sorted([src, tgt]))
        if langpairs_sharing_datasets is not None:
            src_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src), "NotInCache"
            )
            tgt_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, tgt), "NotInCache"
            )
            align_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src, tgt), "NotInCache"
            )

        # a hack: any one is not in cache, we need to reload them
        if (
            langpairs_sharing_datasets is None
            or src_dataset == "NotInCache"
            or tgt_dataset == "NotInCache"
            or align_dataset == "NotInCache"
            or split != getattr(self.args, "train_subset", None)
        ):
            # source and target datasets can be reused in reversed directions to save memory
            # reversed directions of valid and test data will not share source and target datasets
            src_dataset, tgt_dataset, align_dataset = self.load_lang_dataset(
                data_path,
                split,
                src,
                src_dict,
                tgt,
                tgt_dict,
                combine,
                dataset_impl,
                upsample_primary,
                max_source_positions=max_source_positions,
                prepend_bos=prepend_bos,
                load_alignments=load_alignments,
                truncate_source=truncate_source,
            )
            src_dataset = src_dataset_transform_func(src_dataset)
            tgt_dataset = tgt_dataset_transform_func(tgt_dataset)
            if langpairs_sharing_datasets is not None:
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src)
                ] = src_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, tgt)
                ] = tgt_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src, tgt)
                ] = align_dataset
                if align_dataset is None:
                    # no align data so flag the reverse direction as well in sharing
                    langpairs_sharing_datasets[
                        (data_path, split, norm_direction, tgt, src)
                    ] = align_dataset
        else:
            logger.info(
                f"Reusing source and target datasets of [{split}] {tgt}-{src} for reversed direction: "
                f"[{split}] {src}-{tgt}: src length={len(src_dataset)}; tgt length={len(tgt_dataset)}"
            )

        # europarl 15
        language_dict = {'en': 1, 'de': 2, 'nl': 3, 'da': 4, 'es': 5, 'pt': 6, 'ro': 7,
                         'it': 8, 'sl': 9, 'pl': 10, 'cs': 11, 'bg': 12, "fi": 13, "hu": 14, "et": 15}
        # ted 19 mbart
        # language_dict = {'en_XX': 1, 'ar_AR': 2, 'he_IL': 3, 'ru_RU': 4, 'ko_KR': 5, 'it_IT': 6, 'ja_XX': 7,
        #                  'zh_CN': 8, 'es_XX': 9, 'nl_XX': 10, "vi_VN": 11, "tr_TR": 12, "fr_XX":13,
        #                  "pl_PL": 14, "ro_RO":15, "fa_IR":16, "hr_HR":17, "cs_CZ":18, "de_DE": 19}
        # ted 19
        # language_dict = {'en': 1, 'ar': 2, 'he': 3, 'ru': 4, 'ko': 5, 'it': 6, 'ja': 7,
        #                  'zh': 8, 'es': 9, 'nl': 10, "vi": 11, "tr": 12, "fr":13,
        #                  "pl": 14, "ro":15, "fa":16, "hr":17, "cs":18, "de": 19}
        # opus 100
        # language_dict = {"en": 1,"af": 2,"am": 3,"ar": 4,"as": 5,"az": 6,"be": 7,"bg": 8,"bn": 9,"br": 10,
        #                  "bs": 11,"ca": 12,"cs": 13,"cy": 14,"da": 15,"de": 16,"el": 17,"eo": 18,"es": 19,"et": 20,
        #                  "eu": 21,"fa": 22,"fi": 23,"fr": 24,"fy": 25,"ga": 26,"gd": 27,"gl": 28,"gu": 29,"ha": 30,
        #                  "he": 31,"hi": 32,"hr": 33,"hu": 34,"id": 35,"ig": 36,"is": 37,"it": 38,"ja": 39,"ka": 40,
        #                  "kk": 41,"km": 42,"kn": 43,"ko": 44,"ku": 45,"ky": 46,"li": 47,"lt": 48,"lv": 49,"mg": 50,
        #                  "mk": 51,"ml": 52,"mr": 53,"ms": 54,"mt": 55,"my": 56,"nb": 57,"ne": 58,"nl": 59,"nn": 60,
        #                  "no": 61,"oc": 62,"or": 63,"pa": 64,"pl": 65,"ps": 66,"pt": 67,"ro": 68,"ru": 69,"rw": 70,
        #                  "se": 71,"sh": 72,"si": 73,"sk": 74,"sl": 75,"sq": 76,"sr": 77,"sv": 78,"ta": 79,"te": 80,
        #                  "tg": 81,"th": 82,"tk": 83,"tr": 84,"tt": 85,"ug": 86,"uk": 87,"ur": 88,"uz": 89,"vi": 90,
        #                  "wa": 91,"xh": 92,"yi": 93,"zh": 94,"zu": 95}

        single_src_direction = language_dict[src]
        single_tgt_direction = language_dict[tgt]
        src_direction = torch.LongTensor([single_src_direction for _ in range(len(src_dataset))])
        src_direction = RawLabelDataset(src_direction)
        tgt_direction = torch.LongTensor([single_tgt_direction for _ in range(len(tgt_dataset))])
        tgt_direction = RawLabelDataset(tgt_direction)

        return LanguagePairDatasetZero(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset.sizes if tgt_dataset is not None else None,
            tgt_dict,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )