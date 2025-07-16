from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from ZeroTrans.tasks import MultilingualDatasetManagerZero


@register_task("translation_multi_simple_epoch_zero")
class TranslationMultiSimpleEpochTaskZero(TranslationMultiSimpleEpochTask):

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.data_manager = MultilingualDatasetManagerZero.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )