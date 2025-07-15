#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import os
import sys
import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.generate')

def main(args):
    utils.import_user_module(args)

    try:
        # Fix seed for stochastic decoding
        if args.seed is not None and not args.no_seed_provided:
            utils.set_torch_seed(args.seed)

        # 设置任务
        task = tasks.setup_task(args)

        # 加载模型
        logger.info('loading model(s) from {}'.format(args.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(args.path),
            arg_overrides=ast.literal_eval(args.model_overrides),
            task=task,
            suffix=getattr(args, "checkpoint_suffix", ""),
            strict=(args.checkpoint_shard_count == 1),
            num_shards=args.checkpoint_shard_count,
        )

        # 优化模型以便于生成
        for model in models:
            if model is None:
                continue
            if args.fp16:
                model.half()
            if args.cpu:
                model.cpu()
            if not args.no_beamable_mm:
                model.make_generation_fast_()

        # 加载数据集
        logger.info('loading dataset for epoch {}'.format(args.num_workers))
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(), *[model.max_positions() for model in models]
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
        )

        # 初始化生成器
        gen_timer = progress_bar.StopwatchMeter()
        generator = task.build_generator(models, args)

        # 生成并输出
        num_sentences = 0
        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = progress_bar.TimeMeter()
            for sample in t:
                sample = utils.move_to_cuda(sample) if args.cuda else sample
                if 'net_input' not in sample:
                    continue

                gen_timer.start()
                hypos = task.inference_step(generator, models, sample)
                num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
                gen_timer.stop(num_generated_tokens)

                for i, sample_id in enumerate(sample['id'].tolist()):
                    has_target = sample['target'] is not None

                    # 移除填充符
                    src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], task.source_dictionary.pad())
                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(sample['target'][i, :], task.target_dictionary.pad()).int().cpu()

                    # 处理源文本
                    src_str = task.source_dictionary.string(src_tokens, args.remove_bpe)
                    print('S-{}\t{}'.format(sample_id, src_str))

                    # 处理目标文本
                    if has_target:
                        tgt_str = task.target_dictionary.string(target_tokens, args.remove_bpe, escape_unk=True)
                        print('T-{}\t{}'.format(sample_id, tgt_str))

                    # 处理假设
                    if not args.quiet:
                        for j, hypo in enumerate(hypos[i][:args.nbest]):
                            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                                hypo_tokens=hypo['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=hypo['alignment'],
                                align_dict=None,
                                tgt_dict=task.target_dictionary,
                                remove_bpe=args.remove_bpe,
                                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                            )
                            print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                            print('P-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                            ))
                            if args.print_alignment:
                                print('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                                ))

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += sample['nsentences']

        logger.info('生成了 {} 个句子，平均每个句子 {:.1f} 秒，{:.2f} 句/秒，{:.2f} 词/秒'.format(
            num_sentences, gen_timer.sum / num_sentences,
            1. / gen_timer.avg, 1. / gen_timer.avg
        ))
    except:
        raise

def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main() 