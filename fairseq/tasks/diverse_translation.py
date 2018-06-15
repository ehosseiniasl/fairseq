# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import torch

from fairseq import utils, options
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset,
)

from . import register_task
from .translation import TranslationTask


@register_task('diverse_translation')
class DiverseTranslationTask(TranslationTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        super(DiverseTranslationTask, DiverseTranslationTask).add_args(parser)
        parser.add_argument('--latent-category', default=2, type=int, metavar='N',
                            help='number of categories of the latent variable')
        parser.add_argument('--latent-impl', default='tgt', choices=['src', 'tgt'],
                            help='how to add the latent variable')

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data)
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        # add special tokens for latent categories
        for k in range(args.latent_category):
            src_dict.add_symbol('<latent_%d>' % k)
            tgt_dict.add_symbol('<latent_%d>' % k)

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def add_latent_src(self, net_input, latent):
        latent += self.src_dict.index('<latent_0>')
        return {
            'src_tokens': torch.cat((net_input['src_tokens'], latent.unsqueeze(-1)), 1),
            'src_lengths': net_input['src_lengths'] + 1,
            'prev_output_tokens': net_input['prev_output_tokens'],
        }

    def add_latent_tgt(self, net_input, latent):
        latent += self.tgt_dict.index('<latent_0>')
        net_input['prev_output_tokens'][:, 0] = latent
        return net_input

    def add_latent_variable(self, net_input, latent):
        if self.args.latent_impl == 'src':
            return self.add_latent_src(net_input, latent)
        return self.add_latent_tgt(net_input, latent)

    def add_latent_variables(self, net_input, target, latents=None):
        if latents is None:
            latents = torch.arange(0, self.args.latent_category)
        latents = latents.type_as(net_input['src_tokens'])
        bsz = net_input['src_tokens'].size(0)
        latent = latents.repeat(bsz, 1).t().contiguous().view(-1)

        k = latents.numel()
        repeated_net_input = {
            'src_tokens': net_input['src_tokens'].repeat(k, 1),
            'src_lengths': net_input['src_lengths'].repeat(k),
            'prev_output_tokens': net_input['prev_output_tokens'].repeat(k, 1),
        }
        return self.add_latent_variable(repeated_net_input, latent), \
            target.repeat(k, 1) if target is not None else None
