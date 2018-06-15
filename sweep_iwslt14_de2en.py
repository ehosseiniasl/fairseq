#!/usr/bin/env python

import sweep
from sweep import hyperparam


def get_grid(args):
    return [
        #hyperparam('--criterion', 'label_smoothed_cross_entropy'),
        hyperparam('--task', 'diverse_translation'),
        hyperparam('--criterion', 'latent_var'),
        hyperparam('--latent-category', 2, save_dir_key=lambda val: f'latent{val}'),
        hyperparam('--latent-impl', 'tgt', save_dir_key=lambda val: f'{val}'),

        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),

        #hyperparam('--max-epoch', 200),
        hyperparam('--max-update', 20000),      #with 4 or 8 gpus

        hyperparam('--arch', 'transformer_iwslt_de_en', save_dir_key=lambda val: val.split('_')[0]),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.98)', save_dir_key=lambda val: 'beta0.9-0.98'),
        hyperparam('--lr', 0.0001, save_dir_key=lambda val: f'lr{val}'),
        #hyperparam('--min-lr', 1e-9),
        hyperparam('--lr-scheduler', 'fixed'),
        #hyperparam('--lr-scheduler', 'inverse_sqrt'),
        #hyperparam('--weight-decay', 0.0001, save_dir_key=lambda val: f'weightdecay{val}'),
        #hyperparam('--force-anneal', 200, save_dir_key=lambda val: f'fa{val}'),
        #hyperparam('--warmup-updates', 4000, save_dir_key=lambda val: f'warmup{val}'),
        #hyperparam('--warmup-init-lr', 1e-7, save_dir_key=lambda val: f'initlr{val}'),

        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),
        hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--label-smoothing', 0.0, save_dir_key=lambda val: f'ls{val}'),
        #hyperparam('--share-all-embeddings', save_dir_key=lambda val: 'shareemb'),

        hyperparam('--max-tokens', 4000, save_dir_key=lambda val: f'maxtok{val}'),
        #hyperparam('--max-sentences', 32, save_dir_key=lambda val: f'maxsent{val}'),

        hyperparam('--seed', 1, save_dir_key=lambda val: f'seed{val}'),

        hyperparam('--log-format', 'simple'),       # latent_count is not JSON serializable
        hyperparam('--log-interval', 100),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    #if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    #else:
    #    config['--max-tokens'].current_value = 300
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
