# from Myle Ott
python train.py data-bin/iwslt14.tokenized.de-en --max-update 50000 --arch transformer_iwslt_de_en --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --lr 0.0005 --min-lr '1e-09' --clip-norm 0.0 --dropout 0.3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 --seed 1 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-layers 6 --decoder-embed-dim 512 --decoder-ffn-embed-dim 1024 --decoder-layers 6 --no-progress-bar --log-interval 100

# from Michael Auli
python train.py data-bin/iwslt14.tokenized.de-en --source-lang de --target-lang en \
  --save-dir checkpoints/auli \
  --max-update 50000 --arch transformer_iwslt_de_en \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 4 --encoder-layers 6 \
  --decoder-embed-dim 512 --decoder-ffn-embed-dim 1024 --decoder-attention-heads 4 --decoder-layers 6 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.0005 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0 --dropout 0.3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4000 --no-progress-bar --log-interval 100 --seed 1

# run on cluster
sbatch --job-name baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1 \
  --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --mem-per-cpu 6G \
  --output /checkpoint02/tianxiao/2018-06-17/baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1.ngpu1/train.log \
  --error /checkpoint02/tianxiao/2018-06-17/baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1.ngpu1/train.stderr.%j \
  --open-mode append --partition uninterrupted --requeue --wrap \
  'srun --job-name baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1 \
  --output /checkpoint02/tianxiao/2018-06-17/baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1.ngpu1/train.log \
  --error /checkpoint02/tianxiao/2018-06-17/baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1.ngpu1/train.stderr.%j \
  --open-mode append --unbuffered \
  python train.py data-bin/iwslt14.tokenized.de-en/ \
  --save-dir /checkpoint02/tianxiao/2018-06-17/baseline.fp16.transformer.adam.beta0.9-0.98.lr0.0005.weightdecay0.0001.warmup4000.initlr1e-07.clip0.0.drop0.3.ls0.1.maxtok4000.seed1.ngpu1 \
  --fp16 --max-update 50000 --arch transformer_iwslt_de_en \
  --optimizer adam --adam-betas '"'"'(0.9, 0.98)'"'"' --lr 0.0005 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0.0 --dropout 0.3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4000 --seed 1 --log-format json --log-interval 100'

# test
python generate.py data-bin/iwslt14.tokenized.de-en \
  --task diverse_translation --path $CKPT/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe | tee $CKPT/gen.out



# latent variable
python train.py data-bin/iwslt14.tokenized.de-en --save-dir checkpoints/latent_var \
  --task diverse_translation --criterion latent_var --latent-category 2\
  --max-update 50000 --arch transformer_iwslt_de_en \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.0005 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0 --dropout 0.3 --label-smoothing 0.1 \
  --max-tokens 4000 --no-progress-bar --log-interval 100 --seed 1


# wmt14 en-de
python train.py data-bin/wmt14_en_de_joined_dict \
  --task diverse_translation --criterion latent_var --latent-category 2\
  --max-epoch 50 --arch transformer_vaswani_wmt_en_de_big \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler fixed --lr 0.0001 \
  --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --label-smoothing 0.0 \
  --max-tokens 4000 --no-progress-bar --log-interval 10 --seed 2

python train.py data-bin/wmt14_en_de_joined_dict --save-dir checkpoints/latent_src --task diverse_translation --criterion latent_var --latent-category 10 --latent-impl src --fp16 --max-epoch 17 --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --label-smoothing 0.0 --max-tokens 3000 --seed 2 --log-format simple --log-interval 10
