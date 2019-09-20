
export CUDA_VISIBLE_DEVICES=$1
EXPERIMENT=eli_seq2seq_qd

python train.py data-bin/eli5 --task translation --source-lang qd_source_bpe --target-lang qd_target_bpe --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed --dropout 1e-1 --attention-dropout 1e-1 --relu-dropout 1e-1 --criterion label_smoothed_cross_entropy --label-smoothing 1e-1 --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 --min-lr 1e-9 --clip-norm 0 --no-progress-bar --log-interval 100 --max-tokens 10000 --skip-invalid-size-inputs-valid-test --fp16 --tensorboard-logdir ./logs/$EXPERIMENT --no-epoch-checkpoints \
	--max-source-positions 4096 \
       --max-target-positions 4096 \
       --max-sentences 1 \
       --save-dir ./checkpoints/$EXPERIMENT        

