#!/usr/bin/bash

set -x
data_dir=/data/open_sub/SAN
save_dir=/models/open_sub/cache
log_dir=/log/open_sub/cache
user_dir=./
SPM=/models/open_sub/SAN/checkpoint_best.pt
mkdir ${log_dir}
mkdir ${output}
output=${output}/opensub.big.batch.cache
source ~/.bashrc
conda activate base
# cosine max-lr ==> lr 1.0 updated
fairseq-train $data_dir \
    --user-dir ${user_dir} \
	--save-dir $save_dir \
	-s zh -t en \
	--task cachetranslation \
	--arch cachemodell_vaswani_wmt_en_de_big \
    --cache 'Tu' --init-tokens 35628 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
	--lr-scheduler cosine --lr 1e-07 --max-lr 1e-4 --min-lr 1e-9 \
    --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 50000 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens 4000 --update-freq 12 \
    --fp16 --fp16-scale-tolerance 0.3 \
	--encoder-normalize-before  --decoder-normalize-before  \
	--dropout 0.3 --attention-dropout 0.1 --weight-decay 0.00001 \
	--max-update 60000 \
	--keep-interval-updates 50 --save-interval-updates 1000  --log-interval 100 --no-epoch-checkpoints \
    --log-format simple \
	--pretrained-checkpoint $SPM \
    --eval-bleu  --eval-bleu-args '{"beam":4,"lenpen":1.2}' --eval-bleu-remove-bpe \
    --validate-interval-updates 5000 \
    --tensorboard-logdir ${log_dir} \
	--ddp-backend=no_c10d |tee ${log_dir}/opensub-big-batch-cache-doc-fp16.log








