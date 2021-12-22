#!/usr/bin/bash

set -x
data_dir=/apdcephfs/share_916081/mingzhouxu/data/WMT_ZhEn/
save_dir=/apdcephfs/share_916081/mingzhouxu/models/WMT_ZhEn_Rec_zpt_eval
log_dir=/apdcephfs/share_916081/mingzhouxu/log/WMT_ZhEn_Rec_zpt_eval
user_dir=./
#pip install fairseq==0.7.1
#SPM=/apdcephfs/share_916081/mingzhouxu/models/WMT_ZhEn_Rec_tune_eval/checkpoint_best.pt
mkdir ${log_dir}
#source /apdcephfs/private_mingzhouxu/miniconda/etc/profile.d/conda.sh
#conda activate old
source ~/.bashrc
conda activate base

#pip uninstall fairseq -y

#pip install --editable /apdcephfs/share_916081/mingzhouxu/program/fairseq/ 

src=zh
tgt=en
ARCH=transformer_vaswani_wmt_en_de_big

fairseq-train ${data_dir} \
--arch $ARCH -s zh -t en \
--save-dir ${save_dir} \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler cosine --lr 1e-03 --min-lr 1e-9 \
--warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 50000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 4000 --update-freq 16 \
--fp16 --fp16-scale-tolerance 0.3 \
--encoder-normalize-before  --decoder-normalize-before \
--dropout 0.3 --attention-dropout 0.1 --weight-decay 0.00001 \
--max-update 40000 \
--keep-interval-updates 50 --save-interval-updates 1000  --log-interval 100  \
--log-format simple \
--eval-bleu  --eval-bleu-args '{"beam":5,"lenpen":1.2}' --eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--patience 5 \
--validate-interval-updates 5000 \
--tensorboard-logdir ${log_dir} \
--ddp-backend=no_c10d | tee ${log_dir}/wmt-rec-fp16-zpt.log
#--pretrained-checkpoint $SPM --reset-lr-scheduler --reset-optimizer \
