save_dir=/apdcephfs/share_916081/mingzhouxu/models/open_sub/rec_tuned_zpt


ls ${save_dir}/*.pt |while read file;do echo $file ;fairseq-generate /apdcephfs/share_916081/mingzhouxu/data/mZPRT-main/data_mzprt/processed/context/original/MS_ctx/ --path $file -s zh -t en --lenpen 1.2 --remove-bpe --beam 5 --quiet >> /apdcephfs/share_916081/mingzhouxu/data/mZPRT-main/data_mzprt/processed/context/original/MS_ctx/rec_eval_log ;done  
