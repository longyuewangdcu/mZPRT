# mZPRT

Multi-domain Zero Pronoun Recovery and Translation Dataset

    .
    ├── 1_testset_mzprt         # Our benchmark testset
    ├── 2_metric_azpt           # Our evaluation metric
    │   ├── script              # evaluation toolkit
    │   ├── human_score         # human judgements on 6 systems
    ├── 3_comparative_models    # Bechmark related resources
    │   ├── mt                  # Machine translation task
    │   │   ├── data
    │   │   ├── code
    │   │   ├── model
    │   ├── zpr                 # Zero pronoun recovery task
    │   │   ├── data
    │   │   ├── code
    │   │   ├── model
    │   └── zpt                 # Zero pronoun translation task
    │   │   ├── data
    │   │   ├── code
    │   │   ├── model
    └── README.md



### Training MT Baseline

All our models are trained on 8xV100 32Gb, fairseq toolkits.

1. fairseq version is 0.10.2

   ```
   pip install fairseq 
   ```

2. Preparing data bin files:

   ```bash
   #!/usr/bin/bash
   
   set -x
   data_dir=data_bin
   save_dir=where_to_save
   log_dir=where_to_log
   #preprocessing the data
   fairseq-preprocess -s zh -t en --trainpref training_set  --validpref valid_set --testpref test_set --destdir ${data_dir} --workers 20 
   ```

3. Training parameters for Movie Subtitle, Deep model

   ```bash
   #training sentence-level MT model
   # Deep model is shown here
   
   mkdir ${log_dir}
   # cosine max-lr ==> lr fairseq-1.0 updated
   fairseq-train $data_dir \
   	--save-dir $save_dir \
   	-s zh -t en \
   	--arch transformer_vaswani_wmt_en_de_big \
    --encoder-layers 12 --decoder-layers 12 \ # Big model both are 6
   	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
   	--lr-scheduler cosine --max-lr 1e-3 --min-lr 1e-9 \
    --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 50000 \
   	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
   	--max-tokens 4800 --update-freq 12 \
    --fp16 --fp16-scale-tolerance 0.3 \
   	--encoder-normalize-before  --decoder-normalize-before  \
   	--dropout 0.3 --attention-dropout 0.1 --weight-decay 0.00001 \
   	--max-update 60000 \
   	--keep-interval-updates 50 --save-interval-updates 1000  --log-interval 100 --no-epoch-checkpoints \
    --log-format simple \
    --eval-bleu  --eval-bleu-args '{"beam":5,"lenpen":1.2}' --eval-bleu-remove-bpe \
    --validate-interval-updates 5000 \
    --tensorboard-logdir ${log_dir} \
   	--ddp-backend=no_c10d |tee ${log_dir}/log_file
   ```
   
   Note that:
   
   1.  Our models are trained on 8 x V100 32Gb, therefore the batch size is 4800x12x8 approximate 460K tokens/batch
   2.  This the parameter for **Deep** model, **Big** model just remove the '--encoder-layers 12 --decoder-layers 12' or set them to 6.
   3.  For **base** model, set '--arch transformer', others are same as **Big** model
   
4. Generation:

   ```bash
   fairseq-generate ${data_dir} --path path/checkpoint_best.pt -s zh -t en --lenpen 1.2 --remove-bpe --beam 5 > output
   sh ~/mZPRT/scripts/get_hyp.sh output  #  extract hypothesis
   # output.hyp is the generated translations
   perl ~/mZPRT/scripts/get_hyp.sh reference < output.hyp
   # reference is test_set english side without BPE
   ```

   

