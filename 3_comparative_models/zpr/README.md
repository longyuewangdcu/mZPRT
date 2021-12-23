# mZPRT

Multi-domain Zero Pronoun Recovery and Translation Dataset

    .
    ├── zpr                    
    │   ├── ZPR                 # code for ZPR task
    │   ├── data                # Training data
    │   ├── models              # ZPR checkpoints
    │   ├── scripts             # ZPR scipt
    └── README.md



### Training MT Baseline

ZPR model based on ``[ZPR2: Joint Zero Pronoun Recovery and Resolution using Multi-Task Learning and BERT](https://aclanthology.org/2020.acl-main.482/)''. We didn't use the multi-task version. We only run the recovery task. This method is based on Transformers which provides by hugging face. The BERT model we used is bert-base-chinese.

1. Dataset:
    1.1 The training data are put in data dir, QA_Forum is the dataset for QA domain, Movie_Subtitle is for Movie Subtitle and General_Domain are for others.
    
    |Dataset|General_Domain|Movie_Subtitle|QA_Forum|
    |--|--|--|--|
    |Train|30000|2150759|5504|
    |Dev|1175|1049|1175|
    
    1.2 testset are putted on 1_testset_mzprt

2. Training:
    2.1 Trainging code and script are release on ZPR dir.
   ```bash
        # config_recover*.json are the training configure, e.g
        sh run.sh config_recover.json
   ```
   
4. Generation:
    4.1 Turn the testset into input format(json)
    ```bash
       sh ./script/turn_oracle_2_zpr.sh testset
       # finally, there is a ZP_in.json, this is the input format
    ```
    4.2 Predicting:
    ```bash
    python predict_recovery.py --prefix_path ./logs_base_BK_base/ZP.recovery_bertchar --in_path data/BK/ZP_in.json --out_path output_file
    ```
    4.3 Feed into MT system:
    ```
    # remove the ``<<'',``>>'' with regular or using the make_mt_data.py scripts to get the tokenized data
    # then apply-bpe, we use the fastbpe
    fast applybpe test.zh-en.zh ZPR_output zh.code(in mt data)
    # feed into MT model, turn --dataset-impl to raw if you don't want to make data-bin
    fairseq-generate . --path checkpoint_big.pt -s zh -t en --lenpen 1.2 --beam 5 --remove-bpe --dataset-impl raw > output
    ```


   

