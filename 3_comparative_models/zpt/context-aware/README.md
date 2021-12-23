## Doc-level MT 

### Models
1. ZhangModel, a model proposed by Zhang et al. EMNLP 2018
2. Cache, a model proposed by Tu et al. TACL 2018
3. Uinifed, model proposed by Ma et al. ACL 2020

### Usage:

#### Preparation
1. Data for Zhang model and unified model (the data of cache model is same with baseline). 
```bach
1. python ../scripts/make_doc_context.py -i [inputs] -o [outputs] -n [num_of_ctx] -w [making noise data for wmt]

2. fairseq-preprocess -s zh -t en --trainpref train_ctx --validpref valid_ctx --testpref test_ctx --only-source --workers 20 --destdir TMP
# rename and move to the data dir for training baseline system. e.g.
3. mv TMP/train.zh-en.zh.bin SAN/train.ctx.zh.bin
# finally, we give a data-bin like this:
    .
    ├── train.zh-en.en.bin   
    ├── train.zh-en.en.idx   
    ├── train.zh-en.zh.bin   
    ├── train.zh-en.zh.idx        
    ├── train.ctx.zh.bin   
    ├── train.ctx.zh.idx 
    ├── valid.zh-en.en.bin   
    ├── valid.zh-en.en.idx    
    ├── valid.zh-en.zh.bin   
    ├── valid.zh-en.zh.idx 
    ├── valid.ctx.zh.bin   
    ├── valid.ctx.zh.idx 
    └── ...
```
The context-aware models are trained on two-stage training process. Therefore, we should training a conventional Transformer first.
1. Plz refer to [MT](../../mt/README.md) to train the baseline
2. --pretrained-checkpoint (shell script is the SPM=) baseline_checkpoint.pt # adding this to load the parameters from sent-level checkpoint

#### Training model
```bash
# run cache model
sh start-cache.sh
# run Zhang model
sh start-zhang.sh
# run Unified model
sh start-unified.sh
```

Note that: Generation is same as sent-level baseline.