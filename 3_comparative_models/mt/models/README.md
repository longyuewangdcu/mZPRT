# mZPRT

MT models

    .
    ├── QA_Forum         # Tuned models for QA_Forum
    ├── Web_Fict         # Tuned models for Web_Fict
    ├── Mov_Sub          # Models for Mov_Sub
    ├── Others           # Models for other two without domain-shift
    └── README.md



## How to use

Task QA_Forum as an example:

```bash
data_bin=~/mZPRT/1_testset_mzprt/processed/context-agnostic/QA_data_bin/
dict_path=~/mZPRT/3_comparative_models/mt/data/WMT2021/

fairseq-preprocess -s zh -t en --trainpref train.bpe  --validpref valid.bpe --destdir ${data_bin} --workers 20 --srcdict ${dict_path}/dict.zh.txt --tgtdict ${dict_path}/dict.en.txt

fairseq-generate  ${data_bin} --path QA_Forum/Big_best.pt -s zh -t en --lenpen 1.2 --remove-bpe --beam 5 > output
sh ~/mZPRT/scripts/get_hyp.sh output  #  extract hypothesis
# output.hyp is the generated translations
perl ~/mZPRT/scripts/get_hyp.sh reference < output.hyp
# reference is test_set english side without BPE
```

Note that:

1. QA_Forum, Web_Fict and Others are using the same dictionary files in WMT2021

   

