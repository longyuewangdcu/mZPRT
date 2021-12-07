# mZPRT
Multi-domain Zero Pronoun Recovery and Translation Dataset

## Catalogue

```
.

├──scripts
├──raw                                              # Our Benchamrk testset without any processing
├──processed                                        # Our processed testset(tokenized, BPE)
│    ├──context-agnostic                            # Processed testset for sent-level MT
│    │    ├──original                               # Testset that omitted ZPs
│    │    ├──zp-labeled                                 # Testset contains ZPs that recovered by human
│    ├──context-aware                               # Processed testset for Doc-level MT, hear we only provide previous sentences (only one)
│    │    ├──original                               # Same as the context-agnostic version   
│    │    ├──zp-labeled                                 # Same as the context-agnostic version   
└── README.md
```


Note that: the processed data are tokenized and contain BPE tag.  In details:

* English dataset are tokenized by tokenizer.pl from Moses;
  
  ````
  sh ./scripts/preprocess_en.sh dir_path prefix_of_file
  ````
  
* Chinese dataset are tokenized by jieba toolkit with a userdict.txt (ZPs);
  
  ```
  sh ./scripts/preprocess_zh.sh dir_path prefix_of_file
  ```
  
* Same as Chinese-Bert, we lowercase the English word in Chinese dataset

* For the BPE, we use fastbpe

---
## How to use:

* Processed data can be used directly
  * About Context-Aware dataset, we only provide a previous sentence as context.
  * Here,  take 'Baidu_knows.ctx.zh' in Context-Aware dir as an instance, it is the context file of 'Baidu_knows.bpe.zh' in Context-agnositc dir. 

* About the Validation set:
  * We use the `Movie_Subtitle_valid' to select the best model for the Movie_Subtitle domain.
  * Other domains, We just using the 'official validation set' from WMT translation task.
  * Therefore, we only provide the processed validation set of Movie_Subtitle.

* From Raw to Original or Oracle, take Movie_Subtitle_test.zh as example:

  ```bash
  python ./scripts/make_mt_data.py -i Movie_Subtitle_test.zh -o M.S.original.zh
  python ./scripts/make_mt_data.py -i Movie_Subtitle_test.zh -o M.S.oracle.zh -z
  sh ./scripts/preprocess_zh.sh path_of_M.S.original.zh M.S.original
  sh ./scripts/preprocess_zh.sh path_of_M.S.oracle.zh M.S.oracle
  ```

  

  



