### mZPRT

---

Multi-domain Zero Pronoun Recovery and Translation Dataset



#### Catalogue

.

├──scripts

├──raw                                                         # Our Benchamrk testset without any processing

├──processed                                              # Our processed testset(tokenized, BPE)

│    ├──context-agnostic                            # Processed testset for sent-level MT

│    │    ├──original                                      # Testset that omitted ZPs

│    │    ├──oracle                                        # Testset contains ZPs that recovered by human

│    ├──context-aware                                # Processed testset for Doc-level MT, hear we only provide previous sentences (only one)

│    │    ├──original                                      # Same as the context-agnostic version   

│    │    ├──oracle                                        # Same as the context-agnostic version   

└── README.md



Note that: the processed data are tokenized and contain BPE tag.  In details:

* English dataset are tokenized by tokenizer.pl from Moses;
  * sh ./scripts/preprocess_en.sh dir_path prefix_of_file
* Chinese dataset are tokenized by jieba toolkit with a userdict.txt (ZPs);
  * sh ./scripts/preprocess_zh.sh dir_path prefix_of_file
* Same as Chinese-Bert, we lowercase the English word in Chinese dataset



---

#### How to use:

* Raw dataset need to tokenized first than

* Processed data can be used directly

  * About Context-Aware dataset, we only provide a previous sentence as context.
  * Here,  take 'Baidu_knows.ctx.zh' in Context-Aware dir as an instance, this is the context file of 'Baidu_knows.bpe.zh' in Context-agnositc dir. 

  





