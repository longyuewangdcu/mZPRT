# mZPRT

Multi-domain Zero Pronoun Recovery and Translation Dataset

### Detail

| Dataset   | Sentences |
| --------- | --------- |
| Processed | 42817898  |
| Raw       | 42817898  |

1. Processd dataset:: mingzhouxu@100.77.35.216:/home/mingzhouxu/data/WMT_ZH/paralle/wmt2021/alignment/BPE
   1. en.code  adn zh.code are the BPE code file
2. Raw dataset:mingzhouxu@100.77.35.216:/home/mingzhouxu/data/WMT_ZH/paralle/wmt2021/raw/raw
3. We use newstest2019 as the validation set.



---

### How to use

1. For doc-level MT, the context file could be generate by our scripts:

```python
# make context data for movie subtitle
python ../scrips/make_doc_context.py -i train.bpe.zh -o train.ctx.zh -n 1 -w True
```

​	The number of sentences in ctx file should be the same as the  input file

