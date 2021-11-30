# mZPRT

Multi-domain Zero Pronoun Recovery and Translation Dataset, MT data for Movie Subtitle

### Detail

| Data      | Documents | Sentences(w/o doc boundary) |
| --------- | --------- | --------------------------- |
| Processed | 14946     | 12885410                    |
| Raw       | 14946     | 13366889                    |

1. '[doc]' is the document boundary
2. For the processed data set, we remove the sentence pairs that are English on both sides.
3.  The processed dataset: mingzhouxu@100.77.35.216:/home/mingzhouxu/data/open_sub/static/processd
4. The raw dataset: mingzhouxu@100.77.35.216:/home/mingzhouxu/data/open_sub/static/raw

---

### How to use

1. For sent-level MT, the doc boundary can be removed easly, for example(linux):   
   ```bash
   sed '/\[doc\]/d' train.bpe.zh >train.sent.bpe.zh
   ```

2.  For doc-level MT, the context file could be generate by our scripts:   
      ```python
      # make context data for movie subtitle
      python ../scrips/make_doc_context.py -i train.bpe.zh -o train.ctx.zh -n 1
      # -n is the num_of_ctxs
      ```   
   
      2.1. The number of sentences in the output file should be the same as the sent-level file
   
3. 

