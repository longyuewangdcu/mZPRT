# mZPRT
Multi-domain Zero Pronoun Recovery and Translation Dataset


    .
    ├── aZPT                    # aZPT score
    ├── aZPT_output             # aZPT_output in details
    ├── human_score             # human evaluation results
    ├── sample                  # runing sample for aZPT
    ├── scripts                 # scripts for training and applying aligment 
    └── README.md

---
### Usage

aZPT score:
```bash
   python aZPT/aZPT.py -h
   usage: aZPT.py [-h] [-s SOURCE] [-t TARGET] [-at TGT_ALIGN] [-n NUM_NEIGHBORS] [-o OUTPUT]

    optional arguments:
    -h, --help                          #show this help message and exit
    -s, --source SOURCE                 #source input of MT
    -t, --target TARGET                 #hypothesis of MT
    -at, --tgt-align TGT_ALIGN          #Alignment between source and hypothesis
    -n, --num-neighbors NUM_NEIGHBORS   #Number of neighbors
    -o, --output OUTPUT
```
