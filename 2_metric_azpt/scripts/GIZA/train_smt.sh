Moses=/home/user/mosesdecoder
Trainer=scripts/training/train-model.perl 
f=zh
e=en
Root=/home/user/SMT
Corpus=/home/user/data/corpus/train.tok


nohup $Moses/$Trainer -root-dir $Root/train \
  -corpus $Corpus \
  -f $f -e $e \
  -alignment intersect -reordering msd-bidirectional-fe \
  --cores 80 \
  --first-step 1 \
  --last-step 1 \
  -external-bin-dir ~/tools_GIZA++_mkcls_dir >& training.out &
