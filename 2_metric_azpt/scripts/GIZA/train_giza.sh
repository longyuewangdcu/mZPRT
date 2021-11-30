GIZA=~/giza-pp/GIZA++-v2
MKCLS=~/giza-pp/mkcls-v2/mkcls
ROOT=$1
f=zh
e=en
trainpref=$2
#first step
$MKCLS -c50 -n2 -p$trainpref.$f -V$ROOT/corpus/$f.vcb.classes opt
$MKCLS -c50 -n2 -p$trainpref.$e -V$ROOT/corpus/$e.vcb.classes opt
GIZA/palin2snt.out $trainpref.$f $trainpref.$e
mv $trainpref.$f-$trainpref.$e $ROOT/corpus/${f}-${e}-int-train.snt
mv $trainpref.$e-$trainpref.$f $ROOT/corpus/${e}-${f}-int-train.snt
mv $trainpref.$f.vcb $ROOT/corpus/$f.vcb
mv $trainpref.$e.vcb $ROOT/corpus/$e.vcb

#python ./make_vocab.py -i $trainpref.$f -o $ROOT/corpus/$f.vcb
#python ./make_vocab.py -i $trainpref.$e -o $ROOT/corpus/$e.vcb
#second step
${GIZA}/snt2cooc.out $ROOT/corpus/$f.vcb $ROOT/corpus/$e.vcb $ROOT/corpus/${f}-${e}-int-train.snt >$ROOT/giza.${f}-${e}/${f}-${e}.cooc
${GIZA}/snt2cooc.out  $ROOT/corpus/$e.vcb $ROOT/corpus/$f.vcb $ROOT/corpus/${e}-${f}-int-train.snt > $ROOT/giza.${e}-${f}/${e}-${f}.cooc

$GIZA/GIZA++ -CoocurrenceFile $ROOT/giza.${f}-${e}/${f}-${e}.cooc c $ROOT/corpus/${f}-${e}-int-train.snt -m1 5 -m2 0 -m3 3 -m4 3 -model1dumpfrequency 1 -model4smoothfactor 0.4 -nodumps 0 -nsmooth 4 -o $ROOT/giza.${f}-${e}/${f}-${e} -onlyaldumps 0 -p0 0.999 -s $ROOT/corpus/${e}.vcb -t $ROOT/corpus/${f}.vcb > giza-${f}${e}.log &

$GIZA/GIZA++ -CoocurrenceFile $ROOT/giza.${e}-${f}/${e}-${f}.cooc c $ROOT/corpus/${e}-${f}-int-train.snt -m1 5 -m2 0 -m3 3 -m4 3 -model1dumpfrequency 1 -model4smoothfactor 0.4 -nodumps 0 -nsmooth 4 -o $ROOT/giza.${e}-${f}/${e}-${f} -onlyaldumps 0 -p0 0.999 -s $ROOT/corpus/${f}.vcb -t $ROOT/corpus/${e}.vcb > giza-${e}${f}.log &


