#!/usr/bin/env bash
set -x 
GIZA=/home/whozhwang/giza-pp/GIZA++-v2
MGIZA=/home/whozhwang/mgiza/mgizapp/inst
if [ $# -lt 4 ]; then
	echo "OK, this is simple, put me into your Moses training directory, link your source/target corpus" 1>&2
	echo "and run " $0 " PREFIX src_tag tgt_tag root-dir." 1>&2
	echo "and get force-aligned data: root-dir/giza.[src-tgt|tgt-src]/*.A3.final.* " 1>&2
	echo "make sure I can find PREFIX.src_tag-tgt_tag and PREFIX.tgt_tag-src_tag, and \${QMT_HOME} is set" 1>&2
	exit
fi

PRE=$1
SRC=$2
TGT=$3
ROOT=$4
NUM=$5

src=zh
tgt=en

Pre_SRC_VCB=/home/whozhwang/xmz/SMT/train/corpus/zh.vcb
Pre_TGT_VCB=/home/whozhwang/xmz/SMT/train/corpus/en.vcb

mkdir -p $ROOT/giza-inverse.${NUM}
mkdir -p $ROOT/giza.${NUM}
mkdir -p $ROOT/prepared.${NUM}

echo "Generating corpus file " 1>&2

python2 ${MGIZA}/plain2snt-hasvcb.py ${Pre_SRC_VCB} ${Pre_TGT_VCB} ${SRC} ${TGT} $ROOT/prepared.${NUM}/${tgt}-${src}.snt $ROOT/prepared.${NUM}/${src}-${tgt}.snt $ROOT/prepared.${NUM}/$src.vcb $ROOT/prepared.${NUM}/$tgt.vcb

ln -sf ${Pre_SRC_VCB}.classes ${Pre_TGT_VCB}.classes $ROOT/prepared.${NUM}/

echo "Generating co-occurrence file " 1>&2

${GIZA}/snt2cooc.out $ROOT/prepared.${NUM}/$src.vcb $ROOT/prepared.${NUM}/$tgt.vcb $ROOT/prepared.${NUM}/${tgt}-${src}.snt >$ROOT/giza.${NUM}/${tgt}-${src}.cooc
${GIZA}/snt2cooc.out  $ROOT/prepared.${NUM}/$tgt.vcb $ROOT/prepared.${NUM}/$src.vcb $ROOT/prepared.${NUM}/${src}-${tgt}.snt > $ROOT/giza-inverse.${NUM}/$src-${tgt}.cooc

echo "Running force alignment " 1>&2

${GIZA}/GIZA++ ${PRE}/giza.$tgt-$src/$tgt-$src.gizacfg -c $ROOT/prepared.${NUM}/$tgt-$src.snt -o $ROOT/giza.${NUM}/$tgt-${src} \
-s $ROOT/prepared.${NUM}/$src.vcb -t $ROOT/prepared.${NUM}/$tgt.vcb -m1 0 -m2 0 -mh 0 -coocurrence $ROOT/giza.${NUM}/$tgt-${src}.cooc \
-restart 11 -previoust ${PRE}/giza.$tgt-$src/$tgt-$src.t3.final \
-previousa ${PRE}/giza.$tgt-$src/$tgt-$src.a3.final -previousd ${PRE}/giza.$tgt-$src/$tgt-$src.d3.final \
-previousn ${PRE}/giza.$tgt-$src/$tgt-$src.n3.final -previousd4 giza.$tgt-$src/$tgt-$src.d4.final \
-previousd42 ${PRE}/giza.$tgt-$src/$tgt-$src.D4.final -m3 0 -m4 1

${GIZA}/GIZA++ ${PRE}/giza.$src-$tgt/$src-$tgt.gizacfg -c $ROOT/prepared.${NUM}/$src-$tgt.snt -o $ROOT/giza-inverse.${NUM}/$src-${tgt} \
-s $ROOT/prepared.${NUM}/$tgt.vcb -t $ROOT/prepared.${NUM}/$src.vcb -m1 0 -m2 0 -mh 0 -coocurrence $ROOT/giza-inverse.${NUM}/$src-${tgt}.cooc \
-restart 11 -previoust ${PRE}/giza.$src-$tgt/$src-$tgt.t3.final \
-previousa ${PRE}/giza.$src-$tgt/$src-$tgt.a3.final -previousd ${PRE}/giza.$src-$tgt/$src-$tgt.d3.final \
-previousn ${PRE}/giza.$src-$tgt/$src-$tgt.n3.final -previousd4 ${PRE}/giza.$src-$tgt/$src-$tgt.d4.final \
-previousd42 ${PRE}/giza.$src-$tgt/$src-$tgt.D4.final -m3 0 -m4 1

echo "Extracting the alignment " 1>&2
path=$(readlink -f "$ROOT")
/data5/whozhwang/mosesdecoder/scripts/training/giza2bal.pl -d ${path}/giza.5/en-zh.A3.final -i ${path}/giza-inverse.5/zh-en.A3.final |/data5/whozhwang/mosesdecoder/bin/symal -alignment="intersect" -diagonal="no" -final="no" -both="no" > ./aligned.intersect

