File=/Users/scewiner/Documents/Project/mZPRT/2_metric_azpt/scripts/fastAlign
align=/Users/scewiner/Documents/Project/new_eval/zh-en-out
source=$1
target=$2
bitext=$3
output=$4
cat $target |dd conv=lcase > $target.low
perl $File/prepare-fast-align.perl $source $target.low > $bitext
python2 /Users/scewiner/Documents/Project/fast_align/build/force_align.py $align/forwar.table $align/forward.err $align/inverse.table $align/inverse.err < $bitext > $output
