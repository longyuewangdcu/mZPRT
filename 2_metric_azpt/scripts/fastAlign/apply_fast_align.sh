File=fastalign_output_dir
source=$1
target=$2
bitext=$3
output=$4
cat $target |dd conv=lcase > $target.low
perl ./prepare-fast-align.perl $source $target.low > $bitext
python2 ~/fast_align/build/force_align.py $File/forwar.table $File/forward.err $File/inverse.table $File/inverse.err < $bitext > $output
