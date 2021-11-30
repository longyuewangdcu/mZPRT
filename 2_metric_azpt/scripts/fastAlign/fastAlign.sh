[ $# -eq 0 ] && { 
	echo "Usage: sh $0 [source] [target] [output_tmp_folder]";
	echo "	Source and target file should not contain line with 0-length."
	echo "	Otherwise, the alignment will stop earlier."
	exit -1;
}

program_path=/home/whozhwang/fast_align/build
source=$1
target=$2
output_directory=$3
f=zh
e=en
mkdir -p $output_directory

perl ~/scripts/prepare-fast-align.perl $source $target > $output_directory/corpus.${f}-${e}

$program_path/fast_align -i $output_directory/corpus.${f}-${e} -d -o -v -p $output_directory/${f}-${e}.table > $output_directory/forward.aligned 2> $output_directory/forward.err

$program_path/fast_align -i $output_directory/corpus.${f}-${e} -d -o -v -r -p $output_directory${e}-${f}.table > $output_directory/inverse.aligned 2> $output_directory/inverse.err 




