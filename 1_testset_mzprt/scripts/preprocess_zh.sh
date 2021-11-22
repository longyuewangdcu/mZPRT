DATA_ROOT=$1
FILE_HEAD=$2
LANG=zh


SCRIPT_ROOT=./
FILE=$FILE_HEAD
cat ${DATA_ROOT}/${FILE_HEAD}.zh | python $SCRIPT_ROOT/jieba_cws.py | sed  's/\t/\s/g' | sed  's/^[ \t]*//g' | sed  's/[ \t]*$//g' | sed  's/  */ /g' |  dd conv=lcase >$FILE_HEAD.tk.rs.lc.$LANG

