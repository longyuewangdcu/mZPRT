DATA_ROOT=$1
FILE_HEAD=$2
LANG=en

FILE=$FILE_HEAD
SCRIPT_ROOT=./
perl $SCRIPT_ROOT/tokenizer.perl -no-escape -l en -threads 20 < $DATA_ROOT/$FILE.$LANG > $DATA_ROOT/$FILE.tk.$LANG

SCRIPT_ROOT=./
FILE=$FILE_HEAD.tk
$SCRIPT_ROOT/remove_space.sh $DATA_ROOT $FILE $LANG
