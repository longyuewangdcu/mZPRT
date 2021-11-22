DATA_ROOT=$1                                                                                 

FILE=$2

LANG_TRG=$3

cp $DATA_ROOT/$FILE.$LANG_TRG $DATA_ROOT/$FILE.rs.$LANG_TRG
sed -i 's/\t/\s/g' $DATA_ROOT/$FILE.rs.$LANG_TRG
sed -i 's/^[ \t]*//g' $DATA_ROOT/$FILE.rs.$LANG_TRG
sed -i 's/[ \t]*$//g' $DATA_ROOT/$FILE.rs.$LANG_TRG
sed -i 's/  */ /g' $DATA_ROOT/$FILE.rs.$LANG_TRG
