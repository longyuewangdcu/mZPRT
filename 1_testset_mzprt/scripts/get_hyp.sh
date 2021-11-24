data=$1
cat ${data} |grep -P '^H' |sort -V |cut -f 3 > ${data}.hpy
