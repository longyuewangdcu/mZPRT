data=$1
cat ${data} |grep -p '^H' |sort -V |cut -f 3 > ${data}.hpy
