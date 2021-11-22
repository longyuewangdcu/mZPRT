cp $1 ctx
sed -i ':a;N;$!ba;s/\n<p>/<p>/g' ctx
sed '$d' ctx
sed -i '/<p>/d' $1
paste ctx $1 > tmp
sed -i 's/.*<p>//g' tmp
sed -i 's/^\t//g' tmp
sed -i 's/\t/ <S> /g' tmp
sed -i 's/>_[SOPR]a*/>/g' tmp
#sed -i 's/<[^>S]*>//g' tmp
#sed -i 's/  / /g' tmp
python ~/mingzhou/tool/scripts/make_zpr_data.py -i tmp -o ZP_in
sed -i 's/<[^>S]*>//g' ZP_in
sed -i 's/  / /g' ZP_in
python ~/mingzhou/ZP/tvsub/1_make_json_new.py ZP_in
