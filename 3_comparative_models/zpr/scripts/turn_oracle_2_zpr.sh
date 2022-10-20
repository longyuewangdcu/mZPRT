cp $1 ctx
sed -i'.bak' -e ':a' -e 'N' -e '$!ba' -e 's/\n<p>/<p>/g' ctx
sed -i'.bak' -e '$d' ctx
sed -i'.bak' -e '/<p>/d' $1
paste ctx $1 > tmp
sed -i'.bak' -e 's/.*<p>//g' tmp
sed -i'.bak' -e 's/^\t//g' tmp
sed -i'.bak' -e 's/\t/ <S> /g' tmp
sed -i'.bak' -e 's/>_[SOPR]a*/>/g' tmp
#sed -i 's/<[^>S]*>//g' tmp
#sed -i 's/  / /g' tmp
python /Users/scewiner/Documents/Project/mZPRT/3_comparative_models/zpr/scripts/make_zpr_data.py -i tmp -o ZP_in -z ZP_in.dict
sed -i'.bak' -e 's/<[^>S]*>//g' ZP_in
sed -i'.bak' -e 's/  / /g' ZP_in
#python /Users/scewiner/Documents/Project/mZPRT/3_comparative_models/zpr/scripts/1_make_json_new.py ZP_in
