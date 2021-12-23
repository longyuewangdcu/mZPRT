import argparse
import hashlib
import json
def make_hash(line):
    return hashlib.md5(line.encode('utf8')).hexdigest()

def load_data(filename):
    data = {}
    with open(filename,'r',encoding='utf8') as f:
        for n,line in enumerate(f):
            line = line.strip()
            code = make_hash(line)
            data[code]=n
    return data

def make_index_mapping(src,filename):
    index_mapping={}
    keys = src.keys()
    with open(filename,'r',encoding='utf8') as f:
        for n,line in enumerate(f):
            line = line.strip()
            code = make_hash(line)
            if code in keys:
               index_mapping[n]=src[code]
            else:
               index_mapping[n]=None
    return index_mapping

def wirte_to_file(filename,data):
    with open(filename,'w',encoding='utf8') as f:
        json.dump(data,f,indent=4)

def main(args):
    source = load_data(args.source)
    print(len(source))
    #input_file = load_data(args.input)
    index_mapping = make_index_mapping(source,args.input)
    wirte_to_file(args.output,index_mapping)
        


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input')
    params.add_argument('-s', '--source')
    params.add_argument('-o','--output')
    args = params.parse_args()
    main(args)
