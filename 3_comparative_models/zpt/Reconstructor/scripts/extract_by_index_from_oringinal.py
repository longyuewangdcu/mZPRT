import argparse
import json

def load_data(filename):
    data = {}
    with open(filename,'r',encoding='utf8') as f:
        for n,line in enumerate(f):
            line = line.strip()
            data[n]=line

    return data

def load_index(filename):
    return json.load(open(filename,'r',encoding='utf8'))

def reconstruct_data(index,data,tgt):
    target = {}
    for k,v in index.items():
        if v is None:
           print(k)
           line = '@@EMPTY@@'
           target[int(k)] = line +'\t' + tgt[int(k)]
        else:
           line = data[v]
           target[int(k)] = line +'\t' + tgt[int(k)]
    target = dict(sorted(target.items()))
    return target

def write_to_file(filename,data):
    with open(filename,'w',encoding='utf8') as f:
        for k, v in data.items():
            f.write('{}\n'.format(v))

def main(args):
    source = load_data(args.source)
    target = load_data(args.target)
    index = load_index(args.index)
    data = reconstruct_data(index,source,target)
    write_to_file(args.output,data)

if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--index')
    params.add_argument('-o','--output')
    params.add_argument('-s','--source')
    params.add_argument('-t','--target')
    args = params.parse_args()
    main(args)
