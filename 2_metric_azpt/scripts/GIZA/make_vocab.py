import argparse
from typing import AnyStr,Dict
def load_vcb(path:AnyStr)->Dict:
    data = {}
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            for w in line:
                data[w] = data.get(w,default=0)+1


    data =dict(sorted(data.items(),key=lambda x:x[-1],reverse=True))
    return data

def make_vocab(data:Dict,output:AnyStr):
    with open(output,'w',encoding='utf8') as f:
        f.write('1\tUNK\t0\n')
        for n,line in enumerate(data.items()):
            n=str(n+2)
            w,c = line[0],str(line[1])
            line = '\t'.join([n,w,c])
            f.write(line+'\n')



def main(args:argparse.Namespace):
    data = load_vcb(args.input)
    make_vocab(data,args.output)



if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--input')
    params.add_argument('--output')
    args = params.parse_args()
    main(args)
