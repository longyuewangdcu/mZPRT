import argparse
from typing import (
    List,
    Dict,
    AnyStr
)
def load_docs(path:AnyStr)->Dict:
    docs = dict()
    count = 0
    with open(path,'r',encoding='utf8') as f:
        tmp = []
        for line in f:
            if '< doc >' in line:
                if len(tmp) >0:
                    docs[count] = tmp
                    count += 1
                    tmp = []
                continue
            tmp.append(line.strip())
        if len(tmp) !=0:
            docs[count] = tmp
    return docs

def make_data(docs_:Dict,n:int)->Dict:
   
    def make_doc(doc:List,n)->List:
        tmp = []
        tmps = []
        for i in range(n,0,-1):

            doc_tmp = ['']*i + doc[:-i]

            tmps.append(doc_tmp)
        
        for i in zip(*tmps):
            line = ' '.join(i)
            tmp.append(line)
        return tmp

    for k,v in docs_.items():
        print(k)
        doc=make_doc(v,int(n))
        docs_[k]=doc

def main(args):
    docs_data = load_docs(args.input)
  
    make_data(docs_data,args.num)
    with open(args.output,'w',encoding='utf8') as f:
        for k,v in docs_data.items():
            for l in v:
                f.write(l+'\n')


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input')
    params.add_argument('-o','--output')
    params.add_argument('-n','--num')
    args = params.parse_args()
    main(args)