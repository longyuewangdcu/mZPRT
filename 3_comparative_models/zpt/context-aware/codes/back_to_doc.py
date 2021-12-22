import argparse


def read_src(path):
    tmp = []
    with open(path,'r') as f:
        for n,line in enumerate(f):
            if '<p>' in line:
                tmp.append(n)
    return tmp

def read_data(path):
    tmp = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            tmp.append(line)
    return tmp

def add_doc(data,idx):
    for i in idx:
        data.insert(i,'<p>')

def main(args):
    idx = read_src(args.s)
    data = read_data(args.t)
    add_doc(data,idx)
    with open(args.o,'w',encoding='utf8') as f:
        for line in data:
            f.write(line+'\n')


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-s')
    params.add_argument('-t')
    params.add_argument('-o')
    args = params.parse_args()
    main(args)
