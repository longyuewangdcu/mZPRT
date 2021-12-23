import argparse
import csv
from typing import List

def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            data.append(line)
    return data


def make_labels(dp: List, zpr: List,finename):
    with open(finename, 'w', encoding='utf8') as f:
        for ld, lr in zip(dp, zpr):
            assert len(ld) == len(lr)

            while '<DP>' in ld:
                index = ld.index('<DP>')
                label = lr[index]
                label = '<{}>'.format(label)
                ld[index]='*pro*' 
                lr[index]=label
            lr = ' '.join(lr)
            f.write(lr+'\n')


def main(args):
    ins = load_data(args.input)
    print(len(ins))
    zps = load_data(args.zp_input)
    print(len(zps))
    make_labels(ins,zps,args.output)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i', '--input')
    params.add_argument('-z', '--zp_input')
    params.add_argument('-o', '--output')
    args = params.parse_args()
    main(args)

