import argparse
from argparse import Namespace
from typing import AnyStr
import re

def make_data(args:Namespace,pattern:AnyStr):
    with open(args.input, 'r',encoding='utf') as f:
        with open(args.output, 'w',encoding='utf8') as out:
            for line in f:
                line = line.strip()
                if getattr(args,'zp',False):
                    line = re.sub(pattern,r'\1',line)
                else:
                    line = re.sub(pattern,'',line)
                out.write(line+'\n')


def main(args):
    pattern = '<([你我他它她们的自己]*?)>(_[SOPR]a*)*(_un)*'
    make_data(args,pattern)



if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input',help='Raw data with type tag')
    params.add_argument('-o','--output',help='output data without tag or original format')
    params.add_argument('-z','--zp',action='store_true',help='turn on for oracle format')
    args = params.parse_args()
    main(args)
