import argparse
import csv
from typing import Dict,List
import json
import re


def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            data.append(line)
    return data

def load_dict(filename):
    with open(filename,'r',encoding='utf8') as f:
        return json.load(f)


def make_dict(ZPR:Dict,data:List):
    pattern = '<[你我他她它].*?>'
    tmp = {}
    for line in data:
        group = re.findall(pattern, line)
        if len(group) > 0:
            for m in group:
                if m in tmp.keys():
                    tmp[m]=tmp[m]+1
                else:
                    tmp[m]=1
    tmp = sorted(tmp.items(),key=lambda x:x[1],reverse=True)
    for n,m in enumerate(tmp):
        ZPR[m[0]]=n+1


def make_labels(dp: List, zpr: Dict,finename):
    pattern = '<[你我他她它].*?>'
    with open(finename, 'w', encoding='utf8') as f:
        fnames = ['ID','Sent','Label']
        writer = csv.DictWriter(f,fieldnames=fnames,delimiter='\t')
        for ld in dp:
            labels = []
            group = re.findall(pattern,ld)
            if len(group)>0:
                for m in group:
                    ld=ld.replace(m,'*pro*')
                    labels.append(zpr[m])

            line = {'ID': 0, 'Sent': ld, 'Label':str(labels)}
            writer.writerow(line)


def main(args):
    ins = load_data(args.input)
    if args.zp_input is not None:
        ZPR = load_dict(args.zp_input)
    else:
        ZPR = {'None': 0}
        make_dict(ZPR,ins)
    with open(args.output+'.dict','w',encoding='utf8') as f:
        json.dump(ZPR,f,ensure_ascii=False)
    make_labels(ins,ZPR,args.output)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i', '--input')
    params.add_argument('-z', '--zp_input',default=None)
    params.add_argument('-o', '--output')
    args = params.parse_args()
    main(args)

