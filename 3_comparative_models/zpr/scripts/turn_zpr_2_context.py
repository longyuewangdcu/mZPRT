import argparse
import csv
import pandas
import more_itertools
import ast
from itertools import chain
import random
def read_csv(filename):
    with open(filename,'r',encoding='utf8') as f:
        csv_file = csv.reader(f,delimiter='\t')
        data =[line for line in csv_file]
        return data

def make_combination(data,size):
    return list(more_itertools.windowed(data,n=size,step=1))

def make_context(data):
    new = []
    remove_index = []
    for n,line in enumerate(data):
        sents = [s[1] for s in line]
        sents = ' [SPE] '.join(sents)
        zprs = [ast.literal_eval(x[-1]) for x in line]
        zprs = list(chain(*zprs))
        if len(zprs)==0:
            remove_index.append(n)
        new_line = {'ID':0,'Sent':sents,"Label":str(zprs)}
        new.append(new_line)
    return new,remove_index

def write_to_csv(filename,data,slices):
    fnames = ['ID', 'Sent', 'Label']
    ID = []
    Sent = []
    Lables = []

    for n,line in enumerate(data):
        print(n)
        if n in slices:
            continue
        ID.append(line['ID'])
        Sent.append(line['Sent'])
        Lables.append(line['Label'])
    tocsv = {"ID":ID,"Sent":Sent,"Lables":Lables}

    frame = pandas.DataFrame(tocsv)
    frame.to_csv(filename,sep='\t',index=False)



def filter_(data,remove_index):
    l = len(data)
    lr = len(remove_index)
    keep = l-lr
    slices = random.sample(remove_index,keep)
    return slices


def main(args):
    data = read_csv(args.input)
    data = make_combination(data,args.size)
    data,index = make_context(data)
    slices = filter_(data,index)
    write_to_csv(args.output,data,slices)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input')
    params.add_argument('-s','--size',default=1,type=int)
    params.add_argument('-o','--output')
    args = params.parse_args()
    main(args)
