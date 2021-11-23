import argparse
from typing import List,AnyStr,Dict


def load_data(filename:AnyStr)->Dict:
    data = {}
    count = 0
    with open(filename,'r',encoding='utf8') as f:
        tmp = []
        for line in f:
            if '[doc]' in line:
                if len(tmp)!=0:
                    data[count] = tmp
                    count+=1
                    tmp=[]
                continue
            line = line.strip()
            tmp.append(line)
        if len(tmp) != 0:
            data[count] = tmp
    return data


def make_context_file(data:Dict,num_ctx:int):

    for k,v in data.items():
        tmp = []
        for i in range(num_ctx,0,-1):
            # insert empty lines
            sep = ['']*i
            #remove lines in the end of document
            tmp_ = sep+v[:-i]
            tmp.append(tmp_)
        new_v = []
        for x in zip(*tmp):
            #concate previous sentences
            x=' [sep] '.join(list(x))
            new_v.append(x)
        assert len(new_v) ==len(v)
        data[k]=new_v
        tmp=[]


def write_to_file(filename:AnyStr,data:Dict):
    with open(filename,'w',encoding='utf8') as f:
        for i in range(len(data)):
            v = data[i]
            for line in v:
                f.write('{}\n'.format(line))


def make_noise_data(filename:AnyStr,output_file:AnyStr):
    with open(filename,'r',encoding='utf8') as f:
        with open(output_file,'w',encoding='utf8') as out:
            for line in f:
                out.write('{}\n'.format('<sep>'))

def main(args):
    #load_data
    if args.wmt:
        make_noise_data(args.input,args.output)
    else:
        data = load_data(args.input)
        #make ctx data
        make_context_file(data,args.num_ctx)
        write_to_file(args.output,data)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input',help='input file with document boundary \'[doc]\'')
    params.add_argument('-o','--output',help='output file')
    params.add_argument('-n','--num_ctx',type=int,help='num of ctx')
    params.add_argument('-w','--wmt',type=bool,default=False,help='make noise context for WMT dataset')
    args = params.parse_args()
    main(args)

