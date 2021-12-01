# @Time : 2021/10/8 23:35
# @Author : Scewiner, Xu, Mingzhou
# @File: zp_score_constructive.py
# @Software: PyCharm
import argparse
import re

pattern='<.*?>'

def load_data(tgt_file,loss_file):
    data = []
    with open(tgt_file,'r',encoding='utf8') as tgt:
        with open(loss_file,'r',encoding='utf8') as loss:
            for n,(line,loss_line) in enumerate(zip(tgt,loss)):
                line = line.strip()
                loss_line = loss_line.strip().split()[:-1]
                m = re.findall(pattern,line)
                index = []
                line = line.split()
                if len(m)>0:
                    for mm in m:
                        idx = line.index(mm)

                        index.append(idx)
                else:
                    continue

                assert len(line)==len(loss_line),'tgt_line {} vs loss_line {}'.format(len(line),len(loss_line))
                new_line = [(w,l) for w,l in zip(line,loss_line)]
                data.append({'ID':str(n),'sent':new_line,'zpr':index})
    return data

def extract_sent(data):
    new_data = []
    for line in data:
        sent_id = line['ID']
        sent = line['sent']
        index = line['zpr']
        for idx in index:

            w,l = sent[idx]
            tmp = '\t'.join([sent_id,w,l])
            new_data.append(tmp)
    return new_data


def write_to_file(filename,data):
    with open(filename,'w',encoding='utf8') as f:
        for line in data:
            f.write('{}\n'.format(line))


def main(args):
    data = load_data(args.target,args.score)
    data = extract_sent(data)
    write_to_file(args.output,data)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-out','--output')
    params.add_argument('-tgt','--target')
    params.add_argument('-s','--score')
    args = params.parse_args()
    main(args)