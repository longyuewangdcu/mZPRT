# @Time : 2021/10/9 00:10
# @Author : Scewiner, Xu, Mingzhou
# @File: zp_score_acc.py
# @Software: PyCharm
import argparse
import numpy as np


def load_data(filename):
    data = {}
    with open(filename,'r',encoding='utf8') as f:
        for n,line in enumerate(f):
            score = line.strip().split('\t')
            sent_id = score[0]
            word = score[1]
            score = score[-1]
            data[n]= {'id':sent_id,'word':word,'score':abs(float(score))}

    return data


def constructive_score(base,hpo):
    assert len(base) == len(hpo)
    better=[]
    equal=[]
    worse=[]
    ranges = range(len(base))
    for i in ranges:
        b = base[i]
        bid = b['id']
        bw = b['word']
        b = b['score']
        h = hpo[i]
        hid = h['id']
        hw = h['word']
        h = h['score']
        assert bid == hid
        assert bw == hw
        delta = b-h
        if delta==0:
            equal.append(delta)
        elif delta >0:
            better.append(delta)
        else:
            worse.append(delta)
    totol = len(better)+len(equal)+len(worse)
    b_s = len(better)/totol
    e_s = len(equal)/totol
    w_s = len(worse)/totol
    print('Better:{} Equal:{} Worse:{}'.format(b_s,e_s,w_s))
    b_delta = np.mean(np.asarray(better))
    w_delta = np.mean(np.asarray(worse))
    print('Std: {}'.format(b_delta-w_delta))



def main(args):
    base = load_data(args.baseline)
    hpyo = load_data(args.hpo)
    constructive_score(base,hpyo)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-b','--baseline')
    params.add_argument('-p','--hpo')
    args = params.parse_args()
    main(args)