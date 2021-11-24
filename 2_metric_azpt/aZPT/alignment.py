import numpy as np
from typing import List,AnyStr,Dict,Tuple


def load_alignments(filename:AnyStr,index:List)->Dict:
    aligns = {}
    with open(filename,'r',encoding='utf8') as f:
        for n,line in enumerate(f):
            if n not in index:
                continue
            line = line.strip().split()
            tmp = {}
            for ll in line:
                ll=list(map(int,ll.split('-')))
                s,t = ll[0],ll[1]
                if s in tmp.keys():
                    tmp[s].append(t)
                else:
                    tmp[s]=[t]
            aligns[n] = tmp
    return aligns


def get_left_positions(mid:int,src_pos:List[int])->int:
    # [x,x,pre,mid,x,x,x]->pre
    if mid not in src_pos:
        src_pos.append(mid)
    src_pos = sorted(src_pos)
    idx = src_pos.index(mid)
    if idx ==0:
        return None
    pre = idx-1
    return src_pos[pre]

def get_right_positions(mid:int,src_pos:List[int])->int:
    # [x,x,x,mid,post,x,x]->post

    if mid not in src_pos:
        src_pos.append(mid)
    src_pos = sorted(src_pos)
    idx = src_pos.index(mid)
    if idx ==len(src_pos)-1:
        return None
    post = idx+1
    return src_pos[post]

def get_target_range(zpt_p:int,n_neighbors:int,max_len:int)->Tuple:
    pre =max((zpt_p-n_neighbors),0)
    post = min((zpt_p+n_neighbors),(max_len-1))+1
    return pre,post

def get_zpt_pos(source:Dict,alignments:Dict)->Dict:
    '''
    :param source:  Dict{'sent_id':{'sent': source_sent,'zp':{'pos':<zp>}}}
    :param alignments: Dict{'sent_id':{src_pos:[tgt_poss]}}
    :return: Dict{'sent_id':{src_p:[tgt_pos]}}
    '''
    keys=list(source.keys())
    new_dict = {}
    for k in keys:
        src_pos = list(source[k]['zp'].keys()) # [zp_pos,zp2_pos]
 
        align = alignments[k] #{src_pos:[tgt_poss]}


        tmp={}
        for src_p in src_pos:
            # check if <zp> have aligned information

            zpt_p = align.get(src_p)
            if zpt_p is None:
                # search range in target
                pre = get_left_positions(src_p,list(align.keys()))
                post = get_right_positions(src_p,list(align.keys()))

                if pre is not None:
                    pre = align[pre]
                else:
                    pre = [0]
                if post is not None:
                    post= align[post]
                else:
                    post = [-1]
                zpt_p = [pre,post]
                tmp[src_p]=(zpt_p,False)
            else:
                tmp[src_p]=(zpt_p,True)
        new_dict[k]=tmp
    return new_dict






