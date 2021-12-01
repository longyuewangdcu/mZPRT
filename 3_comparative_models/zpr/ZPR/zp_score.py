import argparse
import re
import numpy as np

ZPR = {'<它>_S': 'it', '<它>_O': 'it', '<我>_S': 'i', '<我>_O': "me", '<他>_S': 'he', '<他>_O': 'him', '<你>_S': 'you',
       '<你>_O': 'you', '<她>_S': 'she', '<她>_O': 'her', '<我们>_S': 'we', '<我们>_O': 'us', '<你们>_S': 'you', '<你们>_O': 'you',
       '<他们>_S': 'they', '<她们>_S': 'they', '<它们>_S': 'they', '<他们>_O': 'them', '<她们>_O': 'them', '<它们>_O': 'them',
       '<它的>_P': 'its', '<它的>_Pa': 'its', '<我的>_Pa': 'my', '<我的>_P': "mine", '<他的>_Pa': 'his', '<他的>_P': 'his',
       '<你的>_Pa': 'your', '<你的>_P': 'yours', '<她的>_Pa': 'her', '<她的>_P': 'her', '<我们的>_Pa': 'our', '<我们>_P': 'ours',
       '<你们的>_Pa': 'your', '<你们的>_Pa': 'yours', '<他们的>_Pa': 'their', '<她们的>_Pa': 'their', '<它们的>_Pa': 'their',
       '<他们的>_P': 'theirs', '<你>_Pa': 'your','<你的>_S': 'you',
       '<她们的>_P': 'theirs', '<它们的>_P': 'theirs','<我自己>_R':'myself',
       }
ZPT = {"S": ['I', 'you', 'he', 'she', 'it', 'we', 'they'], "O": ['me', 'you', 'him', 'her', 'it', 'us', 'them'],
       "Pa": ['my', 'your', 'him', 'her', 'its', 'our', 'their'],
       "P": ['mine', 'yours', 'his', 'her', 'its', 'ours', 'theirs'],
       "R":['myself','yourself','himself','hersefl','itself','ourselves','yourselves','themselves']
       }
ZPR_reverse = {}


def load_sents(filename, is_src=False):
    data = []
    pattern = '<.*?>_[OSPR]a*'
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            if is_src:
                m = re.findall(pattern, line)
            line = line.strip().split()
            if is_src:
                new = {'sent': line, 'zpr': list(m)}
            else:
                new = line
            data.append(new)
    return data


def load_align(filename):
    aling = []
    with open(filename, 'r', encoding='utf8') as f:

        for line in f:
            line = line.strip().split()
            tmp = {}
            for w in line:
                w = w.split('-')

                s = int(w[0])
                t = int(w[1])
                if s in tmp.keys():

                    tmp[s].append(t)

                else:
                    tmp[s] = [t]
            aling.append(tmp)
    return aling




def check_zpr_in_target(zp_id, tgt, align):

    if zp_id in align.keys():
        tgt_ids = align[zp_id]

    #     # pre = min(tgt_id)
    #     # post = max(tgt_id)+1
    # else:
    k=0
    pre, post = find_closest_index(zp_id, list(align.keys()))
    pre_tgt_id = align[pre]
    post_tgt_id = align[post]
    tgt_ids = pre_tgt_id + post_tgt_id
    if zp_id in align.keys():
        tgt_ids += align[zp_id]
    if pre == post:
        k=3

    pre = max(min(tgt_ids)-k,0)
    post = max(tgt_ids) +1+ k
    tgt_words = ' '.join(tgt[pre:post]).lower().split()
    return tgt_words


def check_zpr_in_target_arround(zp_id, tgt, align, k):
    pre, post = find_closest_index(zp_id, list(align.keys()))
    pre_tgt_id = align[pre]
    post_tgt_id = align[post]
    tgt_ids = pre_tgt_id + post_tgt_id
    t=0
    if zp_id in align.keys():
        tgt_ids += align[zp_id]
    if pre == post:
        t=1
    pre = max(min(tgt_ids) - t, 0)
    post = max(tgt_ids) + 1 + t
    tgt_words = ' '.join(tgt[pre:post]).lower().split()

    if zp_id in align.keys():
        tgt_id = align[zp_id]
        tgt_words = [tgt[i] for i in tgt_id]
        arroud = close_to_arround(tgt_id, tgt_ids, k)
    else:
        arroud = -1

    return tgt_words, arroud


def find_closest_index(idx, idxs):
    idxs = sorted(idxs)
    if idx in idxs:
        idx = idxs.index(idx)
        pre = idx - 1 if idx >= 1 else idx
        pre = idxs[pre]
        post = idx + 1 if idx < len(idxs)-1 else idx
        post = idxs[post]
    else:
        idxs_ = np.asarray(idxs)
        closest_index = (np.abs(idxs_-idx)).argmin()
        tmp = idxs[closest_index]
        if tmp > idx:
            post = tmp
            pre = idxs[closest_index-1] if closest_index>0 else post
        else:
            pre = tmp
            post = idxs[closest_index+1] if closest_index<(len(idxs)-1) else pre

    return pre, post


def close_to_arround(zp_id, arround_idx, k):
    arround_idx = np.asarray(arround_idx)
    for idx in zp_id:
        if min(np.abs(arround_idx - idx)) <= k:
            return 1

    return 0


def align_count(source, target, alignment, cosider_arround=0):
    T = 0
    F = 0
    HT = 0
    Totol = 0
    for src, tgt, align in zip(source, target, alignment):
        if len(src['zpr']) < 1:
            continue
        for zp in src['zpr']:
            Totol += 1
            types = zp.split('_')[-1]
            zp_id = src['sent'].index(zp)
            tgt_zp = ZPR[zp]
            if cosider_arround != 0:
                # print('consider arround')
                tgt_words, arround = check_zpr_in_target_arround(zp_id, tgt, align, cosider_arround)
                if arround == 1:
                    if tgt_zp in tgt_words:
                        T += 1
                    else:
                        F += 1
                elif arround == -1:
                    if tgt_zp in tgt_words:
                        T += 1
                    elif any(tgt_words) in ZPT[types]:
                        HT += 1
                else:
                    if tgt_zp in tgt_words:
                        HT += 1
                    else:
                        F += 1
            else:
                # print('Don\'t consider arround')
                tgt_words = check_zpr_in_target(zp_id, tgt, align)
                if tgt_zp in tgt_words:
                    T += 1
                elif any(tgt_words) in ZPT[types]:
                    HT += 1

                else:
                    F += 1

    return T, F, HT, Totol


def score_zpr(Totol, T, HT, F):
    acc = T / Totol
    half_acc = (T + HT) / Totol
    print('True {} False {} Half_{} Totol {}'.format(T,F,HT,Totol))
    print('Acc. {} and Half_Acc. {}'.format(acc, half_acc))


def main(args):
    source = load_sents(args.source, True)  # [{'sent':[words],'zpr':[]}]
    target = load_sents(args.target)  # [[words]]
    align = load_align(args.alignment)  # [{s_id:[t_ids]}]
    T, F, HT, Totol = align_count(source, target, align, args.window)
    score_zpr(Totol, T, HT, F)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-src', '--source')
    params.add_argument('-tgt', '--target')
    params.add_argument('-align', '--alignment')
    params.add_argument('-w', '--window', type=int, default=0)
    args = params.parse_args()
    main(args)
