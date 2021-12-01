import argparse
import re
import numpy as np

ZPR = {'<它>_S': 'it', '<它>_O': 'it', '<我>_S': 'i', '<我>_O': "me", '<他>_S': 'he', '<他>_O': 'him', '<你>_S': 'you',
       '<你>_O': 'you', '<她>_S': 'she', '<她>_O': 'her', '<我们>_S': 'we', '<我们>_O': 'us', '<你们>_S': 'you', '<你们>_O': 'you',
       '<他们>_S': 'they', '<她们>_S': 'they', '<它们>_S': 'they', '<他们>_O': 'them', '<她们>_O': 'them', '<它们>_O': 'them',
       '<它的>_P': 'its', '<它的>_Pa': 'its', '<我的>_Pa': 'my', '<我的>_P': "mine", '<他的>_Pa': 'his', '<他的>_P': 'his',
       '<你的>_Pa': 'your', '<你的>_P': 'yours', '<她的>_Pa': 'her', '<她的>_P': 'her', '<我们的>_Pa': 'our', '<我们>_P': 'ours',
       '<你们的>_Pa': 'your', '<你们的>_Pa': 'yours', '<他们的>_Pa': 'their', '<她们的>_Pa': 'their', '<它们的>_Pa': 'their',
       '<他们的>_P': 'theirs', '<你>_Pa': 'your', '<你的>_S': 'you',
       '<她们的>_P': 'theirs', '<它们的>_P': 'theirs', '<我自己>_R': 'myself',
       }
ZPT = {"S": ['I', 'you', 'he', 'she', 'it', 'we', 'they'], "O": ['me', 'you', 'him', 'her', 'it', 'us', 'them'],
       "Pa": ['my', 'your', 'him', 'her', 'its', 'our', 'their'],
       "P": ['mine', 'yours', 'his', 'her', 'its', 'ours', 'theirs'],
       "R": ['myself', 'yourself', 'himself', 'hersefl', 'itself', 'ourselves', 'yourselves', 'themselves']
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


def get_zpt(tgt_zp_id, tgt):
    tmp = [tgt[id].lower() for id in list(tgt_zp_id)]
    return tmp


def is_correct(zp_word, zpr_word):
    return zp_word in zpr_word

def get_boundary(idx,idxs):
    idxs_ = np.asarray(idxs)
    id_ = (idxs_ - idx).argmin()
    closest = idxs[id_]
    return closest


def find_closest_index_inside(idx,idxs):
    idxs = sorted(idxs)
    idx = idxs.index(idx)
    pre = idx - 1 if idx >= 1 else idx
    pre = idxs[pre]
    post = idx + 1 if idx < len(idxs) - 1 else idx
    post = idxs[post]

    return pre,post

def find_closest_index_outside(idx, idxs):
    idxs.append(idx)
    idxs = sorted(idxs)
    index = idxs.index(idx)
    pre = -1 if index == 0 else idxs[index-1]
    post = -1 if index == len(idxs)-1 else idxs[index+1]
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
    for n,(src, tgt, align) in enumerate(zip(source, target, alignment)):
        if len(src['zpr']) < 1:
            continue
        keys = align.keys()
        control_flag = False

        for zp in src['zpr']:

            Totol += 1
            types = zp.split('_')[-1]
            zp_id = src['sent'].index(zp)
            tgt_zp = ZPR[zp]
            if zp_id in keys:
                tgt_zp_id = align[zp_id]
                zpt_words = get_zpt(tgt_zp_id,tgt)
                control_flag = is_correct(tgt_zp,zpt_words)
                if control_flag:
                    T += 1
                    continue
                pre,post = find_closest_index_inside(zp_id,list(align.keys()))
                candis = [pre,post]
                tgt_zp_id = set(tgt_zp_id)
                if zp_id in candis:
                    candis.remove(zp_id)
                while control_flag is False and len(candis)!=0:
                    closed = get_boundary(zp_id,candis)
                    candis.remove(closed)
                    new_tgt_zp_id = align[closed]
                    tgt_zp_id.update(new_tgt_zp_id)
                    start,end = min(tgt_zp_id),max(tgt_zp_id)
                    new_zpt_word = ' '.join(tgt[start:end+1]).lower().split()
                    control_flag = is_correct(tgt_zp, new_zpt_word)
                if control_flag:
                    T += 1
                elif any(zpt_words) in ZPT[types]:
                    HT += 1
                else:
                    F += 1
            else:
                pre, post = find_closest_index_outside(zp_id, list(align.keys()))
                if pre ==-1:
                    start = 0
                    end = max(align[post])
                elif post == -1:
                    start = min(align[pre])
                    end = -1
                else:
                    start = min(align[pre])
                    end = max(align[post])
                zpt_words = ' '.join(tgt[start:end]).lower().split()
                control_flag = is_correct(tgt_zp, zpt_words)
                if control_flag:
                    T+=1
                elif any(zpt_words) in ZPT[types]:
                    HT+=1
                else:
                    F+=1


    return T, F, HT, Totol


def score_zpr(Totol, T, HT, F):
    acc = T / Totol
    half_acc = (T + HT) / Totol
    print('True {} False {} Half_{} Totol {}'.format(T, F, HT, Totol))
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
