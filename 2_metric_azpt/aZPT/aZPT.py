import argparse
import re
from typing import List, Dict, AnyStr, Tuple
from alignment import load_alignments, get_zpt_pos, get_target_range

pattern = '<.*?>_[OSPR]a*?'
pattern1 = '<.*?>'
both_tag = False
case_weight = [1, 0.5, 0, 0, 0, 0]

ZPR = {'<它>_S': 'it','<它>_P': 'its','<它>_Pa': 'it', '<它>_O': 'it', '<我>_S': 'i', '<我>_O': "me",'<我>_Pa': "my", '<他>_S': 'he','<他>_Pa': 'his', '<他>_O': 'him', '<你>_S': 'you',
       '<你>_O': 'you', '<她>_S': 'she', '<她>_O': 'her','<她>_Pa': 'her', '<我们>_S': 'we', '<我们>_O': 'us', '<你们>_S': 'you', '<你们>_O': 'you',
       '<他们>_S': 'they', '<她们>_S': 'they', '<它们>_S': 'they', '<他们>_O': 'them', '<她们>_O': 'them', '<它们>_O': 'them',
       '<它的>_P': 'its', '<它的>_Pa': 'its', '<我的>_Pa': 'my', '<我的>_P': "mine", '<他的>_Pa': 'his', '<他的>_P': 'his',
       '<你的>_Pa': 'your', '<你的>_P': 'yours', '<她的>_Pa': 'her', '<她的>_P': 'her', '<我们的>_Pa': 'our', '<我们>_P': 'ours',
       '<你们的>_Pa': 'your', '<你们的>_Pa': 'yours', '<他们的>_Pa': 'their', '<她们的>_Pa': 'their', '<它们的>_Pa': 'their',
       '<他们的>_P': 'theirs', '<你>_Pa': 'your', '<你的>_S': 'you', '<他自己>_R':'himself','<他>_R':'himself',
       '<她们的>_P': 'theirs', '<它们的>_P': 'theirs', '<我自己>_R': 'myself',
       }
ZPT = {"S": ['i', 'you', 'he', 'she', 'it', 'we', 'they'], "O": ['me', 'you', 'him', 'her', 'it', 'us', 'them'],
       "Pa": ['my', 'your', 'him', 'her', 'its', 'our', 'their'],
       "P": ['mine', 'yours', 'his', 'her', 'its', 'ours', 'theirs'],
       "R": ['myself', 'yourself', 'himself', 'hersefl', 'itself', 'ourselves', 'yourselves', 'themselves']
       }


def check_is_contain_zp(line: AnyStr, pattern: AnyStr):
    m = [(n, x) for n, x in enumerate(line) if re.search(pattern, x)]
    return m, len(m) != 0


def load_data(filename: AnyStr) -> Tuple[Dict, List]:
    data = {}
    with open(filename, 'r', encoding='utf8') as f:
        for n, line in enumerate(f):
            line = line.strip().split()
            m, have_zp = check_is_contain_zp(line, pattern)
            tmp = {}
            if have_zp:
                tmp = {}
                for (i, x) in m:
                    tmp[i] = x

            data[n] = {'sent': ' '.join(line), 'zp': tmp}

    return data


def load_target(filename: AnyStr, index: List) -> Dict:
    data = {}
    with open(filename, 'r', encoding='utf8') as f:
        for n, line in enumerate(f):
            if n in index:
                data[n] = line.strip().lower().split()

    return data


def compare_source_and_target(source: List, target: List) -> List:
    source = set(source)
    target = set(target)
    tmp = source.difference(target)
    tmp2 = target.difference(source)
    return list(tmp), list(tmp2)


def get_zpt_candidates(zpt_range: List, target_sent: List) -> List:
    pre = min(zpt_range[0])
    post = max(zpt_range[-1])

    return target_sent[pre:post] if post==-1 else target_sent[pre:]


def check_zp_in_candidates(tgt_zpt_is: AnyStr, tgt_sent: List, align: List, n_neighbors: int, max_len: int) -> Tuple[
    bool, int]:
    flag = False

    for x in align:
        pre, post = get_target_range(x, n_neighbors, max_len)
        candidate = tgt_sent[pre:post + 1]
        pos = list(range(pre, post, 1))
        if tgt_zpt_is in candidate:
            flag = True
            pos = candidate.index(tgt_zpt_is) + pre
            break
    return flag, pos, candidate


def extract_zpt(zpt_pos: Dict, source: Dict, target: Dict, n_neighbors: int) -> Tuple:
    Case = []
    zpt_ref = []
    zpt_tgt = []
    zpt_tgt_pos = []

    for k, v in zpt_pos.items():
        src_zp = source[k]['zp']
        tgt_sent = target[k]
        max_len = len(tgt_sent)
        for m, n in v.items():
            # m : src_zp_pos,n:target_range
            if '_n' in src_zp[m]:
                tgt_zpt_is = "None"
            else:

                tgt_zpt_is = ZPR[src_zp[m]]
            zpt_ref.append(tgt_zpt_is)

            if n[-1]:
                # src_zp_pos have alignments
                # n=(zpt_range,Flag)
                n = n[0]

                candidate = [tgt_sent[x] for x in n]
                if tgt_zpt_is in candidate:
                    # case 1 target_zpt is same in reference:
                    #
                    Case.append(1)
                    zpt_tgt.append(tgt_zpt_is)
                    idx_tmp = candidate.index(tgt_zpt_is)
                    pos = n[idx_tmp]
                    zpt_tgt_pos.append(pos)
                else:
                    flag, pos, candidate = check_zp_in_candidates(tgt_zpt_is, tgt_sent, n, n_neighbors, max_len)

                    if flag:
                        Case.append(1)
                        zpt_tgt.append(tgt_zpt_is)
                        zpt_tgt_pos.append(pos)
                    else:
                        Case.append(-1)
                        zpt_tgt.append(candidate)
                        zpt_tgt_pos.append(pos)
            else:
                # src_zp_pos didn't have alignments
                # n = (zpt_range,Flag)
                candidate = get_zpt_candidates(n[0], tgt_sent)
                if tgt_zpt_is in candidate:
                    Case.append(1)
                    zpt_tgt.append(tgt_zpt_is)
                    pos = candidate.index(tgt_zpt_is) + min(n[0][0])
                    zpt_tgt_pos.append(pos)
                else:
                    Case.append(-1)
                    zpt_tgt.append(candidate)
                    pre = max(n[0][0])
                    post = max(n[0][-1])
                    post = max_len if post ==-1 else post
                    pos = list(range(pre,post,1))
                    zpt_tgt_pos.append(pos)
    return Case, zpt_ref, zpt_tgt, zpt_tgt_pos


def extract_zpr(data: Dict, is_source=True) -> Tuple:
    if is_source:
        for k, v in data.items():
            del v['sent']
    data = dict(sorted(data.items(),key=lambda x:x[0]))

    pos = []
    sent_id = []
    word = []
    for k, v in data.items():

        for m, n in v['zp'].items():
            sent_id.append(k)
            pos.append(m)
            word.append(n)
    return (sent_id, pos, word)


def write_to_file(filename: AnyStr, sent_id: List, src_zp_pos: List, src_zp: List, ref_zp: List, tgt_zpt_pos: List,
                  tgt_zpt: List, Case: List,tgt_sent:Dict):
    scores = {}
    with open(filename + '.details', 'w', encoding='utf8') as f:
        f.write('Sent\tSRC_ZP_Pos.\tSRC_ZP\tRef_ZP\tTGT_ZPT_Pos.\tTGT_ZPT\tCase\n')
        for si, szp, sz, rz, tzp, tz, c in zip(sent_id, src_zp_pos, src_zp, ref_zp, tgt_zpt_pos, tgt_zpt, Case):
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(si, szp, sz, rz, tzp, tz, c)
            f.write(line)
            if isinstance(tzp,List):
                for tzpp in tzp:
                    tgt_sent[si][tzpp]="<"+tgt_sent[si][tzpp]+">"
            else:
                tgt_sent[si][tzp] = "<" + tgt_sent[si][tzp] + ">"
            if sz in scores.keys():
                if c == 1:
                    scores[sz]['correct'] = scores[sz]['correct'] + 1
                else:
                    scores[sz]['incorrect'] = scores[sz]['incorrect'] + 1
            else:
                if c == 1:
                    scores[sz] = {'correct': 1, 'incorrect': 0}
                else:
                    scores[sz] = {'correct': 0, 'incorrect': 1}

    with open(filename+'.tagged','w',encoding='utf8') as f:
        k = sorted(list(tgt_sent.keys()))
        for i in k:
            line = ' '.join(tgt_sent[i])
            f.write(line+'\n')

    with open(filename + '.score', 'w', encoding='utf8') as f:

        correct = 0
        incorrect = 0
        for k, v in scores.items():
            cor = v['correct']
            incor = v['incorrect']
            if '_n' in k:
                zpr = 'None'
            else:
                zpr = ZPR[k]
            line = 'SRC_ZP:{}\tRef_ZP:{}\tCorrect:{}\tIncorrect:{}\tAcc:{}\n'.format(k, zpr, cor, incor, cor / (incor+cor))
            f.write(line)
            correct += cor
            incorrect += incor

        f.write('Finally Correct entities:{}\t Incorrect:{}\t Score:{}\n'.format(correct, incorrect,
                                                                                 correct / (correct + incorrect)))


def main(args):
    # get configuration
    n_neighbors = args.num_neighbors

    # load data set
    source_file = load_data(args.source)
    source_index = list(source_file.keys())
    target_file = load_target(args.target, source_index)
    # reference_file = load_target(args.reference, source_index)
    assert len(source_file) == len(target_file), '{} vs. {}'.format(len(source_file), len(target_file))
    # assert len(target_file) == len(reference_file), '{} vs. {}'.format(len(target_file), len(reference_file))

    if both_tag:
        target_file, target_index = load_data(args.target)
        fn, fp = compare_source_and_target(source_index, target_index)

    # load alignments
    Align_src_tgt = load_alignments(args.tgt_align, source_index)
    # Align_src_ref = load_alignments(args.ref_align, source_index)

    # get <ZP>'s translation ZPT
    tgt_zpt_pos = get_zpt_pos(source_file, Align_src_tgt)
    # re_zpt_pos = get_zpt_pos(source_file, Align_src_ref)

    # get source pos,and zp
    sent_id, src_pos, src_zp = extract_zpr(source_file, True)
    case, zpt_ref, zpt_tgt, zpt_tgt_pos = extract_zpt(tgt_zpt_pos, source_file, target_file, n_neighbors)
    write_to_file(args.output+'_tgt',sent_id,src_pos,src_zp,zpt_ref,zpt_tgt_pos,zpt_tgt,case,target_file)
    # case, zpt_ref, zpt_tgt, zpt_tgt_pos = extract_zpt(re_zpt_pos, source_file, reference_file, n_neighbors)
    # write_to_file(args.output + '_ref', sent_id, src_pos, src_zp, zpt_ref, zpt_tgt_pos, zpt_tgt, case,reference_file)
    # # scoring


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-s', '--source', help='source input of MT')
    params.add_argument('-t', '--target', help='hypothesis of MT')
    # params.add_argument('-r', '--reference', help='Translation reference')
    params.add_argument('-at', '--tgt-align', help='Alignment between source and hypothesis')
    # params.add_argument('-ar', '--ref-align', help='Alignment between source and reference')
    params.add_argument('-n', '--num-neighbors', type=int, default=2, help='Number of neighbors')
    params.add_argument('-o','--output',type=str)
    args = params.parse_args()
    main(args)
