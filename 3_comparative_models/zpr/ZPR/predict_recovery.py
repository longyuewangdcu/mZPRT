import os, sys, json
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn

import utils
from main import forward_step
from transformers import BertTokenizer

def calc_f1(n_out, n_ref, n_both):
    # print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both / n_out if n_out > 0.0 else 0.0
    rc = n_both / n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0 * pr * rc / (pr + rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1

# def calc_f1(tmp):
#     # print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
#     tp = tmp['tp']
#     fp = tmp['fp']
#     pr = tp / (tp+fp) if (tp+fp) > 0.0 else 0.0
#     rc = tp / tmp['count'] if tmp['count'] > 0.0 else 0.0
#     f1 = 2.0 * pr * rc / (pr + rc) if pr > 0.0 and rc > 0.0 else 0.0
#     return pr, rc, f1
# [0,0] or 0 mean 'not applicable'
def add_counts(out, ref, counts):
    assert type(out) in (int, list)
    assert type(ref) in (int, list)
    if type(out) == int:
        out = [out, ]
    if type(ref) == int:
        ref = [ref, ]
    if sum(out) != 0:
        counts[1] += 1.0
    if sum(ref) != 0:
        counts[2] += 1.0
        if out == ref or tuple(out) in ref:
            counts[0] += 1.0

# def add_counts(out,ref,counts):
#     counts[ref]['count']+=1
#     if isinstance(out,list):
#         out = out[-1]
#     if out ==ref:
#         counts[ref]['tp']+=1
#     else:
#
#         counts[out]['fp']+=1
            
def make_data_recovery(conversation, tokenizer):
    data = {'sentences': [],  # [batch, wordseq]
            'sentences_bert_idxs': [],  # [batch, wordseq, wordlen]
            'sentences_bert_toks': [],  # [batch, seq]
            'zp_info': []}  # [a sequence of ...]
    conversation = json.load(conversation)
    data['sentences'] = conversation['sentences']
    data['sentences_bert_idxs'] = conversation['sentences_bert_idxs']
    data['sentences_bert_toks'] = conversation['sentences_bert_toks']
    data['zp_info'] = conversation['zp_info']
#     for i, sentence in enumerate(conversation):
#         sent=['[CLS]']
#         sent_bert_idxs = [[0],]
#         sent_bert_toks = ['[CLS]',]
#         sentence = sentence.strip().split()
#
#         for w in sentence:
#             sent.append(w)
#
#             char_ = tokenizer.tokenize(w)
#
# #             char_ = list(w)
#             idx = [x+len(sent_bert_toks) for x in range(len(char_))]
#             sent_bert_toks.extend(char_)
#
#             sent_bert_idxs.append(idx)
#         sent.append('[SEP]')
#
#         sent_bert_idxs.append([len(sent_bert_toks)])
#         sent_bert_toks.append('[SEP]')
#         data['sentences'].append(sent)
#         data['sentences_bert_idxs'].append(sent_bert_idxs)
#         data['sentences_bert_toks'].append(sent_bert_toks)

    return data

def dev_eval(model, model_type, development_sets, device,pro_mapping):
    model.eval()
    dev_eval_results = []
    with torch.no_grad():
        for devset in development_sets:

            data_type = devset['data_type']
            batches = devset['batches']

            assert data_type in ('recovery')
            print('Evaluating on dataset with data_type: {}'.format(data_type))
            N = 0
            
            dev_loss = {'total_loss': 0.0, 'detection_loss': 0.0, 'recovery_loss': 0.0}
            dev_counts = {'detection':[0.0 for _ in range(3)], 'recovery':  [0.0 for _ in range(3)]
                          }
            start = time.time()
            for step, ori_batch in enumerate(batches):
                # execution
                batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                         for k, v in ori_batch.items()}
                ori_sent = ori_batch['bert_char']

                step_loss, step_out = forward_step(model, batch)
                # record loss
                for k, v in step_loss.items():
                    dev_loss[k] += v.item() if type(v) == torch.Tensor else v
                # generate outputs
                input_zp, input_zp_cid = \
                    batch['input_zp'], batch['input_zp_cid']
                input_zp = input_zp.cpu().tolist()
                detection_out = step_out['detection_outputs'].cpu().tolist()
                detection_out_ = list(map(str,step_out['detection_outputs'].view(-1).cpu().tolist()))
                detection_out_ = ','.join(detection_out_)

                detect_v,detect_pos = torch.topk(step_out['detection_outputs'].squeeze(-1),3,dim=1)
                   
                if data_type == 'recovery':
                    input_zp_cid = input_zp_cid.cpu().tolist()
                    recovery_out = step_out['recovery_outputs'].cpu().tolist()
                    for i,(w,n) in enumerate(zip(ori_sent,recovery_out[0])):
                        if n == 0:
                            continue
                        else:
                            zpr ='<{}> '.format(pro_mapping[str(n)])
                            w =  zpr+w
                            ori_sent[i]=w
                    tmps = []

                    for k, v in batch['char2word'].items():
                        tmp = []
                        for w in v:
                            tmp.append(ori_sent[w])
                        ww = ''.join(tmp)
                        tmps.append(ww)
                    tmps = tmps[1:]
                    tmps = tmps[:-1]
                    tmps = ' '.join(tmps)
                    dev_eval_results.append(tmps)

                rev_v,rev_pos = torch.topk(step_out['recovery_outputs'],3,dim=-1)
                if detect_v.sum()!=0:
                    detect_pos,_ =detect_pos.sort()
                    rev_pos,_ = rev_pos.sort()
                    #if False in detect_pos.eq(rev_pos):
                        #print('detect_v:{}  detext_p:{}'.format(detect_v,detect_pos))
                        #print('rev_v:{}  rev_p:{}'.format(rev_v,rev_pos))
                # generate decision mask and lenghts
                if model_type == 'bert_char':  # if char-level model
                    mask = batch['input_decision_mask']
                    lens = batch['input_mask'].sum(dim=-1).long()
                else:
                    mask = batch['input_decision_mask']
                    lens = batch['input_wordmask'].sum(dim=-1).long()
                # update counts for calculating F1
                B = list(lens.size())[0]
                for i in range(B):
                    for j in range(1, lens[i] - 1):  # [CLS] A B C ... [SEP]
                        # for bert-char model, only consider word-boundary positions
                        # for word models, every position within 'input_wordmask' need to be considered
                        if mask[i, j] == 0.0:
                            continue
                        add_counts(out=detection_out[i][j], ref=input_zp[i][j],
                                   counts=dev_counts['detection'])
                        if data_type == 'recovery':
                            idx=input_zp_cid[i][j]
                            
                            add_counts(out=recovery_out[i][j], ref=input_zp_cid[i][j],
                                       counts=dev_counts['recovery'])
                    N += B
            # output and calculate performance
            total_loss = dev_loss['total_loss']
            duration = time.time() - start
            print('Loss: %.2f, time: %.3f sec' % (total_loss, duration))
            print(dev_counts['detection'])
            det_pr, det_rc, det_f1 = calc_f1(n_out=dev_counts['detection'][1],
                                             n_ref=dev_counts['detection'][2], n_both=dev_counts['detection'][0])

            print('Detection F1: %.2f, Precision: %.2f, Recall: %.2f' % (100 * det_f1, 100 * det_pr, 100 * det_rc))
            cur_result = {'data_type': data_type, 'loss': total_loss, 'detection_f1': det_f1}
            if data_type == 'recovery':
                print(dev_counts['recovery'])
                rec_pr, rec_rc, rec_f1 =calc_f1(n_out=dev_counts['recovery'][1],
                                             n_ref=dev_counts['recovery'][2], n_both=dev_counts['recovery'][0])
                # for k,v in dev_counts['recovery'].items():
                #     rec_pr, rec_rc, rec_f1 = calc_f1(n_out=v[1],
                #                                      n_ref=v[2], n_both=v[0])
                #     # rec_pr, rec_rc, rec_f1 = calc_f1(v)
                print('Lable: %d, Recovery F1: %.2f, Precision: %.2f, Recall: %.2f' % (k,100 * rec_f1, 100 * rec_pr, 100 * rec_rc))

                cur_result['key_f1'] = rec_f1
            if len(development_sets) > 1:
                print('+++++')


    return dev_eval_results


def inference(model, batches,tok,probing):
    model.eval()
    # output = open('detection_outputs','w',encoding='utf8')
    recovery_decisions = []
    with torch.no_grad():
        for step, ori_batch in enumerate(batches):
            # execution
            batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
            
            step_loss, step_out = forward_step(model, batch)
            recovery_out = step_out['recovery_outputs'].cpu().tolist()  # [batch, seq, 2]
            # generate decision mask and lenghts

            mask = batch['input_decision_mask']
            lens = batch['input_mask'].sum(dim=-1).long()
    
            # update counts for calculating F1
            B = lens.size(0)
            
        
            sent = tok.convert_ids_to_tokens(batch['input_ids'][0].tolist())
            detection_outputs = step_out['detection_outputs'].view(-1)

            pos = None
            
            if detection_outputs.sum().item()>=1:
                x,pos = torch.topk(step_out['recovery_outputs'].view(-1),1)
               
                if pos !=0:
                    x = '<'+probing[str(x.item())]+'> '
#                     x='[MASK]'
                
            tmps = []
            for k,v in batch['char2word'].items():
                tmp=[]
                for w in v:
                    if pos is not None and pos>0:
                        if w == pos.item():
                            tmp.append(x)
                    tmp.append(sent[w])
                ww = ''.join(tmp)
                tmps.append(ww)
            
            tmps = tmps[1:]
            tmps = tmps[:-1]
            tmp=' '.join(tmps)
            recovery_decisions.append(tmp)
    return recovery_decisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, required=True, help='Prefix path to the saved model')
    parser.add_argument('--in_path', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--out_path', type=str, default=None, help='Path to the output file.')
    args, unparsed = parser.parse_known_args()
    FLAGS = utils.load_config(args.prefix_path + ".config.json")

    print(FLAGS.model_type)

    import zp_datastream
    import zp_model


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if FLAGS.pretrained_path is not None:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))

    # ZP setting
    is_only_azp = False

    # load data and make_batches
    print('Loading data and making batches')
    # data_type = 'recovery_inference'
    data_type = 'recovery'
    data = make_data_recovery(open(args.in_path,'r',encoding='utf8'), tokenizer)
    # features = zp_datastream.load_and_extract_features(args.in_path, tokenizer,
    #                                           char2word=FLAGS.char2word, data_type=data_type)
    #
    features = zp_datastream.extract_features(data, tokenizer, char2word=FLAGS.char2word, data_type=data_type)
    
    batches = zp_datastream.make_batch(data_type, features, 1,is_sort=False, is_shuffle=False)
    devsets = []
    devsets.append({'data_type': data_type, 'batches': batches})
    print('Compiling model')
    model = zp_model.BertZP.from_pretrained(FLAGS.pretrained_path, char2word=FLAGS.char2word,
                                            pro_num=len(pro_mapping))
    model.load_state_dict(torch.load(args.prefix_path + ".bert_model.bin"))
    model.to(device)

    outputs=dev_eval(model, FLAGS.model_type, devsets, device,pro_mapping)
    # outputs = inference(model, batches,tokenizer,pro_mapping)
    with open(args.out_path,'w',encoding='utf8') as f:
        for line in outputs:
            f.write(line+'\n')
