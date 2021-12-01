import json
import numpy as np
import torch


def load_and_extract_features(path, tokenizer, char2word="sum", data_type="recovery"):
    data = json.load(open(path, 'r'))
    return extract_features(data, tokenizer, char2word=char2word, data_type=data_type)


def extract_features(data, tokenizer, char2word="sum", data_type="recovery"):
    assert data_type.startswith("recovery")
    print('Data type: {}, char2word: {}'.format(data_type, char2word))
    print("zp_datastream_char.py: for model_type 'bert_char', 'char2word' not in use")

    features = []
    sent_id_mapping = {}
    right, total = 0.0, 0.0
    for i, (sent_bert_toks, sent_bert_idxs) in enumerate(zip(data['sentences_bert_toks'], data['sentences_bert_idxs'])):
        if len(sent_bert_toks) > 512:
            print('Sentence No. {} length {}.'.format(i, len(sent_bert_toks)))
            continue
        tmp = sent_bert_toks
        sent_bert_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sent_bert_toks]
        right += sum([x == '[UNK]' for x in sent_bert_toks])
        total += len(sent_bert_toks)
        input_ids = tokenizer.convert_tokens_to_ids(sent_bert_toks) # [seq]
        sent_bert_toks = tmp
        # Example: sent_bert_idxs: [0] [1, 2, 3] [4]; sent_bert_toks: [CLS] A B C [SEP]; decision_start: 1
        # input_decision_mask = [0, 1, 0, 0, 1]

        decision_start = data['sentences_decision_start'][i] if 'sentences_decision_start' in data else 0
        input_decision_mask = []

        char2word_map = {}
        for j, idxs in enumerate(sent_bert_idxs):
            curlen = len(input_decision_mask)
            input_decision_mask.extend([0 for _ in idxs])
            if j >= decision_start:
                input_decision_mask[curlen-1] = 1
            char2word_map.update({j:idxs})
            
        assert len(input_ids) == len(input_decision_mask)
        features.append({'input_ids':input_ids, 'char2word':char2word_map,
            'input_decision_mask':input_decision_mask,'bert_char':sent_bert_toks})
        sent_id_mapping[i] = len(features) - 1
    print('OOV rate: {}, {}/{}'.format(right/total, right, total))

    is_inference = data_type.find('inference') >= 0
    if data_type.startswith('recovery'):
        extract_recovery(data, features, sent_id_mapping, is_inference=is_inference)
    else:
        assert False, 'Unknown'

    return features

def extract_recovery(data, features, sent_id_mapping, is_inference=False):
    if is_inference:
        return

    for inst in features:
        input_ids = inst['input_ids']
        inst['input_zp'] = [0 for _ in input_ids] # [seq]
        inst['input_zp_cid'] = [0 for _ in input_ids] # [seq]

    for zp_inst in data['zp_info']:
        i, j_char = zp_inst['zp_sent_index'], zp_inst['zp_char_index']
        assert j_char >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        pro_cid = zp_inst['recovery']
        assert type(pro_cid) == int
        features[i]['input_zp'][j_char] = 1
        features[i]['input_zp_cid'][j_char] = pro_cid


def make_batch(data_type, features, batch_size, is_sort=True, is_shuffle=False):
    assert data_type.startswith("recovery") or data_type.startswith("resolution")
    is_inference = data_type.find('inference') >= 0
    if data_type.startswith("recovery"):
        return make_recovery_batch(features, batch_size,
                is_inference=is_inference, is_sort=is_sort, is_shuffle=is_shuffle)
    else:
        assert False, 'Unknown'

def make_recovery_batch(features, batch_size, is_inference=False, is_sort=True, is_shuffle=False):
    if is_shuffle:
        np.random.seed(222)
        np.random.shuffle(features)
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))

    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        maxseq = 0
        for i in range(0, B):
            maxseq = max(maxseq, len(features[N+i]['input_ids']))
        input_ids = np.zeros([B, maxseq], dtype=np.long)
        input_mask = np.zeros([B, maxseq], dtype=np.float)
        input_decision_mask = np.zeros([B, maxseq], dtype=np.float)
        if is_inference == False:
            input_zp = np.zeros([B, maxseq], dtype=np.long)
            input_zp_cid = np.zeros([B, maxseq], dtype=np.long)
        else:
            input_zp = input_zp_cid = None
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            input_decision_mask[i,:curseq] = features[N+i]['input_decision_mask']
            if is_inference == False:
                input_zp[i,:curseq] = features[N+i]['input_zp']
                input_zp_cid[i,:curseq] = features[N+i]['input_zp_cid']
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.long)
        if is_inference == False:
            input_zp = torch.tensor(input_zp, dtype=torch.long)
            input_zp_cid = torch.tensor(input_zp_cid, dtype=torch.long)



        batches.append({'input_ids':input_ids, 'input_mask':input_mask,
            'input_decision_mask':input_decision_mask,
            'input_zp':input_zp, 'input_zp_cid':input_zp_cid,
            'type':'recovery','char2word':features[N]['char2word'],'bert_char':features[N]['bert_char']})
        N += B
    return batches


if __name__ == '__main__':
    pass
