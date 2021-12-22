# @Time : 2020/12/8 16:03
# @Author : Scewiner, Xu, Mingzhou
# @File: cache_object.py
# @Software: PyCharm

import torch


class CacheBase(object):
    def __init__(self):
        self.version = "cache method"

    def print_version(self):
        print('============\n{}\n============'.format(self.version))

    def update_future(self):
        raise NotImplementedError

    def update_future_eval(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def get_idx(self):
        raise NotImplementedError

    def update(self):
        raise


class CacheTu(CacheBase):
    def __init__(self):
        self.version = "cache method propose by Tu et al. EMNLP17"

    def print_version(self):
        print('============\n{}\n============'.format(self.version))

    def update_future(self, cache, index, word, key, value, mask=None):
        cache.pop('future')
        cache['future'] = {}
        key = key.detach().clone().flatten(0, 1)
        value = value.detach().clone().flatten(0, 1)
        word = word.flatten()

        key = key.unbind()
        value = value.unbind()
        word = word.unbind()
        assert len(word) == len(key)
        assert len(word) == len(value)
        for w, k, v in zip(word, key, value):
            if w.item() <= 2:
                continue
            if w.item() in cache['future'].keys():
                ks, vs = cache['future'][w.item()]
                cache['future'][w.item()] = ((ks + k) / 2, (vs + v) / 2)
            else:
                cache['future'][w.item()] = (k, v)

    def update_future_eval(self):
        return self.update_future

    def read(self, cache, id, ctx):
        new_key = []
        new_value = []
        for k, v in cache['now'].items():
            new_key.append(v[0].unsqueeze(0))
            new_value.append(v[1].unsqueeze(0))
        return torch.cat(new_key, dim=0), torch.cat(new_value, dim=0), None

    def get_idx(self, cout):
        argmax_ = torch.argmax(cout, dim=-1).bool()
        argmax_ = argmax_.flatten().nonzero().view(-1)
        return argmax_

    def update(self, cache, cache_size):
        if 'future' not in cache.keys() or len(cache['future']) == 0:
            return 0

        for word, v in cache['future'].items():
            key, value = v[0], v[1]
            cache_keys = cache['now'].keys()
            if word in cache_keys:
                cache['now'][word] = [(key + cache['now'][word][0]) / 2, (value + cache['now'][word][0])]
            elif len(cache['now']) < cache_size:
                cache['now'][word] = [key, value]
            else:
                cache_keys = list(cache_keys)
                cache['now'].pop(cache_keys[0])
                cache['now'][word] = [key, value]
        cache['future'].clear()


class CacheDoc(CacheBase):
    def __init__(self):
        self.version = "cache with doc boundary"

    def print_version(self):
        print('============\n{}\n============'.format(self.version))

    def update_future(self, cache, index, word, key, value, mask=None):
        cache.pop('future')
        cache['future'] = {}
        key = key.detach().clone().transpose(0, 1)
        value = value.detach().clone().transpose(0, 1)
        word = word
        if len(index.tolist()) == 0:
            return 0

        key = key.unbind()
        value = value.unbind()
        word = word.unbind()
        index = index.unbind()
        assert len(word) == len(key)
        assert len(word) == len(value)
        for idx, w, k, v in zip(index, word, key, value):
            w = w.unbind()
            k = k.undind()
            v = v.unbind()
            idx = idx.item()
            if idx not in cache['future'].keys():
                cache['future'][idx] = {}
            for wb, kb, vb in zip(w, k, v):
                if wb.item() <= 3:
                    continue
                if wb.item() in cache['future'][idx].keys():
                    ks, vs = cache['future'][idx][wb.item()]
                    cache['future'][idx][w.item()] = ((ks + kb) / 2, (vs + vb) / 2)
                else:
                    cache['future'][idx][w.item()] = (kb, vb)

    def update_future_eval(self):
        return self.update_future

    def read(self, cache, idx, ctx):
        new_key = []
        new_value = []
        word = []
        idx = idx.unbind()

        for n, ids in enumerate(idx):
            ids = ids.item()
            tmp_k = []
            tmp_v = []
            tmp_w = []
            if ids in cache['now'].keys():
                for k, v in cache['now'][ids].items():
                    tmp_k.append(v[0].unsqueeze(0))
                    tmp_v.append(v[1].unsqueeze(0))
                    tmp_w.append(k)
                tmp_k = torch.cat(tmp_k, dim=0)
                tmp_v = torch.cat(tmp_v, dim=0)
            else:
                tmp_k = [ctx]
                tmp_v = [ctx]
                tmp_w = [1]
            tmp_w = torch.tensor(tmp_w, dtype=torch.int)
            new_key.append(tmp_k)
            new_key.append(tmp_v)
            word.append(tmp_w)
        return torch.nn.utils.rnn.pad_sequence(new_key, batch_first=True), torch.nn.utils.rnn.pad_sequence(new_value,
                                                                                                           batch_first=True), torch.nn.utils.rnn.pad_sequence(
            word, batch_first=True, padding_value=0)

    def get_idx(self, cout):
        argmax_ = torch.argmax(cout, dim=-1).bool()
        argmax_ = argmax_.flatten().nonzero().view(-1)
        return argmax_

    def update(self, cache, cache_size):
        if 'future' not in cache.keys() or len(cache['future']) == 0:
            return 0
        for idx, data in cache['future'].items():
            if idx not in cache['now'].keys() and len(cache['future'][idx]) != 0:
                cache['now'][idx] = {}
            for word, v in data.items():
                key, value = v[0], v[1]
                cache_keys = cache['now'][idx].keys()
                if word in cache_keys:
                    cache['now'][idx][word] = [(key + cache['now'][idx][word][0]) / 2,
                                               (value + cache['now'][idx][word][0])]
                elif len(cache['now']) < cache_size:
                    cache['now'][idx][word] = [key, value]
                else:
                    cache_keys = list(cache_keys)
                    cache['now'][idx].pop(cache_keys[0])
                    cache['now'][idx][word] = [key, value]
        cache['future'].clear()


class CacheDocSelect(CacheBase):
    def __init__(self):
        self.version = "cache with doc boundary,and pos tagger"

    def print_version(self):
        print('============\n{}\n============'.format(self.version))

    def update_future(self, cache, index, word, key, value, select_mask):
        cache.pop('future')
        cache['future'] = {}
        key = key.detach().clone().transpose(0, 1)
        value = value.detach().clone().transpose(0, 1)
        word = word
        if len(index.tolist()) == 0:
            return 0

        key = key.unbind()
        value = value.unbind()
        word = word.unbind()
        index = index.unbind()
        select_mask = select_mask.unbind()
        assert len(word) == len(key)
        assert len(word) == len(value)
        for idx, se, w, k, v in zip(index, select_mask, word, key, value):
            idx = idx.item()
            se = se.eq(0).nonzeros().long().view(-1)
            if se.tolist()[0] > w.size(0):
                print(se)
                print(w.size())
            w = w.index_select(0, se).unbind()
            k = k.index_select(0, se).unbind()
            v = v.index_select(0, se).unbind()

            if idx not in cache['future'].keys():
                cache['future'][idx] = {}
            for wb, kb, vb in zip(w, k, v):
                if wb.item() <= 5:
                    continue
                if wb.item() in cache['future'][idx].keys():
                    ks, vs = cache['future'][idx][wb.item()]
                    cache['future'][idx][w.item()] = ((ks + kb) / 2, (vs + vb) / 2)
                else:
                    cache['future'][idx][w.item()] = (kb, vb)

    def update_future_eval(self, cache, index, word, key, value, select_mask):
        cache.pop('future')
        cache['future'] = {}
        key = key.detach().clone().transpose(0, 1)
        value = value.detach().clone().transpose(0, 1)
        word = word
        if len(index.tolist()) == 0:
            return 0

        key = key.unbind()
        value = value.unbind()
        word = word.unbind()
        index = index.unbind()

        assert len(word) == len(key)
        assert len(word) == len(value)
        for idx, w, k, v in zip(index, word, key, value):
            idx = idx.item()
            k = k.unbind()
            v = v.unbind()
            w = w.unbind()
            if idx not in select_mask.keys():
                continue
            dict_keys = set(select_mask[idx].view(-1).tolist())

            if idx not in cache['future'].keys():
                cache['future'][idx] = {}
            for wb, kb, vb in zip(w, k, v):
                if wb.item() <= 5:
                    continue
                if wb.item() in dict_keys:
                    if wb.item() in cache['future'][idx].keys():
                        ks, vs = cache['future'][idx][wb.item()]
                        cache['future'][idx][wb.item()] = ((ks + kb) / 2, (vs + vb) / 2)
                    else:
                        cache['future'][idx][wb.item()] = (kb, vb)

    def read(self, cache, idx, ctx):
        new_key = []
        new_value = []
        word = []
        idx = idx.unbind()

        for n, ids in enumerate(idx):
            ids = ids.item()
            tmp_k = []
            tmp_v = []
            tmp_w = []
            if ids in cache['now'].keys():

                for k, v in cache['now'][ids].items():
                    tmp_k.append(v[0].unsqueeze(0))
                    tmp_v.append(v[1].unsqueeze(0))
                    tmp_w.append(k)
                tmp_k = torch.cat(tmp_k, dim=0)
                tmp_v = torch.cat(tmp_v, dim=0)
            else:
                tmp_k = ctx
                tmp_v = ctx
                tmp_word = [1]
            tmp_w = torch.tensor(tmp_w, dtype=torch.int)
            new_key.append(tmp_k)
            new_key.append(tmp_v)
            word.append(tmp_w)
        return torch.nn.utils.rnn.pad_sequence(new_key, batch_first=True), torch.nn.utils.rnn.pad_sequence(new_value,
                                                                                                           batch_first=True), torch.nn.utils.rnn.pad_sequence(
            word, batch_first=True, padding_value=0)

    def get_idx(self, cout):
        argmax_ = torch.argmax(cout, dim=-1).bool()
        argmax_ = argmax_.flatten().nonzero().view(-1)
        return argmax_

    def update(self, cache, cache_size):
        if 'future' not in cache.keys() or len(cache['future']) == 0:
            return 0
        for idx, data in cache['future'].items():
            if idx not in cache['now'].keys() and len(cache['future'][idx]) != 0:
                cache['now'][idx] = {}
            for word, v in data.items():
                key, value = v[0], v[1]
                cache_keys = cache['now'][idx].keys()
                if word in cache_keys:
                    cache['now'][idx][word] = [(key + cache['now'][idx][word][0]) / 2,
                                               (value + cache['now'][idx][word][0])]
                elif len(cache['now']) < cache_size:
                    cache['now'][idx][word] = [key, value]
                else:
                    cache_keys = list(cache_keys)
                    cache['now'][idx].pop(cache_keys[0])
                    cache['now'][idx][word] = [key, value]
        cache['future'].clear()
