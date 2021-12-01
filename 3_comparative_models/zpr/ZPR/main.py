import os, sys, json, codecs
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn
from tqdm import tqdm, trange

import zp_datastream

from transformers import BertTokenizer,get_linear_schedule_with_warmup
from transformers.optimization import AdamW

import utils

FLAGS = None


def calc_f1(n_out, n_ref, n_both):
    # print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both / n_out if n_out > 0.0 else 0.0
    rc = n_both / n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0 * pr * rc / (pr + rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


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







# def add_counts_bow(out, ref, counts):
#    if sum(out) != 0:
#        counts[1] += out[1] - out[0] + 1
#    if sum(ref) != 0:
#        counts[2] += ref[1] - ref[0] + 1
#        ins_st, ins_ed = max(out[0],ref[0]), min(out[1],ref[1])
#        counts[0] += max(ins_ed - ins_st + 1, 0)


def dev_eval(model, model_type, development_sets, device, log_file):
    model.eval()
    dev_eval_results = []
    for devset in development_sets:
        data_type = devset['data_type']
        batches = devset['batches']
        assert data_type in ('recovery')
        print('Evaluating on dataset with data_type: {}'.format(data_type))
        N = 0
        dev_loss = {'total_loss': 0.0, 'detection_loss': 0.0, 'recovery_loss': 0.0}
        dev_counts = {'detection': [0.0 for _ in range(3)], 'recovery': [0.0 for _ in range(3)],
                      }
        start = time.time()
        for step, ori_batch in enumerate(batches):
            # execution
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                     for k, v in ori_batch.items()}
            step_loss, step_out = forward_step(model, batch)
            # record loss
            for k, v in step_loss.items():
                dev_loss[k] += v.item() if type(v) == torch.Tensor else v
            # generate outputs
            input_zp, input_zp_cid = \
                batch['input_zp'], batch['input_zp_cid']
            input_zp = input_zp.cpu().tolist()
            detection_out = step_out['detection_outputs'].cpu().tolist()
            if data_type == 'recovery':
                input_zp_cid = input_zp_cid.cpu().tolist()
                recovery_out = step_out['recovery_outputs'].cpu().tolist()
        
    
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
                        add_counts(out=recovery_out[i][j], ref=input_zp_cid[i][j],
                                   counts=dev_counts['recovery'])
                N += B
        # output and calculate performance
        total_loss = dev_loss['total_loss']
        duration = time.time() - start
        print('Loss: %.2f, time: %.3f sec' % (total_loss, duration))
        log_file.write('Loss: %.2f, time: %.3f sec\n' % (total_loss, duration))
        det_pr, det_rc, det_f1 = calc_f1(n_out=dev_counts['detection'][1],
                                         n_ref=dev_counts['detection'][2], n_both=dev_counts['detection'][0])
        print('Detection F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*det_f1, 100*det_pr, 100*det_rc))
        log_file.write(
            'Detection F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100 * det_f1, 100 * det_pr, 100 * det_rc))
        cur_result = {'data_type': data_type, 'loss': total_loss, 'detection_f1': det_f1}
        if data_type == 'recovery':
            rec_pr, rec_rc, rec_f1 = calc_f1(n_out=dev_counts['recovery'][1],
                                             n_ref=dev_counts['recovery'][2], n_both=dev_counts['recovery'][0])
            print('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f' % (100 * rec_f1, 100 * rec_pr, 100 * rec_rc))
            log_file.write(
                'Recovery F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100 * rec_f1, 100 * rec_pr, 100 * rec_rc))
            cur_result['key_f1'] = rec_f1
        if len(development_sets) > 1:
            print('+++++')
            log_file.write('+++++\n')
        log_file.flush()
        dev_eval_results.append(cur_result)
    model.train()
    return dev_eval_results


def forward_step(model, batch):

    input_ids, input_mask, input_decision_mask = \
            batch['input_ids'], batch['input_mask'], batch['input_decision_mask']
    input_zp, input_zp_cid, batch_type = \
            batch['input_zp'], batch['input_zp_cid'], batch['type']

    loss, outputs = model(input_ids, input_mask, input_decision_mask,
                              input_zp, input_zp_cid, batch_type)

    return loss, outputs




def main():
    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/ZP.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(str(FLAGS)))
    log_file.flush()
    utils.save_config(FLAGS, path_prefix + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))
    log_file.write('device: {}, n_gpu: {}, grad_accum_steps: {}\n'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if FLAGS.pretrained_path is not None:
        print('loading tokenizer from pretraining')
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))
    log_file.write('Number of predefined pronouns: {}, they are: {}\n'.format(len(pro_mapping), pro_mapping.values()))

    # ZP setting

    # load data and make_batches
    print('Loading data and making batches')
    train_instance_number = 0
    train_batches = []
    train_range_ends = []
    for path, data_type in zip(FLAGS.train_path, FLAGS.train_type):
        features = zp_datastream.load_and_extract_features(path, tokenizer,
                                                           char2word=FLAGS.char2word, data_type=data_type,
                                                           )
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                                           is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

        train_instance_number += len(features)
        train_batches.extend(batches)
        train_range_ends.append(len(train_batches))

    devsets = []
    for path, data_type in zip(FLAGS.dev_path, FLAGS.dev_type):
        features = zp_datastream.load_and_extract_features(path, tokenizer,
                                                           char2word=FLAGS.char2word, data_type=data_type)
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                                           is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
        devsets.append({'data_type': data_type, 'batches': batches})

    testsets = []
    for path, data_type in zip(FLAGS.test_path, FLAGS.test_type):
        features = zp_datastream.load_and_extract_features(path, tokenizer,
                                                           char2word=FLAGS.char2word, data_type=data_type)
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                                           is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
        testsets.append({'data_type': data_type, 'batches': batches})

    print("Num training examples = {}".format(train_instance_number))
    print("Num training batches = {}".format(len(train_batches)))
    print("Data option: is_shuffle {}, is_sort {}, is_batch_mix {}".format(FLAGS.is_shuffle,
                                                                           FLAGS.is_sort, FLAGS.is_batch_mix))

    # create model
    print('Compiling model')
    model = zp_model.BertZP.from_pretrained(FLAGS.pretrained_path, char2word=FLAGS.char2word,
                                            pro_num=len(pro_mapping))

    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    print('Starting the training loop, ', end="")
    train_steps = len(train_batches) * FLAGS.num_epochs
    if FLAGS.grad_accum_steps > 1:
        train_steps = train_steps // FLAGS.grad_accum_steps
    print("total steps = {}".format(train_steps))

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(grouped_params,
                         lr=FLAGS.learning_rate)
    scheduler  = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=4000, num_training_steps=train_steps)


    best_f1 = 0.0
    finished_steps, finished_epochs = 0, 0
    rates = {'detection_discount': 0.1, 'recovery': 8e-6}
    model.train()
    while finished_steps < train_steps:
        epoch_start = time.time()
        train_loss = {'total_loss': 0.0, 'detection_loss': 0.0, 'recovery_loss': 0.0,}
        train_batch_ids = list(range(0, train_range_ends[-1]))
        print('Current epoch takes {} steps'.format(len(train_batch_ids)))
        if FLAGS.is_batch_mix:
            random.shuffle(train_batch_ids)
        for id in train_batch_ids:
            ori_batch = train_batches[id]
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                     for k, v in ori_batch.items()}

            step_loss, _ = forward_step(model, batch)
            for k, v in step_loss.items():
                train_loss[k] += v.item() if type(v) == torch.Tensor else v

            # modify the loss type
            step_loss['total_loss'] = rates['detection_discount'] * step_loss['detection_loss'] + step_loss[
                '%s_loss' % batch['type']]
            loss = step_loss['total_loss']

            if n_gpu > 1:
                loss = loss.mean()
            if FLAGS.grad_accum_steps > 1:
                loss = loss / FLAGS.grad_accum_steps
            loss.backward()  # just calculate gradient

            # adapt lr for batches of different types
            for param_group in optimizer.param_groups:
                param_group['lr'] = rates[batch['type']]

            if finished_steps % FLAGS.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            finished_steps += 1
            if finished_steps % 100 == 0:
                print('{} '.format(finished_steps), end="")
                sys.stdout.flush()

        duration = time.time() - epoch_start
        print('\nTraining loss: %s, time: %.3f sec' % (str(train_loss), duration))
        log_file.write('\nTraining loss: %s, time: %.3f sec\n' % (str(train_loss), duration))
        cur_f1 = []
        for dev_result in dev_eval(model, FLAGS.model_type, [devsets[0]], device, log_file):
            if dev_result['data_type'] in FLAGS.dev_key_types:
                cur_f1.append(dev_result['key_f1'])
        cur_f1 = np.mean(cur_f1)
        if cur_f1 > best_f1:
            print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
            log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
            best_f1 = cur_f1
            save_model(model, path_prefix)
            FLAGS.best_number = best_f1
            utils.save_config(FLAGS, path_prefix + ".config.json")
        print('-------------')
        log_file.write('-------------\n')
#         for i,testset in enumerate(testsets):
#             print('test_set_{}'.format(i))
        dev_eval(model, FLAGS.model_type, [testsets[0]], device, log_file)
        print('=============')
        log_file.write('=============\n')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finished_epochs += 1


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_bin_path = path_prefix + ".bert_model.bin"
    model_config_path = path_prefix + ".bert_config.json"

    torch.save(model_to_save.state_dict(), model_bin_path)
    model_to_save.config.to_json_file(model_config_path)


def check_config(FLAGS):
    assert type(FLAGS.grad_accum_steps) == int and FLAGS.grad_accum_steps >= 1
    assert hasattr(FLAGS, "cuda_device")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.config_path is not None:
        print('Loading hyperparameters from ' + FLAGS.config_path)
        FLAGS = utils.load_config(FLAGS.config_path)
    check_config(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    
    import zp_datastream
    if not FLAGS.one_model:
        print('Runing the model from Song')
        import zp_model
    else:
        print('Runing the one-model')
        import zp_model_detect as zp_model

    main()

