import torch.nn as nn
import json
def detection_loss(inputs,target,mask):
    loss_func = nn.CrossEntropyLoss(reduction='none')
    bzs,seq,dim = inputs.size()
    inputs = inputs.reshape(-1,dim)
    target = target.reshape(-1)
    loss = loss_func(inputs,target)
    mask=mask.bool()
    loss = loss.reshape(bzs,-1).masked_fill_(~mask,0.0)
    return loss.sum()/mask.sum().item()

def recover_loss(inputs,target,mask):
    loss_func = nn.CrossEntropyLoss(reduction='none')
    bzs, seq, dim = inputs.size()
    inputs = inputs.reshape(-1, dim)
    target = target.reshape(-1)
    loss = loss_func(inputs, target)
    mask = mask.bool()
    loss = loss.reshape(bzs, -1).masked_fill_(~mask, 0.0)
    return loss.sum() / mask.sum().item()

def bin_CEloss(inputs,target,mask):
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    bzs, seq, dim = inputs.size()
    inputs = inputs.reshape(-1, dim)
    target = target.reshape(-1,dim).float()
    loss = loss_func(inputs, target)
    mask = mask.bool()
    loss = loss.reshape(bzs,-1).masked_fill_(~mask,0.0)
    return loss.sum()/mask.sum().item()
 
def nlloss(inputs,target,mask):
    loss_func = nn.NLLLoss(reduction='none')
    bzs, seq, dim = inputs.size()
    inputs = inputs.reshape(-1, dim)
    target = target.reshape(-1)
    loss = loss_func(inputs, target)
    mask = mask.bool()
    loss = loss.reshape(bzs,-1).masked_fill_(~mask,0.0)
    return loss.sum()/mask.sum().item()
    
def div_loss(inputs, target, mask):
    loss_func = nn.KLDivLoss(reduction='none')
    bzs, seq, dim = inputs.size()
    # input = input.reshape(-1, dim)
    # target = target.reshape(-1, dim).float()
    loss = loss_func(inputs, target)
    mask = mask.unsqueeze(-1).bool()
    loss = loss.reshape(bzs, seq,dim).masked_fill_(~mask, 0.0)
    return loss.sum() / mask.sum().item()

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def save_config(FLAGS, out_path):
    FLAGS_dict = vars(FLAGS)
    with open(out_path, 'w') as fp:
        json.dump(FLAGS_dict, fp)

def load_config(in_path):
    with open(in_path, 'r') as fp:
        FLAGS_dict = json.load(fp)
        return Bunch(FLAGS_dict)
