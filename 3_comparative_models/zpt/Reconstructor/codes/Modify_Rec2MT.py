import torch
import argparse
from typing import AnyStr
from argparse import Namespace

def check_rec_or_not(args:Namespace):
    return args.task == 'recostor'

def load_checkpoint(filename:AnyStr):
    checkpoint = torch.load(filename)
    assert check_rec_or_not(checkpoint['args']) is True, 'Only for recostor structure'
    # change the task
    checkpoint['args'].task = 'translation'
    # change the arch
    checkpoint['args'].arch = checkpoint['args'].arch.replace('ReCostor','transformer')
    checkpoint['args'].criterion = 'label_smoothed_cross_entropy'
    # change the pretrained_checkpoint
    del checkpoint['args'].pretrained_checkpoint
    # remove the parameters of recostor
    keys = [k for k in checkpoint['model'].keys() if k.startswith('recostor')]
    for k in keys:
        checkpoint['model'].pop(k, None)
    return checkpoint


def main(args):
    checkpoint = load_checkpoint(args.input)
    torch.save(checkpoint,args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',type=str)
    parser.add_argument('-o','--output',type=str)
    args = parser.parse_args()
    main(args)