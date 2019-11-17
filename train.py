import argparse
import configparser
import glob
import json
import os
import shutil
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io import dloader
from models import LSTMSimilarity, LSTMSimilarityCosWS, LSTMSimilarityCosRes
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_sequence)


def schedule_lr(optimizer, factor=0.1):
    for params in optimizer.param_groups:
        params['lr'] *= factor
    print(optimizer)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_args():
    parser = argparse.ArgumentParser(description='Train nn embedding similarity scoring')
    parser.add_argument('--cfg', type=str, default='./configs/example.cfg')
    parser.add_argument('--epoch-resume', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg)
    args._start_time = time.ctime()
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.data_path = config['Datasets']['data_path']

    args.model_type = config['Model'].get('model_type', fallback='lstm')
    assert args.model_type in ['lstm', 'lstm_cos_ws', 'lstm_cos_res', 'transformer']

    args.lr = config['Hyperparams'].getfloat('lr', fallback=0.2)
    args.max_len = config['Hyperparams'].getint('max_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)
    args.seed = config['Hyperparams'].getint('seed', fallback=123)
    args.num_epochs = config['Hyperparams'].getint('num_epochs', fallback=100)
    args.scheduler_steps = np.array(json.loads(config.get('Hyperparams', 'scheduler_steps'))).astype(int)
    args.scheduler_lambda = config['Hyperparams'].getfloat('scheduler_lambda', fallback=0.1)

    args.base_model_dir = config['Outputs']['base_model_dir']
    args.checkpoint_interval = config['Outputs'].getint('checkpoint_interval', fallback=1)
    return args


def train():
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('-'*10)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('-'*10)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(comment=args.model_type)

    if args.model_type == 'lstm':
        model = LSTMSimilarity()
    if args.model_type == 'lstm_cos_res':
        model = LSTMSimilarityCosRes()
    if args.model_type == 'lstm_cos_ws':
        model = LSTMSimilarityCosWS()
    if args.model_type == 'transformer':
        assert NotImplementedError

    model.to(device)
    model.train()

    if args.epoch_resume:
        model_epoch_filename = os.path.join(args.model_dir, 'epoch_{}.pt'.format(args.epoch_resume))
        print('Resuming training from: {}'.format(model_epoch_filename))
        model.load_state_dict(torch.load(model_epoch_filename))

    optimizer = torch.optim.SGD([{'params': model.parameters()}], 
                                        lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    iterations = 0

    for epoch in range(args.num_epochs):
        total_loss = 0

        if epoch + 1 in args.scheduler_steps:
            schedule_lr(optimizer, factor=args.scheduler_lambda)
            pass

        if args.epoch_resume:
            if epoch + 1 <= args.epoch_resume:
                iterations += len(dl)
                print('Skipped epoch {}'.format(epoch+1))
                continue

        for batch_idx, (feats, labels, _) in enumerate(dl.get_batches()):
            iterations += 1

            feats = torch.FloatTensor(feats).to(device)
            labels = torch.FloatTensor(labels).to(device)

            out = model(feats)
            
            loss = criterion(out.flatten(), labels.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()

            if batch_idx % 5 == 0:
                msg = "{}\tEpoch:{}[{}/{}], Loss:{:.4f} TLoss:{:.4f}, ({})".format(time.ctime(), epoch+1,
                    batch_idx+1, len(dl), loss.item(), total_loss / (batch_idx + 1), feats.shape)
                print(msg)

            writer.add_scalar('loss', loss.item(), iterations)
            writer.add_scalar('Avg loss', total_loss / (batch_idx + 1), iterations)

        if (epoch + 1) % args.checkpoint_interval == 0:
            model.eval().cpu()
            cp_filename = "epoch_{}.pt".format(epoch+1)
            cp_model_path = os.path.join(args.model_dir, cp_filename)
            torch.save(model.state_dict(), cp_model_path)
            model.to(device).train()
            test_loss = test(model, device, criterion)
            print('TEST LOSS: {}'.format(test_loss))
            model.train()
        
    # ---- Final model saving -----
    model.eval().cpu()
    final_model_filename = "final_{}.pt".format(epoch+1)
    final_model_path = os.path.join(args.model_dir, final_model_filename)
    torch.save(model.state_dict(), final_model_path)
    
    print('Training complete. Saved to {}'.format(final_model_path))

def test(model, device, criterion):
    model.eval()
    with torch.no_grad():
        total_batches = 0
        total_loss = 0
        for batch_idx, (feats, labels, _) in enumerate(dl_test.get_batches()):
            feats = torch.FloatTensor(feats).to(device)
            labels = torch.FloatTensor(labels).to(device)
            out = model(feats)
            loss = criterion(out.flatten(), labels.flatten())
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss/total_batches
    
if __name__ == "__main__":
    args = parse_args()
    assert os.path.isfile(args.cfg)
    args = parse_config(args)

    os.makedirs(args.base_model_dir, exist_ok=True)
    args.model_dir = os.path.join(args.base_model_dir, 'ch{}'.format(args.fold))
    args.log_file = os.path.join(args.model_dir, 'exp_out.log')
    os.makedirs(args.model_dir)

    base_path = os.path.join(args.data_path, 'ch{}'.format(args.fold))
    assert os.path.isdir(base_path)

    dl = dloader(os.path.join(base_path, 'train'), max_len=args.max_len)
    dl_test = dloader(os.path.join(base_path, 'test'), max_len=args.max_len, shuffle=False)

    train()
