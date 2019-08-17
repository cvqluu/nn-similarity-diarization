import argparse
import glob
import os
import shutil
import time
from pprint import pprint
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LSTMSimilarity
from tensorboardX import SummaryWriter
from data_io import dloader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

def train():
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('-'*10)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('-'*10)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(comment='bilstm_sim')
    model = LSTMSimilarity()
    model.to(device)
    model.train()

    if args.resume_model_train:
        # pretrained dict
        if args.pretrain:
            print('Resuming training from: {}'.format(args.resume_model_train))
            pretrained_dict = torch.load(args.resume_model_train)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(torch.load(args.resume_model_train))

    if args.epoch_resume:
        model_epoch_filename = os.path.join(args.model_dir, 'epoch_{}.pt'.format(args.epoch_resume))
        print('Resuming training from: {}'.format(model_epoch_filename))
        model.load_state_dict(torch.load(model_epoch_filename))

    
    optimizer = torch.optim.Adam([{'params': model.parameters()}], 
                                        lr=args.lr)
    
    print('Scheduler to halve LR every {} epochs'.format(args.scheduler_period))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_period, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    iterations = 0
    best_loss = (1000, -1)


    for epoch in range(args.epochs):
        total_loss = 0
        scheduler.step()
        # if args.epoch_resume:
        #     if epoch + 1 <= args.epoch_resume:
        #         iterations += len(train_loader)
        #         scheduler.step()
        #         print('Skipped epoch {}'.format(epoch+1))
        #         continue

        for batch_idx, (feats, labels) in enumerate(dl.get_batches()):
            iterations += 1

            feats = torch.FloatTensor(feats).to(device)
            labels = torch.FloatTensor(labels).to(device)

            out = model(feats)
            
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_loss = test(model, device, criterion)
            total_loss += loss.item()
            torch.cuda.empty_cache()

            msg = "{}\tEpoch:{}[{}/{}], Loss:{:.4f} TLoss:{:.4f}".format(time.ctime(), epoch+1,
                    batch_idx+1, len(dl), loss.item(), total_loss / (batch_idx + 1))
            print(msg)
            writer.add_scalar('loss', loss.item(), iterations)
            writer.add_scalar('Avg loss', total_loss / (batch_idx + 1), iterations)
        torch.cuda.empty_cache()
        scheduler.step()
        if (epoch + 1) % args.checkpoint_interval == 0:
            test_loss = test(model, device, criterion)
            if test_loss < best_loss[0]:
                best_loss = (test_loss, epoch+1)
            print('-'*30)
            print('Test Loss at Epoch {}: {}'.format(epoch+1, test_loss))
            print('Best Test Loss: {} ({})'.format(best_loss[0], best_loss[1]))
            print('-'*30)
            model.eval().cpu()
            cp_filename = "epoch_{}.pt".format(epoch+1)
            cp_model_path = os.path.join(args.model_dir, cp_filename)
            torch.save(model.state_dict(), cp_model_path)
            model.to(device).train()
            # remove_old_models()
        
    # ---- Final model saving -----
    model.eval().cpu()
    final_model_filename = "final_{}.pt".format(epoch+1)
    final_model_path = os.path.join(args.model_dir, final_model_filename)
    torch.save(model.state_dict(), final_model_path)
    
    print('Training complete. Saved to {}'.format(final_model_path))
    
    best_model_file = "epoch_{}.pt".format(best_loss[1])
    best_model_file = os.path.join(args.model_dir, best_model_file)
    best_model_newfile = os.path.join(args.model_dir, 'best_model.pt')
    shutil.copy(best_model_file, best_model_newfile)

    print('Copied model {} to best_model.pt, Best loss: {}'.format(best_loss[1], best_loss[0]))


def test(model, device, criterion):
    model.eval().to(device)
    test_loader = DataLoader(ds_test,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True)
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, (labels, toks) in enumerate(tqdm(test_loader)):
            toks = toks.to(device)
            labels = labels.squeeze().to(device)
            out = model(toks).squeeze()
            loss = criterion(out, labels)
            total_loss += loss.item()
    model.train()
    return total_loss/(batch_idx + 1)



def parse_args():
    parser = argparse.ArgumentParser(description='Train a speaker change lstm using pytorch')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-dir', type=str, default='./exp/lstm_sim/',
                        help='Saved model paths')
    parser.add_argument('--model-type', type=str, default='lstm',
                        help='Model type')
    parser.add_argument('--scheduler-period', type=int, default=2,
                        help='Scheduler period (default: 10)')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Number of epochs to run before saving the model to disk for checkpointing (default: 5)') 
    parser.add_argument('--resume-model-train', type=str, default=None,
                        help='Path to a checkpointed model to resume training from (default: None)')
    parser.add_argument('--saved-model-history', type=int, default=5,
                            help='Number of saved models to keep')
    parser.add_argument('--pretrain', action='store_true', default=False,
                            help='Will try and load weights that have been trained on another model')
    parser.add_argument('--epoch-resume', type=int, default=None, help='Resume from a chosen epoch')
    args = parser.parse_args()
    args._start_time = time.ctime()
    args.log_file = os.path.join(args.model_dir, 'exp_out.log')

    pprint(vars(args))
    os.makedirs(args.model_dir, exist_ok=True)
    exp_info = os.path.join(args.model_dir, 'exp_info.log')
    with open(exp_info, 'a+') as fp:
        pprint(vars(args), fp)

    return args

def remove_old_models():
    models = glob.glob(os.path.join(args.model_dir, 'epoch_*.pt'))
    if len(models) > args.saved_model_history:
        num = []
        for model in models:
            _, filename = os.path.split(model)
            examples_seen = int(filename[6:-3])
            num.append(examples_seen)

        num, models = zip(*sorted(zip(num, models)))
        models_to_delete = models[:-args.saved_model_history]
        for model in models_to_delete:
            os.remove(model)
    else:
        pass
    
if __name__ == "__main__":
    args = parse_args()
    rttm = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/data/callhome1/ref.rttm'
    segs = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/exp/xvector_nnet_1a/xvectors_callhome1/segments'
    xvec_scp = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/exp/xvector_nnet_1a/xvectors_callhome1/xvector.scp'
    dl = dloader(segs, rttm, xvec_scp)
    train()
