import argparse
import glob
import os
import shutil
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io import dloader, sim_matrix_target, collate_sim_matrices, load_n_col
from models import LSTMSimilarity
from tqdm import tqdm
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans, SpectralClustering


def parse_args():
    parser = argparse.ArgumentParser(description='Extract and diarize')
    parser.add_argument('--model-path', type=str, default='./exp/lstm_sim_ch{}/final_100.pt',
                        help='Saved model paths')
    parser.add_argument('--model-type', type=str, default='lstm',
                        help='Model type')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--max-len', type=int, default=400)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.model_path = args.model_path.format(args.fold)
    pprint(vars(args))
    return args

def predict_matrices(model, dl_test):
    preds = []
    rids = []
    with torch.no_grad():
        for batch_idx, (feats, _, rec_id) in enumerate(tqdm(dl_test.get_batches(), total=len(dl_test))):
            feats = torch.FloatTensor(feats).to(device)
            out = model(feats)
            preds.append(torch.sigmoid(out).detach().cpu().numpy())
            rids.append(rec_id)
    cm, cids = collate_sim_matrices(preds, rids)
    return cm, cids


if __name__ == "__main__":
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('-'*10)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('-'*10)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = LSTMSimilarity()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    
    rttm = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/data/callhome/fullref.rttm'
    xbase = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/exp/xvector_nnet_1a/xvectors_callhome'
    fold = args.fold
    base_path = '/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/data/ch{}/'.format(fold)
    
    te_segs = os.path.join(base_path, 'test/segments')
    te_xvecscp = os.path.join(base_path, 'test/xvector.scp')
    dl_test = dloader(te_segs, rttm, te_xvecscp, max_len=args.max_len, pad_start=False, xvecbase_path=xbase, shuffle=False)

    cm, cids = predict_matrices(model, dl_test)
    mat_dir = './exp/ch_sim_mat'
    os.makedirs(mat_dir, exist_ok=True)
    
    for mat, rid in tqdm(zip(cm, cids)):
        filename = os.path.join(mat_dir, rid+'.npy')
        np.save(filename, mat)
    



