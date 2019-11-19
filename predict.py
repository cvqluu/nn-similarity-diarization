import argparse
import configparser
import glob
import os
import shutil
import time
from collections import OrderedDict
from pprint import pprint

import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io import (collate_sim_matrices, dloader, load_n_col,
                     sim_matrix_target)
from models import LSTMSimilarity, LSTMSimilarityCosWS, LSTMSimilarityCosRes


def parse_args():
    parser = argparse.ArgumentParser(description='Predict on datasets for final model')
    parser.add_argument('--cfg', type=str, default='./configs/example.cfg')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='simply takes the cosine sim matrix - used only for basebaseline')
    args = parser.parse_args()
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.data_path = config['Datasets']['data_path']

    args.model_type = config['Model'].get('model_type', fallback='lstm')
    assert args.model_type in ['lstm', 'lstm_cos_ws', 'lstm_cos_res', 'transformer', 'convcosres']

    args.num_epochs = config['Hyperparams'].getint('num_epochs', fallback=100)
    args.max_len = config['Hyperparams'].getint('max_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)

    args.base_model_dir = config['Outputs']['base_model_dir']
    pprint(vars(args))
    return args

def predict_matrices(model, dl_test):
    # for a model and dataset, make predictions and collate them by recording
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

def predict_seq_matrices(model, dl_test):
    preds = []
    rids = []
    with torch.no_grad():
        for batch_idx, (feats, _, rec_id) in enumerate(dl_test.get_batches_seq()):
            feats = torch.FloatTensor(feats).unsqueeze(1).to(device)
            out = model(feats)
            preds.append(out.detach().cpu().numpy())
            rids.append(rec_id)
    return preds, rids

def cosine_sim_matrix(dl_test):
    preds = []
    rids = []
    for feats, _, rec_id in tqdm(dl_test.get_batches_seq()):
        preds.append(pairwise_distances(feats, metric='cosine'))
        rids.append(rec_id)
    return preds, rids


if __name__ == "__main__":
    '''
    This makes predictions for all train and test recordings for each fold in base_model_dir
    The predictions are stored as .npy files in base_model_dir/ch*/<tr|te>_preds
    '''
    args = parse_args()
    assert os.path.isfile(args.cfg)
    args = parse_config(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print('='*30)
    print('USE_CUDA SET TO: {}'.format(use_cuda))
    print('CUDA AVAILABLE?: {}'.format(torch.cuda.is_available()))
    print('='*30)

    device = torch.device("cuda" if use_cuda else "cpu")

    folds_models = glob.glob(os.path.join(args.base_model_dir, 'ch*'))
    for fold in range(len(folds_models)):
        model_dir = os.path.join(args.base_model_dir, 'ch{}'.format(fold))
        model_path = os.path.join(model_dir, 'final_{}.pt'.format(args.num_epochs))
        base_path = os.path.join(args.data_path, 'ch{}'.format(fold))

        dl_train = dloader(os.path.join(base_path, 'train'), max_len=args.max_len, shuffle=False)
        dl_test = dloader(os.path.join(base_path, 'test'), max_len=args.max_len, shuffle=False)

        if args.cosine:
            te_cm, te_cids = cosine_sim_matrix(dl_test)
            tr_cm, tr_cids = cosine_sim_matrix(dl_train)
        else:
            if args.model_type == 'lstm':
                model = LSTMSimilarity()
                predfunc = predict_matrices
            if args.model_type == 'lstm_cos_res':
                model = LSTMSimilarityCosRes()
                predfunc = predict_matrices
            if args.model_type == 'lstm_cos_ws':
                model = LSTMSimilarityCosWS()
                predfunc = predict_matrices
            if args.model_type == 'convcosres':
                model = ConvCosResSim()
                predfunc = predict_matrices
            if args.model_type == 'transformer':
                assert NotImplementedError

            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            print('Predicting test recordings for fold {}...'.format(fold))
            te_cm, te_cids = predfunc(model, dl_test)
            print('Predicting train recordings for fold {} ...'.format(fold))
            tr_cm, tr_cids = predfunc(model, dl_train)

        
        tr_mat_dir = os.path.join(model_dir, 'tr_preds')
        os.makedirs(tr_mat_dir, exist_ok=True)
        te_mat_dir = os.path.join(model_dir, 'te_preds')
        os.makedirs(te_mat_dir, exist_ok=True)

        for mat, rid in zip(tr_cm, tr_cids):
            filename = os.path.join(tr_mat_dir, rid+'.npy')
            np.save(filename, mat)
        
        for mat, rid in zip(te_cm, te_cids):
            filename = os.path.join(te_mat_dir, rid+'.npy')
            np.save(filename, mat)
