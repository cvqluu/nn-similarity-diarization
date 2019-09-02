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

def sym(matrix):
    '''
    Symmeterization: Y_{i,j} = max(S_{ij}, S_{ji})
    '''
    return np.maximum(matrix, matrix.T)

def diffusion(matrix):
    '''
    Diffusion: Y <- YY^T
    '''
    return np.dot(matrix, matrix.T)

def row_max_norm(matrix):
    '''
    Row-wise max normalization: S_{ij} = Y_{ij} / max_k(Y_{ik})
    '''
    maxes = np.amax(matrix, axis=0)
    return matrix/maxes

def sim_enhancement(matrix):
    return row_max_norm(diffusion(sym(matrix)))

def spectral_clustering(S, beta=1e-2):
    S = sim_enhancement(S)
    np.fill_diagonal(S, 0.)
    L_norm = laplacian(S, normed=True)
    eigvals, eigvecs = np.linalg.eig(L_norm)
    kmask = np.real(eigvals) < beta
    P = np.real(eigvecs).T[kmask].T
    km = KMeans(n_clusters=P.shape[1])
    return km.fit_predict(P) + 1

def assign_segments(pred_labels, events):
    entries = []
    for plabel, ev in zip(pred_labels, events):
        start = ev[0]
        end = ev[1]
        if not entries:
            entries.append({'s':start, 'e':end, 'id':plabel})
        else:
            if entries[-1]['e' ] < start:
                entries.append({'s':start, 'e':end, 'id':plabel})
                continue
            else:
                if entries[-1]['id'] == plabel:
                    entries[-1]['e'] = end
                    continue
                else:
                    # take average of both to determine boundary
                    fuzzy_start = (entries[-1]['e'] + start)/2.
                    entries[-1]['e'] = fuzzy_start
                    entries.append({'s':fuzzy_start, 'e':end, 'id':plabel})
                    continue
    return entries

def rttm_lines_from_entries(entries, rec_id):
    lines = []
    for entry in entries:
        start = entry['s']
        end = entry['e']
        label = entry['id']
        offset = end-start
        line = 'SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(rec_id, start, offset, label)
        lines.append(line)
    return lines

def lines_to_file(lines, filename, wmode="w+"):
    with open(filename, wmode) as fp:
        for line in lines:
            fp.write(line)

def make_rttm(segments, cids, cm, rttm_file, beta=1e-2):
    if os.path.isfile(rttm_file):
        os.remove(rttm_file)
    segment_cols = load_n_col(segments, numpy=True)

    seg_recording_ids = sorted(set(segment_cols[1]))
    assert len(seg_recording_ids) == len(cids)

    events0 = np.array(segment_cols[2:4]).astype(float).transpose()

    for rec_id, smatrix in zip(cids, cm):
        seg_indexes = segment_cols[1] == rec_id
        ev0 = events0[seg_indexes]
        assert len(smatrix) == len(ev0)
        pred_labels = spectral_clustering(smatrix, beta=beta)
        entries = assign_segments(pred_labels, ev0)
        lines = rttm_lines_from_entries(entries, rec_id)
        lines_to_file(lines, rttm_file, wmode='a')
        print('Wrote {} predictions to rttm {}'.format(rec_id, rttm_file))

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
    betavals = np.linspace(1, 2, 11)
    for beta in betavals:
        rttmdir = './exp/beta_{}'.format(beta)
        os.makedirs(rttmdir, exist_ok=True)
        rttm_path = os.path.join(rttmdir, 'hyp_{}.rttm'.format(args.fold))
        make_rttm(te_segs, cids, cm, rttm_path, beta=beta)



