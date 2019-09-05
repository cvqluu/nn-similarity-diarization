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
import scipy.cluster.hierarchy as hcluster


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
    return km.fit_predict(P)

# def agg_clustering(vectors):
    # cluster_labels = hcluster.fclusterdata(rec_xvecs, t=args.threshold, criterion='distance', method='average')


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
        line = 'SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(rec_id, start, offset, label)
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
    print(len(seg_recording_ids), len(cids))
    assert len(seg_recording_ids) == len(cids)

    events0 = np.array(segment_cols[2:4]).astype(float).transpose()

    for rec_id, smatrix in tqdm(zip(cids, cm)):
        seg_indexes = segment_cols[1] == rec_id
        ev0 = events0[seg_indexes]
        assert len(smatrix) == len(ev0)
        pred_labels = spectral_clustering(smatrix, beta=beta)
        entries = assign_segments(pred_labels, ev0)
        lines = rttm_lines_from_entries(entries, rec_id)
        lines_to_file(lines, rttm_file, wmode='a')




def sort_and_cat(rttms, column=1):
    data = []
    all_rows = []
    for rttm_file in rttms:
        with open(rttm_file) as fp:
            for line in fp:
                data.append(line.strip().split(' '))
                all_rows.append(line)
    all_rows = np.array(all_rows)
    columns = list(zip(*data))
    columns = [np.array(list(i)) for i in columns]
    rec_ids = list(sorted(set(columns[column])))
    final_lines = []
    for rid in rec_ids:
        rindexes = columns[column] == rid
        final_lines += list(all_rows[rindexes])
    return final_lines


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster')
    parser.add_argument('--mat-dir', type=str, default='./exp/ch_{}_mat',
                        help='Saved model paths')
    parser.add_argument('--model-type', type=str, default='mask',
                        help='Model type')
    args = parser.parse_args()
    assert args.model_type in ['lstm', 'mask', 'lstmres']

    args.mat_dir = args.mat_dir.format(args.model_type)
    pprint(vars(args))
    return args

if __name__ == "__main__":
    args = parse_args()
    te_segs = 'exp/ch_segments'
    # mat_dir = './exp/ch_css_mat'
    mat_dir = args.mat_dir
    cm_npys = os.path.join(mat_dir, '*.npy')
    cids = []
    cm = []
    for mpath in glob.glob(cm_npys):
        base = os.path.basename(mpath)
        rid = os.path.splitext(base)[0]
        cm.append(np.load(mpath))
        cids.append(rid)

    betavals = np.linspace(1., 1.1, 5)
    for beta in tqdm(betavals):
        rttmdir = './exp/{}beta_{}'.format(args.model_type, beta)
        os.makedirs(rttmdir, exist_ok=True)
        rttm_path = os.path.join(rttmdir, 'hyp.rttm')
        make_rttm(te_segs, cids, cm, rttm_path, beta=beta)