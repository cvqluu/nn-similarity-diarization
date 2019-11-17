import argparse
import configparser
import glob
import os
import shutil
import subprocess
import time
import re
from collections import OrderedDict
from pprint import pprint

import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_io import (collate_sim_matrices, dloader, load_n_col,
                     sim_matrix_target)
from models import LSTMSimilarity

def parse_args():
    parser = argparse.ArgumentParser(description='Find best cluster threshold on train folds and use on test')
    parser.add_argument('--cfg', type=str, default='./configs/example.cfg')
    args = parser.parse_args()
    pprint(vars(args))
    return args

def parse_config(args):
    config = configparser.ConfigParser()
    config.read(args.cfg)

    args.data_path = config['Datasets']['data_path']

    args.model_type = config['Model'].get('model_type', fallback='lstm')
    assert args.model_type in ['lstm', 'transformer']

    args.num_epochs = config['Hyperparams'].getint('num_epochs', fallback=100)
    args.max_len = config['Hyperparams'].getint('max_len', fallback=400)
    args.no_cuda = config['Hyperparams'].getboolean('no_cuda', fallback=False)

    args.base_model_dir = config['Outputs']['base_model_dir']

    args.cluster_type = config['Clustering'].get('cluster_type', fallback='sc')
    assert args.cluster_type in ['sc', 'ahc']

    args.cparam_start = config['Clustering'].getfloat('cparam_start', fallback=0.0)
    args.cparam_end = config['Clustering'].getfloat('cparam_end', fallback=1.0)
    args.cparam_steps = config['Clustering'].getint('cparam_steps', fallback=20)
    return args

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

def agg_clustering(S, thresh=0.):
    S = sim_enhancement(S)
    ahc = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', compute_full_tree=True, distance_threshold=thresh)
    return ahc.fit_predict(S)

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

def make_rttm(segments, cids, cm, rttm_file, ctype='sc', cparam=1e-2):
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
        if ctype == 'sc':
            pred_labels = spectral_clustering(smatrix, beta=cparam)
        if ctype == 'ahc':
            pred_labels = agg_clustering(smatrix, thresh=cparam)    
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

def score_der(hyp=None, ref=None, outfile=None, collar=0.25):
    '''
    Takes in hypothesis rttm and reference rttm and returns the diarization error rate
    Calls md-eval.pl -> writes output to file -> greps for DER value
    '''
    cmd = './md-eval.pl -1 -c {} -s {} -r {} > {}'.format(collar, hyp, ref, outfile)
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(outfile, 'r') as file:
        data = file.read().replace('\n', '')
    der_str = re.search('DIARIZATION\ ERROR\ =\ [0-9]+([.][0-9]+)?', data).group()
    der = float(der_str.split()[-1])
    return der


if __name__ == "__main__":
    args = parse_args()
    assert os.path.isfile(args.cfg)
    args = parse_config(args)

    folds_models = glob.glob(os.path.join(args.base_model_dir, 'ch*'))
    for fold in range(len(folds_models)):
        # read train .npy files
        tr_mat_dir = os.path.join(args.base_model_dir, 'ch{}/tr_preds'.format(fold))
        tr_npys = glob.glob(os.path.join(tr_mat_dir, '*.npy'))
        tr_recs = [os.path.basename(i)[:-4] for i in tr_npys]
        tr_mats = [np.load(i) for i in tr_npys]
        tr_segs = os.path.join(args.data_path, 'ch{}/train/segments'.format(fold))
        tr_rttm = os.path.join(args.data_path, 'ch{}/train/ref.rttm'.format(fold))

        # perform clustering, across chosen cluster thresholds
        tuning_dir = os.path.join(args.base_model_dir, 'ch{}/tuning'.format(fold))
        os.makedirs(tuning_dir, exist_ok=True)

        cparam_range = np.linspace(args.cparam_start, args.cparam_end, args.cparam_steps)
        best_der = (100, -1) #store best der
        for i, cparam in enumerate(tqdm(cparam_range)):
            rttm_outfile = os.path.join(tuning_dir, '{}.rttm'.format(i))
            make_rttm(tr_segs, tr_recs, tr_mats, rttm_outfile, ctype=args.cluster_type, cparam=cparam)

            #Calc der
            eval_log = os.path.join(tuning_dir, '{}.derlog'.format(i))
            der = score_der(hyp=rttm_outfile, ref=tr_rttm, outfile=eval_log)
            print('Fold {}, cparam {} \t DER: {}'.format(fold, cparam, der))
            if der < best_der[0]:
                best_der = (der, i)

        best_thresh = cparam_range[best_der[1]]

        te_mat_dir = os.path.join(args.base_model_dir, 'ch{}/te_preds'.format(fold))
        te_npys = glob.glob(os.path.join(te_mat_dir, '*.npy'))
        te_recs = [os.path.basename(i)[:-4] for i in te_npys]
        te_mats = [np.load(i) for i in te_npys]
        te_segs = os.path.join(args.data_path, 'ch{}/test/segments'.format(fold))
        
        test_rttm_outfile = os.path.join(tuning_dir, 'test.rttm')
        make_rttm(te_segs, te_recs, te_mats, test_rttm_outfile, ctype=args.cluster_type, cparam=best_thresh)

    # concatenate test rttm files
    te_rttms = [os.path.join(args.base_model_dir, 'ch{}/tuning/test.rttm'.format(fold)) for fold in range(len(folds_models))]
    ftest_rttm = os.path.join(args.base_model_dir, 'fulltest.rttm')
    cat_cmd = "cat {} > {}".format(' '.join(te_rttms), ftest_rttm)
    subprocess.call(cat_cmd, shell=True)

    fullref_rttm = os.path.join(args.base_data_path, 'fullref.rttm')
    test_der = score_der(hyp=ftest_rttm, ref=fullref_rttm)
    print('Full Test Der: {}'.format(test_der))

