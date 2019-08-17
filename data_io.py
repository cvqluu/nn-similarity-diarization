import os

import kaldi_io
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder

def read_xvec(file):
    return kaldi_io.read_vec_flt(file)

def change_base_paths(files, new_base_path='./'):
    filenames = [os.path.basename(file) for file in files]
    new_filenames = [os.path.join(new_base_path, file) for file in filenames]
    return np.array(new_filenames)

def load_n_col(file, numpy=False):
    data = []
    with open(file) as fp:
        for line in fp:
            data.append(line.strip().split(' '))
    columns = list(zip(*data))
    if numpy:
        columns = [np.array(list(i)) for i in columns]
    else:
        columns = [list(i) for i in columns]
    return columns

def calc_overlap(segment, ref_segment):
    '''
    calculates magnitude of overlap
    
    segment format: [start, end]
    '''
    return max(0.0, min(segment[1], ref_segment[1]) - max(segment[0], ref_segment[0]))

def assign_overlaps(events0, events1, events1_labels):
    events0_labels = []
    for event in events0:
        ols = []
        ols_t = []
        for i, evcheck in enumerate(events1):
            overlap = calc_overlap(event, evcheck)
            if overlap > 0.0:
                ols.append(events1_labels[i])
                ols_t.append(overlap)
        if len(ols) == 1:
            events0_labels.append(ols[0])
        if len(ols) == 0:
            events0_labels.append(None)
        if len(ols) > 1:
            events0_labels.append(ols[np.argmax(ols_t)])
    assert len(events0) == len(events0_labels)
    return events0_labels
        
def segment_labels(segments, rttm, xvectorscp):
    segment_cols = load_n_col(segments,numpy=True)
    rttm_cols = load_n_col(rttm,numpy=True)
    vec_utts, vec_paths = load_n_col(xvectorscp, numpy=True)
    
    assert sum(vec_utts == segment_cols[0]) == len(segment_cols[0])
    vec_paths = change_base_paths(vec_paths, new_base_path=os.path.dirname(xvectorscp))

    rttm_cols.append(rttm_cols[3].astype(float) + rttm_cols[4].astype(float))
    recording_ids = sorted(set(segment_cols[1]))
    events0 = np.array(segment_cols[2:4]).astype(float).transpose()
    events1 = np.vstack([rttm_cols[3].astype(float), rttm_cols[-1]]).transpose()
    
    rec_batches = []
    
    for rec_id in recording_ids:
        seg_indexes = segment_cols[1] == rec_id
        rttm_indexes = rttm_cols[1] == rec_id
        ev0 = events0[seg_indexes]
        ev1 = events1[rttm_indexes]
        ev1_labels = rttm_cols[7][rttm_indexes]
        ev0_labels = assign_overlaps(ev0, ev1, ev1_labels)
        ev0_labels = ['{}_{}'.format(rec_id, l) for l in ev0_labels]
        batch = (segment_cols[0][seg_indexes], ev0_labels, vec_paths[seg_indexes])
        rec_batches.append(batch)
        
    return recording_ids, rec_batches


def pairwise_cat_matrix(xvecs, labels):
    '''
    xvecs: (seq_len, d_xvec)
    labels: (seq_len)
    '''
    matrix = []
    label_matrix = []
    for i in range(len(xvecs)):
        row = []
        label_row = []
        for j in range(len(xvecs)):
            entry = np.concatenate([xvecs[i], xvecs[j]], axis=0)
            entry_label = float(labels[i] == labels[j])
            row.append(entry)
            label_row.append(entry_label)
        matrix.append(np.vstack(row))
        label_matrix.append(label_row)
    return np.array(matrix), np.array(label_matrix)

def sim_matrix_target(labels):
    le = LabelEncoder()
    dist = 1.0 - pairwise_distances(le.fit_transform(labels)[:,np.newaxis], metric='hamming')
    return dist

def batch_matrix(xvecpairs, labels, factor=2):
    remainder = len(labels) % factor
    newlen = len(labels) - remainder
    if remainder != 0:
        xvecpairs = xvecpairs[:-remainder, :-remainder, :]
        labels = labels[:-remainder, :-remainder]
    split_batch = []
    split_batch_labs = []
    for i in range(factor):
        start = i * newlen//factor
        end = (i+1) * newlen//factor
        split_rows = np.split(xvecpairs[:,start:end,:], factor)
        split_labs = np.split(labels[:,start:end], factor)
        split_batch += split_rows
        split_batch_labs += split_labs
    return np.array(split_batch), np.array(split_batch_labs)

class dloader:
    
    def __init__(self, segs, rttm, xvec_scp, max_len=400, concat=True):
        assert os.path.isfile(segs)
        assert os.path.isfile(rttm)
        assert os.path.isfile(xvec_scp)
        self.ids, self.rec_batches = segment_labels(segs, rttm, xvec_scp)
        self.lengths = [len(batch[0]) for batch in self.rec_batches]
        self.first_rec = np.argmax(self.lengths)
        self.max_len = max_len
        self.concat = concat
    
    def __len__(self):
        return len(self.ids)
        
    def get_batches(self):
        rec_order = np.arange(len(self.rec_batches))
        np.random.shuffle(rec_order)
        first_rec = np.argwhere(rec_order == self.first_rec).flatten()
        rec_order[0], rec_order[first_rec] = rec_order[first_rec], rec_order[0]

        for i in rec_order:
            utts, labels, paths = self.rec_batches[i]
            xvecs = np.array([read_xvec(file) for file in paths])
            if self.concat:
                pmatrix, plabels = pairwise_cat_matrix(xvecs, labels)
                if len(labels) <= self.max_len:
                    yield pmatrix, plabels
                else:
                    batched_feats, batched_labels = batch_matrix(pmatrix, plabels)
                    for feats, labels in zip(batched_feats, batched_labels):
                        yield feats, labels
            else:
                labels = sim_matrix_target(labels)
                yield xvecs, labels
