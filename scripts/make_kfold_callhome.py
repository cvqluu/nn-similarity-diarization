import os
import sys

from data_io import segment_labels, make_k_fold_dataset, make_subset_rttm

FOLDS = 5

if __name__ == "__main__":
    xvector_dir = sys.argv[1]
    rttm = sys.argv[2]
    base_path = sys.argv[3]

    segments = os.path.join(xvector_dir, 'segments')
    xvectorscp = os.path.join(xvector_dir, 'xvector.scp')

    rec_ids, rec_batches = segment_labels(segments, rttm, xvectorscp)
    print('Making folds...')
    make_k_fold_dataset(rec_ids, rec_batches, base_path, k=FOLDS)

    print('Making subset rttms...')
    for fold in range(FOLDS):
        for sub in ['train', 'test']:
            subdir = os.path.join(base_path, 'ch{}/{}'.format(fold, sub))
            sub_rttm = os.path.join(subdir, 'ref.rttm')
            segs = os.path.join(subdir, 'segments')
            make_subset_rttm(rttm, segs, sub_rttm)
