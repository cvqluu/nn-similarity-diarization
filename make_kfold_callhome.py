import os
import sys

from data_io import segment_labels, make_k_fold_dataset

if __name__ == "__main__":
    xvector_dir = sys.argv[1]
    rttm = sys.argv[2]
    base_path = sys.argv[3]

    segments = os.path.join(xvector_dir, 'segments')
    xvectorscp = os.path.join(xvector_dir, 'xvector.scp')

    rec_ids, rec_batches = segment_labels(segments, rttm, xvectorscp)
    make_k_fold_dataset(rec_ids, rec_batches, base_path, k=5)