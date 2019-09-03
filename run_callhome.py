import subprocess
import glob
import fileinput
import os
import numpy as np


def sort_and_cat(rttms):
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
    rec_ids = list(sorted(set(columns[1])))
    final_lines = []
    for rid in rec_ids:
        rindexes = columns[1] == rid
        final_lines += list(all_rows[rindexes])
    return final_lines


if __name__ == '__main__':
    for fold in range(5):
        cmd = 'python predict.py --fold {}'.format(fold)
        subprocess.call(cmd, shell=True)

    # segfiles = ['/disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/data/ch{}/test/segments'.format(fold) for fold in range(5)]
    # all_seglines = sort_and_cat(segfiles)
    # with open('./exp/ch_segments', 'w+') as fp:
    #     for line in all_seglines:
    #         fp.write(line)
    
    cmd = 'python cluster.py'
    subprocess.call(cmd, shell=True)