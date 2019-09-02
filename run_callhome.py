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
        cmd = 'python diarize.py --fold {}'.format(fold)
        subprocess.call(cmd, shell=True)
    paths = glob.glob('./exp/beta_*/')
    for path in paths:
        rttms = glob.glob(os.path.join(path, 'hyp_*.rttm'))
        full_rttm = os.path.join(path, 'full_hyp.rttm')
        final_lines = sort_and_cat(rttms)
        with open(full_rttm, 'w') as fp:
            for line in final_lines:
                fp.write(line)