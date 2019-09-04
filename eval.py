import subprocess
from glob import glob

for rttm in glob('./exp/maskbeta*/hyp.rttm'):
    cmd = "perl md-eval.pl -1 -c 0.25 -s {} -r /disk/scratch1/s1786813/kaldi/egs/callhome_diarization/v2/data/callhome/fullref.rttm >> ./der.txt".format(rttm)
    subprocess.Popen(cmd, shell=True)
