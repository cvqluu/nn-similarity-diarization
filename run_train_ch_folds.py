import subprocess
import sys

if __name__ == '__main__':
    cfg = sys.argv[1]
    train_script = 'train.py'
    
    for fold in range(3,5):
        cmd = 'python {} --cfg {} --fold {}'.format(train_script, cfg, fold)
        subprocess.call(cmd, shell=True)
