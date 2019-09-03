import subprocess

if __name__ == '__main__':
    for fold in range(5):
        cmd = 'python train_xt_mask.py --fold {}'.format(fold)
        subprocess.call(cmd, shell=True)