import subprocess

if __name__ == '__main__':
    for fold in range(1,5):
        cmd = 'python train_lstmres.py --fold {}'.format(fold)
        subprocess.call(cmd, shell=True)