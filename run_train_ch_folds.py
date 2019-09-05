import subprocess

if __name__ == '__main__':
    train_script = 'train_xtransformer_sim.py'
    for fold in range(1,5):
        cmd = 'python {} --fold {}'.format(train_script, fold)
        subprocess.call(cmd, shell=True)