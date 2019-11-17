# Neural network based similarity scoring for diarization

PyTorch implementation of neural network based similarity scoring for diarization: based on the paper **"LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization"** [1] at INTERSPEECH 2019 https://arxiv.org/abs/1907.10393, https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1388.pdf 

I am not affiliated with the paper authors.

The basic concept behind this method is to learn the similarity scoring matrix needed for diarizing a recording. Inputs of concatenated speaker embeddings (such as x-vectors) are fed through an LSTM or other architecture to predict the similarity of the concatenated embeddings.

![model_fig](figures/bilstm_sim_model.png?raw=true "bilstm_model") Figure taken from [1]

## Requirements

Kaldi, python, kaldi_io, scipy, sklearn, torch, CALLHOME dataset

# TL;DR

You can run most of the steps (make train/test folds -> train -> predict -> cluster) with `run.sh`.

**NOTE: The Kaldi Data preparation must be run first, follow those instructions up until 'Make train/test folds' and then `run.sh` can be run from inside the repo folder. Make sure to configure the variables at the top of `run.sh` as well as your configured `.cfg` file (see `configs/example.cfg`):**

```sh
xvector_dir=/PATH/TO/XVECS #path to extracted xvectors, same as run_data_prep.sh
KALDI_PATH=/PATH/TO/KALDI_ROOT # path to kaldi root, neede for finding egs folder
folds_path=/PATH/TO/FOLDS_DATA # path to where the train/test split folds will be stored
cfg_path=/PATH/TO/CFG # path to cfg file, $folds_path is data_path in the cfg
```

# Kaldi Data preparation

The data-preparation for this will involve the following steps:

1. Make kaldi data folder for CALLHOME
2. Feature extraction (MFCCs)
3. X-vector extraction (using the pre-trained CALLHOME model available on the Kaldi website)
4. As in the paper, make a 5 fold train/test split to train and evaluate on

First, some variables need to be configured in `run_data_prep.sh`. These are located at the top of the file and are as follows:

```sh
callhome_path=/PATH/TO/CALLHOME #path to raw callhome data
xvector_dir=/PATH/TO/XVECS #path to extracted xvectors
```

These need to point to where the CALLHOME dataset is and also where you would like the extracted x-vectors to reside (make sure to use an absolute path). Once this is done, copy this script to the Kaldi recipe folder for CALLHOME (as existing data prep scripts are leveraged):

The location of the Kaldi installation will be referred to as `$KALDI_PATH` in the following instructions.

```sh
cp run_data_prep.sh $KALDI_PATH/egs/callhome_diarization/v2
cd $KALDI_PATH/egs/callhome_diarization/v2
source path.sh
./run_data_prep.sh
```

# Make train/test folds

Changing directory back to where this repo is, run the following command to make the train/test folds, replacing the variables as is necessary. Here `$xvector_dir` is as above and `$folds_path` is the location in which the splits will reside: (recommended to use num_folds=5)

```sh
python -m scripts.make_kfold_callhome $xvector_dir $KALDI_PATH/egs/callhome_diarization/v2/data/callhome/fullref.rttm $folds_path $num_folds

cp $KALDI_PATH/egs/callhome_diarization/v2/data/callhome/fullref.rttm $folds_path
```

which makes a folder structure like so

```
folds_path
├── fullref.rttm
├── ch0
|   ├── train
|   |   ├── ref.rttm
|   |   ├── segments
|   |   ├── utt2spk
|   |   └── xvector.scp
|   └── test
├── ch1
|   ├── train
|   └── test
├── ...
```

# Training

The primary training script `train.py` is mostly defined by the config file which it reads. An example config file is shown in `configs/example.cfg`. The relevant fields to this section are shown below:

```ini
[Datasets]
# this is $folds_path in the data preparation step (also in run.sh)
data_path = /PATH/TO/FOLDS_PATH

[Model]
model_type = lstm

[Hyperparams]
lr = 0.01
max_len = 400
no_cuda = False
seed = 1234
num_epochs = 100
# at the epoch numbers in scheduler_steps, the lr will be multiplied by scheduler_lambda
scheduler_steps = [40, 80]
scheduler_lambda = 0.1

[Outputs]
# this is where models will be saved
base_model_dir = exp/example_models_folder
# Interval at which models will be stored for checkpointing purposes
checkpoint_interval = 1
```

The main fields which need to be configured are `data_path` and `base_model_dir`. The first corresponds to `$folds_path` used above and the latter will be the place in which the models are stored.

Once this cfg file is configured, a model can be trained on a fold like so:

```sh
python train.py --cfg configs/<your_config>.cfg --fold 0
```

This will need to be run (in parallel or sequentially) for each fold [0,1,2,3,4].

This will store `.pt` models into `base_model_dir` in a very similar structure as above:

```
folds_path
├── ch0
|   ├── epoch_1.pt
|   ├── epoch_2.pt
|   ├── ...
|   └── final_100.pt
├── ch1
|   ├── epoch_1.pt
|   ├── ...
├── ...
```

# Inference

Processing the folds of data using the final model is done using `predict.py`. This script assumes a file structure produced as above. The similarity matrix predictions for each recording are stored in a `<recording_id>.npy` format in subfolders called `ch*/<tr|te>_preds`.

To produce predictions:

```sh
python predict.py --cfg configs/<your_config>.cfg
```

# Evaluation

To obtain a diarization prediction, clustering is performed (using `cluster.py`) with the similarity matrix enhancement described in [1]. Like the paper, spectral clustering is included, and agglomerative clustering is also available.

For each fold of the CALLHOME dataset, a configurable range of cluster parameter values are evaluated to find the best performing value on the train set. The single best one is then used to cluster that test set. Each test set hypothesis is then combined to create the overall system hypothesis for CALLHOME.

The relevant sections in the configuration file for clustering are as follows:

```ini
[Clustering]
# Only 'sc' and 'ahc' are supported
cluster_type = sc

# The following values are fed into np.linspace to produce a range of parameters to try clustering the train portion over
# Note: cparam_start must be positive if spectral clustering is used.
cparam_start = 0.95
cparam_end = 1.0
cparam_steps = 20
```

Before running the clustering step, `md-eval.pl` will need to be obtained, which can be downloaded using:

```sh
wget https://raw.githubusercontent.com/foundintranslation/Kaldi/master/tools/sctk-2.4.0/src/md-eval/md-eval.pl
```

Finally:

```sh
python cluster.py --cfg configs/<your_config>.cfg
```

which will have an output similar to this:

```
Fold 0, cparam 0.9       Train DER: 15.5
5%|###########8                                  | 1/20 [00:55<17:33, 55.44s/it]
Fold 0, cparam 0.9052631578947369        Train DER: 15.07
10%|#######################6                    | 2/20 [01:43<15:59, 53.29s/it]
Fold 0, cparam 0.9105263157894737        Train DER: 14.44
...
```

# Results

TODO... (still tuning)


# Other issues/todos

* Test changing num_folds
* I-vectors instead of x-vectors - with system fusion
* Transformer and other architectures (some of which are in models.py)
* logspace option for cluster thresholds, or some other spacing options

# References

```
# [1]
@inproceedings{Lin2019,
  author={Qingjian Lin and Ruiqing Yin and Ming Li and Hervé Bredin and Claude Barras},
  title={{LSTM Based Similarity Measurement with Spectral Clustering for Speaker Diarization}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={366--370},
  doi={10.21437/Interspeech.2019-1388},
  url={http://dx.doi.org/10.21437/Interspeech.2019-1388}
}
```