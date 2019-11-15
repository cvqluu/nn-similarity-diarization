# Neural network based similarity scoring for diarization

PyTorch implementation of neural network based similarity scoring for diarization: based on the paper **"LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization"** at INTERSPEECH 2019 https://arxiv.org/abs/1907.10393, https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1388.pdf 

I am not affiliated with the paper authors.

The basic concept behind this method is to learn the similarity scoring matrix needed for diarizing a recording. Inputs of concatenated speaker embeddings (such as x-vectors) are fed through an LSTM or other architecture to predict the similarity of the concatenated embeddings.

## Requirements

Kaldi, python, kaldi_io, scipy, sklearn, torch, CALLHOME dataset

# TL;DR

(TODO WIP): You can run the whole recipe with `run.sh`

# Data preparation

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

Changing directory back to where this repo is, run the following command to make the train/test folds, replacing the variables as is necessary. Here `$xvector_dir` is as above and `$folds_path` is the location in which the splits will reside.

```sh
python make_kfold_callhome.py $xvector_dir $KALDI_PATH/egs/callhome_diarization/v2/data/callhome/ref.rttm $folds_path
```

which makes a folder structure like so

```
folds_path
├── ch0
|   ├── train
|   |   ├── utt2spk
|   |   ├── segments
|   |   └── xvector.scp
|   └── test
├── ch1
|   ├── train
|   └── test
├── ...
```

# Training

The primary training script `train.py` is mostly defined by the config file which it reads. An example config file is shown in `configs/example.cfg`:

```ini
[Datasets]
data_path = /PATH/TO/FOLDS_PATH

[Model]
model_type = lstm

[Hyperparams]
lr = 0.2
max_len = 400
no_cuda = False
seed = 1234
num_epochs = 100
scheduler_steps = [40, 80]
scheduler_lambda = 0.1

[Outputs]
base_model_dir = exp/example_models_folder
```

The main fields which need to be configured are `data_path` and `base_model_dir`. The first corresponds to `$folds_path` used above and the latter will be the place in which the models are stored.

Once this cfg file is configured, a model can be trained on a fold like so:

```sh
python train.py --cfg configs/<your_config>.cfg --fold 0
```

Alternatively, if you wish to train all folds sequentially with one script:

```sh
python run_train_ch_folds.py configs/<your_config>.cfg
```

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

# Evaluation

TODO

# Results

TODO


# References

```
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