# Neural network based similarity scoring for diarization

PyTorch implementation of neural network based similarity scoring for diarization: based on the paper **"LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization"** at INTERSPEECH 2019 https://arxiv.org/abs/1907.10393, https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1388.pdf 

I am not affiliated with the paper authors.

The basic concept behind this method is to learn the similarity scoring matrix needed for diarizing a recording. Inputs of concatenated speaker embeddings (such as x-vectors) are fed through an LSTM or other architecture to predict the similarity of the concatenated embeddings.

## Requirements

Kaldi, python, kaldi_io, torch, CALLHOME dataset

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

These need to point to where the CALLHOME dataset is and also where you would like the extracted x-vectors to reside. Once this is done, this script is copied to the Kaldi recipe folder for CALLHOME (as existing data prep scripts are leveraged):

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
...
```

# Training

TODO

# Evaluation

TODO

# Results

TODO



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