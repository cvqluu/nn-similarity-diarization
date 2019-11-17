#!/bin/bash

stage=0
xvector_dir=/PATH/TO/XVECS #path to extracted xvectors
KALDI_PATH=/PATH/TO/KALDI_ROOT # path to kaldi root
folds_path=/PATH/TO/FOLDS_DATA # path to where the train/test split folds will be stored
cfg_path=/PATH/TO/CFG # path to main cfg, $folds_path is data_path in the cfg

num_folds=5 #default num folds is 5

if [ $stage -le 0 ]; then
    # makes k-fold dataset (default: 5 folds)
    python -m scripts.make_kfold_callhome $xvector_dir $KALDI_PATH/egs/callhome_diarization/v2/data/callhome/fullref.rttm $folds_path $num_folds
    cp $KALDI_PATH/egs/callhome_diarization/v2/data/callhome/fullref.rttm $folds_path
fi

if [ $stage -le 1 ]; then
    # train on each fold of data sequentially
    for i in $(seq 0 num=$(( $num - 1 )));
        python train.py --cfg $cfg_path --fold $i
    done
fi

if [ $stage -le 2 ]; then
    # make predictions using the final model of each train fold
    python predict.py --cfg $cfg_path
fi

if [ $stage -le 3 ]; then
    # Clustering
    # Finds best train set cluster parameter and then clusters tests sets using this value
    # combines all and evaluates for all test portions combined
    if [ ! -f "md-eval.pl" ]; then
        wget https://raw.githubusercontent.com/foundintranslation/Kaldi/master/tools/sctk-2.4.0/src/md-eval/md-eval.pl
    fi
    python cluster.py --cfg $cfg_path
fi