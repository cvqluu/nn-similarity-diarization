#!/bin/bash

# mostly adapted from original run.sh in 
# https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v2/run.sh

stage=0
nnet_dir=0006_callhome_diarization_v2_1a/exp/xvector_nnet_1a
nj=10
train_cmd=run.pl
callhome_path=/PATH/TO/CALLHOME #path to raw callhome data
xvector_dir=/PATH/TO/XVECS #path to extracted xvectors

if [ $stage -le 0 ]; then
    local/make_callhome.sh $callhome_path data
    utils/combine_data.sh data/callhome data/callhome1 data/callhome2
    cat data/callhome1/ref.rttm data/callhome2/ref.rttm > data/callhome/ref.rttm
fi

if [ $stage -le 1 ]; then
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/callhome exp/make_mfcc mfcc
    utils/fix_data_dir.sh data/callhome

    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      data/callhome exp/make_vad mfcc
    utils/fix_data_dir.sh data/callhome

    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
        data/callhome data/callhome_cmn exp/callhome_cmn
    cp data/callhome/vad.scp data/callhome_cmn/
    cp data/callhome/segments data/callhome_cmn/
    utils/fix_data_dir.sh data/callhome_cmn

fi

if [ $stage -le 2 ]; then
    # download pretrained kaldi model
    wget https://kaldi-asr.org/models/6/0006_callhome_diarization_v2_1a.tar.gz
    tar -xvf 0006_callhome_diarization_v2_1a.tar.gz
fi

if [ $stage -le 3 ]; then
    diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" \
        --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
        --min-segment 0.5 $nnet_dir \
        data/callhome_cmn $xvector_dir
fi