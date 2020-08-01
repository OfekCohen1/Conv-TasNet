#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
import json
from pystoi import stoi
from pesq import pesq

from src.pit_criterion import calc_si_sdr
from src.separate import separate
from src.preprocess import preprocess_one_dir
from tqdm import tqdm


def evaluate(est_dir, clean_dir, use_cuda, sample_rate, batch_size):
    est_json = os.path.join(est_dir, 'est.json')
    clean_json = os.path.join(clean_dir, 'clean.json')

    def sort(infos): return sorted(
        infos, key=lambda info: int(info[0].split("_")[-1].split(".")[0]))  # splits to get fileid

    with open(est_json, 'r') as f:
        est_infos = json.load(f)
    with open(clean_json, 'r') as f:
        clean_infos = json.load(f)

    sorted_est_infos = sort(est_infos)
    sorted_clean_infos = sort(clean_infos)

    assert len(sorted_clean_infos) == len(sorted_est_infos)
    sisdr_test = 0
    pesq_test = 0
    stoi_test = 0
    for i in tqdm(range(len(sorted_clean_infos))):
        est_path = sorted_est_infos[i][0]
        clean_path = sorted_clean_infos[i][0]
        est_source, _ = librosa.load(est_path, sr=sample_rate)
        clean_source, _ = librosa.load(clean_path, sr=sample_rate)
        source_length = torch.tensor([[len(est_source)]])
        tensor_est_source = torch.tensor(est_source).reshape(1, len(est_source))  # batch_size = 1
        tensor_clean_source = torch.tensor(clean_source).reshape(1, len(clean_source))  # batch_size = 1

        sisdr_test = sisdr_test + calc_si_sdr(tensor_clean_source, tensor_est_source, source_length).item() / len(
            sorted_clean_infos)
        pesq_test = pesq_test + pesq(sample_rate, clean_source, est_source, 'nb') / len(sorted_clean_infos)
        stoi_test = stoi_test + stoi(clean_source, est_source, sample_rate) / len(sorted_clean_infos)

    print('SI-SDR on DNS Test Set: {0:.2f}'.format(sisdr_test))
    print('PESQ on DNS Test Set: {0:.2f}'.format(pesq_test))
    print('STOI on DNS Test Set: {0:.2f}'.format(100 * stoi_test))


if __name__ == '__main__':
    model_path = "../egs/models/Librispeeech_DPRNN_SE_LSTM_N_64_B_96_hidden_128_chunk_180_L_6_sr_16k.pth"
    noisy_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/noisy"
    clean_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/clean"
    noisy_json = ""
    # Where to dump estimated clean audio, make sure this folder is empty
    est_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/estimated_me"

    use_cuda = 1
    sample_rate = 16000
    batch_size = 1

    # First separate to a specific folder - then calculate SISNR
    separate(model_path, noisy_dir, noisy_json, est_dir, use_cuda, sample_rate, batch_size)

    preprocess_one_dir(est_dir, est_dir, 'est', sample_rate=sample_rate)
    preprocess_one_dir(clean_dir, clean_dir, 'clean', sample_rate=sample_rate)
    evaluate(est_dir, clean_dir, use_cuda, sample_rate, batch_size)
