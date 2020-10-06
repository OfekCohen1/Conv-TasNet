import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
import json
from pystoi import stoi
from pesq import pesq
import pysepm

from src.pit_criterion import calc_si_sdr
from src.separate import separate
from src.preprocess import preprocess_one_dir
from tqdm import tqdm
import math


def mapped_mos2raw_mos(mapped):
    return (math.log(4.0/(mapped - 0.999) - 1.0) - 4.6607) / (-1.4945)


def evaluate(est_dir, clean_dir, sample_rate):
    print('Sample rate is', int(sample_rate), 'Hz')
    est_json = os.path.join(est_dir, 'mix.json')
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
    pesq_test2 = 0
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
        pesq_test = pesq_test + pesq(sample_rate, clean_source, est_source, 'wb') / len(sorted_clean_infos)
        # pesq_test = pesq_test + pesq(clean_source, est_source, sample_rate) / len(sorted_clean_infos)
        pesq_test = pesq_test + (pysepm.pesq(clean_source,est_source, sample_rate)[0]) / len(sorted_clean_infos)
        pesq_test2 = pesq_test2 + (pysepm.pesq(clean_source,est_source, sample_rate)[1]) / len(sorted_clean_infos)

        stoi_test = stoi_test + stoi(clean_source, est_source, sample_rate) / len(sorted_clean_infos)
    #
    print('SI-SDR on DNS Test Set: {0:.2f}'.format(sisdr_test))
    print('PESQ on DNS Test Set: {0:.2f}'.format(mapped_mos2raw_mos(pesq_test)))
    print('PESQ2 on DNS Test Set: {0:.2f}'.format(mapped_mos2raw_mos(pesq_test2)))

    print('STOI on DNS Test Set: {0:.2f}'.format(100 * stoi_test))


def calc_num_params(model):
    num_params = 0
    for paramter in model.parameters():
        num_params += torch.numel(paramter)
    print("Model size is:" + str(num_params / 1e6))
    return num_params/1e6


def create_input_for_model(batch_size, sample_length, model_type):
    if model_type == 'TASNET' or model_type == 'DPRNN' or model_type == 'TPRNN':
        dummy_input = torch.rand(batch_size, sample_length)
    else:  # In the future add other models
        dummy_input = torch.rand(batch_size, sample_length)
    return dummy_input


def count_macs_for_forward(model, model_name, mode='cpu',
                           sample_length=8000, batch_size=4):
    from thop import profile
    mixture = create_input_for_model(batch_size, sample_length,
                                     model_name)
    model.eval()
    if mode == 'gpu':
        mixture = mixture.cuda()
        model.cuda()
    macs, _ = profile(model, inputs=(mixture,))
    print('GMACS: {}'.format(round(macs / 10 ** 9, 3)))
    return macs


if __name__ == '__main__':
    model_path = "../egs/models/DPRNN_SE_LSTM_N_64_B_96_hidden_128_chunk_180_L_6.pth"
    noisy_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/noisy"
    clean_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/clean"
    noisy_json = ""
    # Where to dump estimated clean audio, make sure this folder is empty
    est_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/estimated_me"

    use_cuda = 1
    sample_rate = 8000
    batch_size = 1

    # First separate to a specific folder - then calculate SISNR
    # separate(model_path, noisy_dir, noisy_json, est_dir, use_cuda, sample_rate, batch_size)
    # preprocess_one_dir(est_dir, est_dir, 'est', sample_rate=sample_rate)
    # preprocess_one_dir(clean_dir, clean_dir, 'clean', sample_rate=sample_rate)
    est_dir = "../egs/SE_dataset/tt/synthetic/no_reverb/noisy"
    evaluate(est_dir, clean_dir, sample_rate)

    from src.DCCRN import DCCRN
    # model_path = "../egs/models/TPRNN_SE_GRU_N_128_B_256_hidden_256_layers_3_L_6_sr_8k_batch_256_hours_30.pth"
    # model_tprnn = DCCRN.load_model(model_path)

    # macs_dprnn = count_macs_for_forward(model_dprnn, 'DPRNN', mode='cpu')
    # macs_tasnet = count_macs_for_forward(model_tasnet, 'TASNET', mode='cpu')
    # macs_tprnn = count_macs_for_forward(model_tprnn, 'TPRNN', mode='cpu')
    #
    # print('DPRNN GMACS: {0:.2f}, number of parameters: {1:.2f} M'
    #       .format(round(macs_dprnn / 10 ** 9, 3), calc_num_params(model_dprnn)))
    # print('Tasnet GMACS: {0:.2f}, number of parameters: {1:.2f} M'
    #       .format(round(macs_tasnet / 10 ** 9, 3), calc_num_params(model_tasnet)))
    # print('TPRNN GMACS: {0:.2f}, number of parameters: {1:.2f} M'
    #       .format(round(macs_tprnn / 10 ** 9, 3), calc_num_params(model_tprnn)))