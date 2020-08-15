#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU


import torch
from src.data import AudioDataLoader, AudioDataset
from src.solver import Solver
from src.conv_tasnet import ConvTasNet
from src.DPRNN_model import DPRNN
from src.TPRNN_omlsa import TPRNN_omlsa
from torch.utils.data.dataset import random_split
import math
from src.deepspeech_model import DeepSpeech
from fairseq.models.wav2vec import Wav2VecModel


def train(data_dir, epochs, batch_size, model_path, model_features_path, max_hours=None, continue_from=""):
    # General config
    # Task related
    train_dir = data_dir + "tr"
    sample_rate = 8000
    segment_len = 4

    # Network architecture
    causal = 1
    rnn_type = 'GRU'
    input_size = 128
    bottleneck_size = 256
    hidden_size = 256
    num_layers = 3

    # input_size = 48
    # bottleneck_size = 64
    # hidden_size = 96
    # num_layers = 1
    # chunk_size = 180
    L = 6
    C = 1  # TODO: Currently doesn't output noise, only audio

    norm_type = 'cLN' if causal else 'gLN'

    use_cuda = 1
    device = torch.device("cuda" if use_cuda else "cpu")
    half_lr = 1  # Half the learning rate when there's a small improvement
    early_stop = 1  # Stop learning if no imporvement after 10 epochs
    max_grad_norm = 5  # gradient clipping

    shuffle = 1  # Shuffle every epoch
    num_workers = 4
    # optimizer
    optimizer_type = "adam"
    lr = 5e-4
    momentum = 0
    l2 = 0  # Weight decay - l2 norm

    # save and visualize
    save_folder = "../egs/models"
    enable_checkpoint = 0  # enables saving checkpoints
    print_freq = 50
    visdom_enabled = 1
    visdom_epoch = 1
    visdom_id = "Conv-TasNet Training"

    # deep_features_model = DeepSpeech.load_model(model_features_path)
    # deep_features_model.eval()

    # cp = torch.load('../egs/models/loss_models/wav2vec_large.pt')
    # deep_features_model = Wav2VecModel.build_model(cp['args'], task=None)
    # deep_features_model.load_state_dict(cp['model'])
    # deep_features_model.eval()
    deep_features_model = None

    arg_solver = (use_cuda, epochs, half_lr, early_stop, max_grad_norm, save_folder, enable_checkpoint, continue_from,
                  model_path, print_freq, visdom_enabled, visdom_epoch, visdom_id, deep_features_model, device)

    # Datasets and Dataloaders
    tr_cv_dataset = AudioDataset(train_dir, batch_size,
                              sample_rate=sample_rate, segment=segment_len, max_hours=max_hours)
    cv_len = int(0.2 * math.ceil(len(tr_cv_dataset)))
    tr_dataset, cv_dataset = random_split(tr_cv_dataset, [len(tr_cv_dataset) - cv_len, cv_len])

    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=num_workers)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    model = TPRNN_omlsa(input_size, bottleneck_size, hidden_size, C, num_layers=num_layers,
                   rnn_type=rnn_type, L=L, norm_type=norm_type, causal=causal)

    if use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=l2)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizer, arg_solver)
    solver.train()


if __name__ == '__main__':
    print('train main')
    # args = parser.parse_args()
    # print(args)
    # train(args)