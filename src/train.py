#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU


import torch

from src.data import AudioDataLoader, AudioDataset
from src.solver import Solver
from src.conv_tasnet import ConvTasNet
from src.DPRNN_model import DPRNN


def train(data_dir, epochs, batch_size, model_path, max_hours=None, continue_from=""):
    # General config
    # Task related
    json_dir = data_dir
    train_dir = data_dir + "tr"
    valid_dir = data_dir + "cv"
    sample_rate = 8000
    segment_len = 4

    # Network architecture
    causal = 1
    rnn_type = 'LSTM'
    input_size = 64
    bottleneck_size = 96
    hidden_size = 128
    num_layers = 6
    chunk_size = 180
    L = 10
    C = 2

    bidirectional = False if causal else True
    norm_type = 'cLN' if causal else 'gLN'

    use_cuda = 1

    half_lr = 1  # Half the learning rate when there's a small improvement
    early_stop = 1  # Stop learning if no imporvement after 10 epochs
    max_grad_norm = 5  # gradient clipping

    shuffle = 1  # Shuffle every epoch
    num_workers = 0
    # optimizer
    optimizer_type = "adam"
    lr = 1e-3
    momentum = 0
    l2 = 0  # Weight decay - l2 norm

    # save and visualize
    save_folder = "../egs/models"
    enable_checkpoint = 0  # enables saving checkpoints
    # continue_from = ""  # model to continue from  # TODO check this
    # model_path = ""  # TODO: Fix this
    print_freq = 5
    visdom_enabled = 0
    visdom_epoch = 0
    visdom_id = "Conv-TasNet Training"  # TODO: Check what this does

    arg_solver = (use_cuda, epochs, half_lr, early_stop, max_grad_norm, save_folder, enable_checkpoint, continue_from,
                  model_path, print_freq, visdom_enabled, visdom_epoch, visdom_id)

    # Datasets and Dataloaders
    tr_dataset = AudioDataset(train_dir, batch_size,
                              sample_rate=sample_rate, segment=segment_len, max_hours=max_hours)
    cv_dataset = AudioDataset(valid_dir, batch_size,
                              sample_rate=sample_rate,
                              segment=segment_len)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    model = DPRNN(input_size, bottleneck_size, hidden_size, C, num_layers=num_layers,
                  chunk_size=chunk_size, rnn_type=rnn_type, L=L, norm_type=norm_type, causal=causal)
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
    solver = Solver(data, model, optimizer, arg_solver)  # TODO: Fix solver thing
    solver.train()


if __name__ == '__main__':
    print('train main')
    # args = parser.parse_args()
    # print(args)
    # train(args)
