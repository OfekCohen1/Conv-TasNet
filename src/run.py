import numpy as np

# from src.conv_tasnet import
from src.preprocess import preprocess
from src.train import train
import torch
from src.data import AudioDataset, AudioDataLoader

# Trying to imitate the run.sh script from the original github

# I'm using the librispeech dataset, which comes in .flac type, so first thing use flac_to_wav.py
# Then use the "script create_txt_file_like_wsj0.py" to create the txt file, and then
# the matlab code to create the wsj0-2mix dataset

# To open visdom, run this command: "python -m visdom.server", and then open http://localhost:8097

# data_dir = "../egs/SE_dataset/"
# json_dir = "../egs/SE_dataset/"
data_dir = "../egs/Librispeech_SE_dataset/"

train_dir = data_dir + "tr"
valid_dir = data_dir + "cv"
test_dir = data_dir + "tt"

id = 0
epochs = 50

# save and visualize

continue_from = ""
model_path = "test.pth"
model_features_path = "../egs/models/loss_models/librispeech_pretrained_v2.pth"

if __name__ == '__main__':
    sample_rate = 16000
    check_dataset_dir = "../egs/SE_dataset/"
    data_dir = check_dataset_dir
    json_dir = check_dataset_dir
    # preprocess(data_dir, json_dir, sample_rate)

    batch_size = 3
    max_hours = None
    num_workers = 4
    # check_dataset_dir = "../egs/Librispeech_SE_dataset/tr"
    # sample_rate = 16000
    # dataset = AudioDataset(check_dataset_dir, batch_size, segment=4, max_hours=max_hours, sample_rate=sample_rate)
    # dataloader = AudioDataLoader(dataset, batch_size=1, num_workers=num_workers)
    # i = 0
    # for data in dataloader:
    #     i += 1
    # print(i)
    train(data_dir, epochs, batch_size, model_path, model_features_path, max_hours=max_hours, continue_from=continue_from)
