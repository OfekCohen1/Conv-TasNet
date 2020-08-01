# # Created on 2018/12/10
# # Author: Kaituo XU
#
# import torch
# torch.manual_seed(123)
#
# Cin, Cout, F, S = 4, 3, 2, 1
# B, Lin = 2, 5
# D = 2
# print('Cin, Cout, F, S, B, Lin, D=', Cin, Cout, F, S, B, Lin, D)
# conv1d = torch.nn.Conv1d(Cin, Cin, F, S, bias=False, padding=(F-1)*D, dilation=D, groups=Cin)
# # conv1d = torch.nn.Conv1d(Cin, Cin, F, S, bias=False, padding=1, dilation=1, groups=Cin)
# inputs = torch.randint(3, (B, Cin, Lin))
# conv1d.weight.data = torch.randint(5, conv1d.weight.size())
# outputs = conv1d(inputs)
# # Lout = (Lin - F) / S + 1
#
# print('weight', conv1d.weight.size())
# print('inputs', inputs.size())
# print('outputs', outputs.size())
#
# print('inputs\n', inputs)
# print('weight\n', conv1d.weight)
# print('outputs\n', outputs)
# print('chomp outputs\n', outputs[:,:,:-(F-1)*D])
#
# # m = torch.nn.Conv1d(16, 33, 3, stride=2)
# # print(m.weight.size())
# # input = torch.randn(20, 16, 50)
# # output = m(input)
# # print(output.size())

import torch
import torch.nn as nn
from src.DPRNN_model import DPRNN


class test_class(nn.Module):

    def __init__(self):
        super(test_class, self).__init__()
        self.conv1 = nn.Conv1d(50, 100, 3)
        self.conv2 = nn.Conv1d(50, 100, 3)

    def forward(self, x):
        return self.conv1(x), self.conv2(x)


# Checks lookahead of model
def check_lookahead(lookahead_time, model, sample_rate):
    lookahead_samples = int(lookahead_time * sample_rate)
    input_wav = torch.randn(3, 32000).to('cuda')
    T = 7014
    with torch.no_grad():
        output_wav = model(input_wav)[:, 0, :]  # Only take signal output, not noise
        # I changed from 5000 and after, which means this shouldn't influence things before (5000-lookahead)
        input_wav[:, T:] = 0
        output_changed_wav = model(input_wav)[:, 0, :]
        lookahead_works = (output_changed_wav[:, :(T - lookahead_samples)] -
                           output_wav[:, :(T - lookahead_samples)]).sum().item()
        print(lookahead_works)
        if abs(lookahead_works) < 1e-3:
            print('Lookahead is smaller than {0:0.2f}ms, WORKS AS INTENDED'.format(lookahead_time * 1e3))
        else:
            print('Lookahead is BIGGER than {0:0.2f}ms, DOES NOT WORK'.format(lookahead_time * 1e3))


if __name__ == '__main__':
    model_path = "../egs/models/DPRNN_SE_LSTM_N_64_B_96_hidden_128_chunk_180_L_6_sr_16k.pth"
    sample_rate = 16000
    # Lookahead should be small to check if it works.
    chunk_size = 180
    lookahead_time = chunk_size * 3 / sample_rate
    # lookahead_time = 30 * 1e-3
    model = DPRNN.load_model(model_path)

    model.eval()
    model.cuda()
    check_lookahead(lookahead_time, model, sample_rate)
