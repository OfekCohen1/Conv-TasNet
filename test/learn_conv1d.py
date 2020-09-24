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


class test_class(nn.Module):

    def __init__(self):
        super(test_class, self).__init__()
        self.conv1 = nn.Conv1d(50, 100, 3)
        self.conv2 = nn.Conv1d(50, 100, 3)

    def forward(self, x):
        return self.conv1(x), self.conv2(x)

if __name__ == '__main__':
    import torch
    import os
    from fairseq.models.wav2vec import Wav2VecModel

    cp = torch.load('../egs/models/loss_models/wav2vec_large.pt')
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.eval()

    wav_input_16khz = torch.randn(3, 10000)
    print(wav_input_16khz.shape)
    z = model.feature_extractor(wav_input_16khz)
    z2 = model.feature_extractor(wav_input_16khz * 127)
    print(torch.mean(z))
    print(torch.mean(z2))
    print(z.shape)
    c = model.feature_aggregator(z)
    print(c.shape)
