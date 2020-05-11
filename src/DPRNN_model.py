# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

from src.utils import overlap_and_add

EPS = 1e-8


class DPRNN(nn.Module):
    def __init__(self, input_size, bottleneck_size, hidden_size, C,
                 num_layers=6, chunk_size=180, rnn_type='LSTM', L=16, norm_type="cLN", causal=True):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        # TODO: Fix args here
        super(DPRNN, self).__init__()
        # Hyper-parameter
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.causal = causal
        self.L = L
        self.chunk_size = chunk_size
        self.output_size = bottleneck_size
        self.C = C
        self.rnn_type = rnn_type

        bidirectional = False if causal else True
        # Components
        self.encoder = Encoder(L, input_size)

        self.bottleneck_conv1v1 = nn.Conv1d(input_size, bottleneck_size, 1, bias=False)
        self.separator = DPRNN_separator(rnn_type, bottleneck_size, hidden_size, self.output_size,
                                         num_layers=num_layers, bidirectional=bidirectional)
        num_params = 0
        for paramter in self.separator.parameters():
            num_params += torch.numel(paramter)
        print("Model size is:" + str(num_params / 1e6))
        self.mask_conv1v1 = nn.Conv1d(bottleneck_size, input_size * C, 1, bias=False)

        self.decoder = Decoder(input_size, L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)  # M x N x K
        mixture_bottleneck = self.bottleneck_conv1v1(mixture_w)
        mixture_bottleneck, rest = self.split_feature(mixture_bottleneck, self.chunk_size)  # M x B x chunk_size x S
        est_mask = self.separator(mixture_bottleneck)  # M x B x chunk_size x S
        est_mask = self.merge_feature(est_mask, rest)
        est_mask = self.mask_conv1v1(est_mask)  #
        M, _, K = est_mask.shape
        est_mask = est_mask.view(M, self.C, self.input_size, K)

        est_source = self.decoder(mixture_w, est_mask)  # M x C x T_conv

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = functional.pad(est_source, (0, T_origin - T_conv))

        # import GPUtil
        # GPUtil.showUtilization()

        return est_source

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['input_size'], package['bottleneck_size'], package['hidden_size'], package['C'],
                    num_layers=package['num_layers'], chunk_size=package['chunk_size'], rnn_type=package['rnn_type'], L=package['L'],
                    norm_type=package['norm_type'], causal=package['causal'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'input_size': model.input_size, 'bottleneck_size': model.bottleneck_size, 'hidden_size': model.hidden_size,'C': model.C,
            'num_layers': model.num_layers, 'chunk_size': model.chunk_size, 'rnn_type': model.rnn_type, 'L': model.L, 'norm_type': model.norm_type,
            'causal': model.causal,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mix_wave):
        """
        Args:
            mix_wave: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        # L is the length of each mini frame, and K is the number of frames. conv basically
        # transfers from L length base to N length base, using N vectors (N dimensions).
        # one conv filter is basically taking one basis vector on all K frames
        mix_wave = torch.unsqueeze(mix_wave, 1)  # [M, 1, T]
        mixture_w = functional.relu(self.conv1d_U(mix_wave))  # [M, N, K]

        # Segment from [M, N, K] to [M, N, chunk_len, S]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        # N - in features. L - out features. works on all K vectors.
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M - from Tasnet paper
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N] - K vectors, each vector length N
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, L] - K vectors, each vector length L
        # we now have K vectors of length L with overlap of L/2 cause of stride. now to fix it:
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T
        return est_source


class DPRNN_separator(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True):
        super(DPRNN_separator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        norm_type = 'gLN' if bidirectional else 'cLN'
        # dual-path RNN
        self.in_segment_rnn = nn.ModuleList([])
        self.between_segments_rnn = nn.ModuleList([])
        self.in_segment_norm = nn.ModuleList([])
        self.between_segments_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.in_segment_rnn.append(SingleRNN(rnn_type, input_size, hidden_size, dropout,
                                                 bidirectional=True))  # intra-segment RNN is always noncausal
            self.between_segments_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.in_segment_norm.append(chose_norm(norm_type, input_size))
            self.between_segments_norm.append(chose_norm(norm_type, input_size))

        # output layer
        # self.output = nn.Sequential(nn.PReLU(),
        #                             nn.Conv2d(input_size, output_size, 1)
        #                             )
        self.output = nn.Sequential(nn.PReLU())

    def forward(self, input):
        # input shape: batch, N, chunk_size, S
        # apply RNN on chunk_size first and then S
        # TODO: Check between 1) norm before reshape, and 2) norm after reshape. currently norm before reshape
        batch_size, _, chunk_size, S = input.shape
        input = input.float()
        output = input
        for i in range(len(self.in_segment_rnn)):
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * S, chunk_size,
                                                                     -1)  # B*S, chunk_size, N
            # print(row_input.shape)
            row_output = self.in_segment_rnn[i](row_input)  # B*S, chunk_size, H
            row_output = row_output.view(batch_size, S, chunk_size, -1).permute(0, 3, 2,
                                                                                1).contiguous()  # B, N, chunk_size, S
            row_output = self.in_segment_norm[i](row_output)
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * chunk_size, S,
                                                                     -1)  # B*chunk_size, S, N
            col_output = self.between_segments_rnn[i](col_input)  # B*chunk_size, S, H
            col_output = col_output.view(batch_size, chunk_size, S, -1).permute(0, 3, 1,
                                                                                2).contiguous()  # B, N, chunk_size, S
            col_output = self.between_segments_norm[i](col_output)
            output = output + col_output

        output = self.output(output)

        return output


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return InterChunkNorm(channel_size)
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


class InterChunkNorm(nn.Module):
    """Inter Chunk Normalization (cLN)"""

    def __init__(self, channel_size):
        super(InterChunkNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, chunk, S], M is batch size, N is channel size, chunk is Chunk length, S is num of chunks.
        Returns:
            cLN_y: [M, N, chunk, S]
        """
        # TODO: Think if this is a causal norm when it's Batch x N x chunk x S
        mean = torch.mean(y, dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1, S]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, 1, S]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, chunk, S], M is batch size, N is channel size, chunk is Chunk length, S is num of chunks.
        Returns:
            cLN_y: [M, N, chunk, S]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)  # [M, 1, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, segment_size)
    batch_size, channnel_size, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, channnel_size, -1, segment_size)
    segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, channnel_size, -1, segment_size)
    segments = torch.cat([segments1, segments2], 3).view(batch_size, channnel_size, -1, segment_size).transpose(2, 3)

    return segments, rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, K, L

    input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
    input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


if __name__ == "__main__":
    torch.manual_seed(123)
    # M, N, L, T = 4, 64, 20, 32000
    # norm_type, causal = "cLN", True
    # mixture = torch.randint(3, (M, T))
    #
    # rnn_type = 'LSTM'
    # bidirectional = False if causal else True
    # input_size = N
    # bottleneck_size = 96
    # hidden_size = 128
    # num_layers = 6
    # chunk_size = 180
    # # test Encoder
    # model = DPRNN(input_size, bottleneck_size, hidden_size,
    #               layer=num_layers, chunk_size=chunk_size, rnn_type=rnn_type,
    #               L=10, norm_type=norm_type, causal=True)
    # mixture = mixture.float()
    #
    # estimated_stuff = model(mixture)
    #
    # encoder = Encoder(L, N)
    #
    # bottleneck_conv1v1 = nn.Conv1d(N, bottleneck_size, 1, bias=False)
    # mask_conv1v1 = nn.Conv1d(bottleneck_size, N * C, 1, bias=False)
    #
    # encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    # mixture_w = encoder(mixture)
    # print('mixture', mixture)
    # print('U', encoder.conv1d_U.weight)
    # print('mixture_w', mixture_w)
    # print('mixture_w size', mixture_w.size())
    #
    #
    # mixture_w = bottleneck_conv1v1(mixture_w)
    # mixture_w, rest = split_feature(mixture_w, chunk_size)
    # # test TemporalConvNet
    # separator = DPRNN_separator(rnn_type, bottleneck_size, hidden_size, bottleneck_size,
    #              dropout=0, num_layers=1, bidirectional=bidirectional)
    # est_mask = separator(mixture_w)
    # print('est_mask', est_mask)
    # est_mask = merge_feature(est_mask, rest)
    # est_mask = mask_conv1v1(est_mask)
    # M, _, L = est_mask.shape
    # est_mask = est_mask.view(M, C, N, L)
