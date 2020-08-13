# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

from src.utils import overlap_and_add

EPS = 1e-8


# Temporal Attention RNN

class TARNN(nn.Module):
    def __init__(self, input_size, bottleneck_size, hidden_size, conv_size, pyramid_size, P, X, R, C,
                 rnn_type='LSTM', L=16, norm_type="cLN", causal=True, num_rnns_long_term=1, num_rnns_short_term=1):
        super(TARNN, self).__init__()
        # Hyper-parameter
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.hidden_size = hidden_size
        self.norm_type = norm_type
        self.causal = causal
        self.L = L
        self.output_size = bottleneck_size
        self.C = C
        self.rnn_type = rnn_type

        bidirectional = False if causal else True
        # Components
        self.encoder = Encoder(L, input_size)

        self.bottleneck_conv1v1 = nn.Sequential(chose_norm(norm_type, input_size),
                                                nn.Conv1d(input_size, bottleneck_size, 1, bias=False))

        self.separator = TARNN_separator(rnn_type, bottleneck_size, hidden_size, bottleneck_size, conv_size,
                                         pyramid_size, P, X, R, bidirectional=bidirectional,
                                         num_rnns_short_term=num_rnns_short_term, num_rnns_long_term=num_rnns_long_term)

        self.mask_conv1v1 = nn.Conv1d(2 * bottleneck_size, input_size * C, 1, bias=False)

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
        mixture_bottleneck = self.bottleneck_conv1v1(mixture_w)  # M x B x K
        est_mask = self.separator(mixture_bottleneck)  # M x B x K
        est_mask = self.mask_conv1v1(est_mask)  # M x N x K
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
                    num_layers=package['num_layers'], rnn_type=package['rnn_type'],
                    L=package['L'],
                    norm_type=package['norm_type'], causal=package['causal'])
        model.load_state_dict(package['state_dict'])
        return model


    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'input_size': model.input_size, 'bottleneck_size': model.input_size,
            'hidden_size': model.hidden_size,
            'C': model.C,
            'num_layers': model.num_layers, 'rnn_type': model.rnn_type, 'L': model.L,
            'norm_type': model.norm_type,
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


class TARNN_separator(nn.Module):
    """
    Currently input_size = hidden_size = output_size = B

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        bottleneck_size : int, dimension of the input feature. The input should have shape
                    (batch, seq_len, bottleneck_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, bottleneck_size, hidden_size, output_size, conv_size, pyramid_size, P=3, X=8, R=3,
                 dropout=0, num_rnns_long_term=1, num_rnns_short_term=1, bidirectional=False):
        super(TARNN_separator, self).__init__()
        self.long_term_module = Long_Term_Module(rnn_type, bottleneck_size, hidden_size, output_size, conv_size,
                                                 pyramid_size, P, X, R, dropout, num_rnns_long_term, bidirectional)
        self.short_term_module = Short_Term_Module(rnn_type, 2 * bottleneck_size, 2 * hidden_size, dropout,
                                                   num_rnns_short_term, bidirectional)

    def forward(self, input):
        # input shape: batch, B, K
        output_long_term = self.long_term_module(input)
        input_short_term = torch.cat((input, output_long_term), dim=1)  # Cat in dimensions
        output = self.short_term_module(input_short_term)
        return output


class Long_Term_Module(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, output_size, conv_size, pyramid_size, P=3, X=8, R=3,
                 dropout=0, num_rnns_long_term=1, bidirectional=False):
        super(Long_Term_Module, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size  # Currently input_size = hidden_size = B
        self.conv_size = conv_size
        norm_type = 'gLN' if bidirectional else 'cLN'
        causal = not bidirectional

        self.temporal_pyramid = Temporal_Pyramid(input_size, conv_size, P, X, R, norm_type, causal)
        self.temporal_attention = Temporal_Attention(input_size, pyramid_size)
        self.rnn_chain = nn.ModuleList([])
        self.norm_chain = nn.ModuleList([])
        self.prelu_chain = nn.ModuleList([])

        for i in range(num_rnns_long_term):
            self.rnn_chain.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional))
            self.norm_chain.append(chose_norm(norm_type, input_size))
            self.prelu_chain.append(nn.PReLU())

    def forward(self, input):
        # input shape: batch, B, K

        temporal_pyramid_input = self.temporal_pyramid(input)
        output = self.temporal_attention(input, temporal_pyramid_input)

        for i in range(len(self.rnn_chain)):
            rnn_output = self.rnn_chain[i](output.permute(0, 2, 1))
            rnn_output = rnn_output.permute(0, 2, 1)
            normalized_output = self.norm_chain[i](rnn_output)
            output = output + self.prelu_chain[i](normalized_output)
        return output


class Short_Term_Module(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_rnn_short_term=1, bidirectional=False):
        super(Short_Term_Module, self).__init__()

        self.bottleneck_size = input_size
        norm_type = 'gLN' if bidirectional else 'cLN'

        self.rnn_chain = nn.ModuleList([])
        self.norm_chain = nn.ModuleList([])
        self.prelu_chain = nn.ModuleList([])
        for i in range(num_rnn_short_term):
            self.rnn_chain.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional))
            self.norm_chain.append(chose_norm(norm_type, input_size))
            self.prelu_chain.append(nn.PReLU())

    def forward(self, input):
        # input shape: batch, B, K
        output = input

        for i in range(len(self.rnn_chain)):
            rnn_output = self.rnn_chain[i](output.permute(0, 2, 1))
            rnn_output = rnn_output.permute(0, 2, 1)
            normalized_output = self.norm_chain[i](rnn_output)
            output = output + self.prelu_chain[i](normalized_output)
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

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        return rnn_output


class Temporal_Attention(nn.Module):
    """
    input: original signal, should be size B x K
    temporal_input: list of length (pyramid_size), each element a tensor of shape B x K
    W_i_list: matrix for each output of pyramid
    W_T: matrix for all pyramid outputs, after they're summed together
    W_r: matrix for output after summation with input
    """

    def __init__(self, channel_size, pyramid_size):
        super(Temporal_Attention, self).__init__()
        self.channel_size = channel_size
        self.pyramid_size = pyramid_size
        self.W_i_list = nn.ModuleList()
        for i in range(pyramid_size):
            self.W_i_list.append(nn.Conv1d(channel_size, channel_size, 1))

        self.W_T = nn.Conv1d(channel_size, channel_size, 1)
        self.W_X = nn.Conv1d(channel_size, channel_size, 1)

        self.after_summation = nn.Sequential(nn.ReLU(), nn.Conv1d(channel_size, channel_size, 1), nn.Sigmoid())

    def forward(self, input, temporal_inputs):
        T_i = []
        for i in range(len(temporal_inputs)):
            T_i.append(self.W_i_list[i](temporal_inputs[i]))
        # TODO: Might want to put a sigmoid before T summation
        temporal_output = self.W_T(sum(T_i))
        X_output = self.W_X(input)

        attention = self.after_summation(temporal_output + X_output)
        output = attention * input
        return output


class Temporal_Pyramid(nn.Module):
    def __init__(self, bottleneck_size, conv_size, P, X, R, norm_type="cLN", causal=True):
        """
        Args:
            N: Number of filters in autoencoder
            bottleneck_size: Number of channels in bottleneck 1 × 1-conv block
            conv_size: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of temporal_blocks_repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
        """
        super(Temporal_Pyramid, self).__init__()
        # Hyper-parameter
        # Components
        temporal_blocks_repeats = nn.ModuleList()
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [TemporalBlock(bottleneck_size, conv_size, P, stride=1,
                                         padding=padding,
                                         dilation=dilation,
                                         norm_type=norm_type,
                                         causal=causal)]
            temporal_blocks_repeats.append(nn.Sequential(*blocks))
        self.temporal_blocks_repeats = temporal_blocks_repeats

    def forward(self, mixture_w):
        """
        Args:
            mixture_w: [M, B, K], M is batch size
        returns:
            est_mask: [M, B, K]
        """
        M, B, K = mixture_w.size()
        output = mixture_w
        temporal_pyramid_outputs = []
        for i in range(len(self.temporal_blocks_repeats)):
            output = self.temporal_blocks_repeats[i](output)
            temporal_pyramid_outputs.append(output)
        return temporal_pyramid_outputs


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="cLN", causal=True):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="cLN", causal=True):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                     pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


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
        return ChannelNorm(channel_size)


class ChannelNorm(nn.Module):
    """Inter Chunk Normalization (cLN)"""

    def __init__(self, channel_size):
        super(ChannelNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is signal length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
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
    input_size = 64
    bottleneck_size = 96
    hidden_size = bottleneck_size
    conv_size = 128
    pyramid_size = 3
    P = 3
    X = 8
    R = 3
    C = 1
    rnn_type = 'GRU'
    L = 6

    net = TARNN(input_size, bottleneck_size, hidden_size, conv_size, pyramid_size, P, X, R, C, rnn_type=rnn_type,
                L=L, norm_type='cLN', causal=True)
    from src.evaluate import calc_num_params
    calc_num_params(net)
    x = torch.randn((2, 32000))
    y = net(x)

    # temp_pyr = Temporal_Pyramid(64, 128, 3, 8, 3, C=1)
    # x = torch.randn((2, 64, 32000 // 3))
    # y = temp_pyr(x)
    # print(len(y))
    # from src.evaluate import calc_num_params
    #
    # calc_num_params(temp_pyr)
