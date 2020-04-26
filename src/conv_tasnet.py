# Created on 2018/12
# Author: Kaituo XU

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.utils import overlap_and_add

EPS = 1e-8


class ConvTasNet(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
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
        super(ConvTasNet, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
        self.decoder = Decoder(N, L)
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
        est_mask = self.separator(mixture_w)  # M x C x N x K
        est_source = self.decoder(mixture_w, est_mask)  # M x C x T_conv

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = functional.pad(est_source, (0, T_origin - T_conv))
        return est_source

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['L'], package['B'], package['H'],
                    package['P'], package['X'], package['R'], package['C'],
                    norm_type=package['norm_type'], causal=package['causal'],
                    mask_nonlinear=package['mask_nonlinear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'N': model.N, 'L': model.L, 'B': model.B, 'H': model.H,
            'P': model.P, 'X': model.X, 'R': model.R, 'C': model.C,
            'norm_type': model.norm_type, 'causal': model.causal,
            'mask_nonlinear': model.mask_nonlinear,
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
        # TODO: Might not need relu for encoder
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


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
                 mask_nonlinear='relu'):
        """
        Args:
            N: Number of filters in autoencoder
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
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        # TODO: I changed here from layer norm to global norm
        self.layer_norm = chose_norm(norm_type, N)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        # self.temporal_conv_blocks = []  # List of conv1D blocks
        self.temporal_conv_blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                # self.temporal_conv_blocks += [TemporalBlock(B, H, P, stride=1,
                #                                             padding=padding,
                #                                             dilation=dilation,
                #                                             norm_type=norm_type,
                #                                             causal=causal)]
                self.temporal_conv_blocks.append(TemporalBlock(B, H, P, stride=1,
                                                            padding=padding,
                                                            dilation=dilation,
                                                            norm_type=norm_type,
                                                            causal=causal))
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        # [M, N, K]
        M, N, K = mixture_w.size()
        mixture = self.layer_norm(mixture_w)
        mixture = self.bottleneck_conv1x1(mixture)
        skip_connections = 0
        for conv_block in self.temporal_conv_blocks:
            mixture, skip = conv_block(mixture)
            skip_connections += skip
            # mixture = conv_block(mixture)
        mixture = mixture + skip_connections
        score = self.mask_conv1x1(mixture)  # now it's  [M, C*N, K]

        score = score.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        # TODO: Might want to change to sigmoid here
        if self.mask_nonlinear == 'softmax':
            est_mask = functional.softmax(score, dim=1)
        elif self.mask_nonlinear == 'relu':
            est_mask = functional.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    """ Returns (output, skip connection) """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        dsconv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # [M, H, K] -> [M, B, K]
        # I assume that B = Sc for simplicity
        conv1x1_output = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        conv1x1_sc = nn.Conv1d(out_channels, in_channels, 1, bias=False)

        # Put together
        self.output_block = nn.Sequential(dsconv, conv1x1_output)
        self.skip_connection_block = nn.Sequential(dsconv, conv1x1_sc)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.output_block(x)
        out = out + residual
        skip_connection = self.skip_connection_block(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out, skip_connection  # look like w/o F.relu is better than w/ F.relu
        # return out
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, B, K] -> [M, H, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        if causal:
            chomp = Chomp1d(padding)
        pointwise_prelu = nn.PReLU()
        depthwise_prelu = nn.PReLU()
        pointwise_norm = chose_norm(norm_type, out_channels)
        depthwise_norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=out_channels,
                                   bias=False)
        # Put together
        # TODO: Not sure where to chomp here, since i change the order of stuff (causal might not work)
        if causal:
            self.net = nn.Sequential(chomp, pointwise_conv, pointwise_prelu, pointwise_norm,
                                     depthwise_conv, depthwise_prelu, depthwise_norm)
        else:
            self.net = nn.Sequential(pointwise_conv, pointwise_prelu, pointwise_norm,
                                     depthwise_conv, depthwise_prelu, depthwise_norm)

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
        return ChannelwiseLayerNorm(channel_size)
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
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
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


if __name__ == "__main__":
    torch.manual_seed(123)
    M, N, L, T = 4, 256, 20, 32000
    K = 2 * T // L - 1
    B, H, P, X, R, C, norm_type, causal = 256, 512, 3, 8, 4, 2, "gLN", False
    mixture = torch.randint(3, (M, T))
    # test Encoder
    encoder = Encoder(L, N)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    mixture_w = encoder(mixture)
    print('mixture', mixture)
    print('U', encoder.conv1d_U.weight)
    print('mixture_w', mixture_w)
    print('mixture_w size', mixture_w.size())

    # test TemporalConvNet
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print('est_mask', est_mask)

    # test Decoder
    decoder = Decoder(N, L)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask)
    print('est_source', est_source)

    # test Conv-TasNet
    conv_tasnet = ConvTasNet(N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture)
    print('est_source', est_source)
    print('est_source size', est_source.size())
