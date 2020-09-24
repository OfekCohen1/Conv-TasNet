import torch
import torch.nn as nn
from torch_stft import STFT
import math
import torch.nn.functional as functional
from torch.autograd import Variable

EPS = 1e-8


# TODO: Write shapes of every input/output in modules

class DCCRN(nn.Module):
    def __init__(self, fft_length, window_length, hop_size, window, num_convs, enc_channel_list, dec_channel_list,
                 freq_kernel_size, time_kernel_size, stride, dilation, norm_type, rnn_type, num_layers_rnn, mask_type):
        super(DCCRN, self).__init__()
        self.window_length = window_length
        self.fft_length = fft_length
        self.hop_size = hop_size
        self.window = window
        self.mask_type = mask_type

        self.stft_obj = STFT(self.fft_length, self.hop_size, self.window_length, self.window)
        self.encoder = Encoder(num_convs, enc_channel_list, freq_kernel_size, time_kernel_size, stride,
                               norm_type, dilation=dilation)
        channel_size_decoder = []
        self.decoder = Decoder(num_convs, dec_channel_list, freq_kernel_size, time_kernel_size, stride, norm_type,
                               dilation=dilation)
        # fft_length/2 is freq size. pow(2,num_convs) is after downsampling in freq axis due to stride
        freq_size_dilated = int((fft_length / 2) / pow(2, num_convs))
        rnn_hidden_size = int(enc_channel_list[-1] * 2)  # downsample to size of channels
        rnn_input_size = rnn_hidden_size * freq_size_dilated
        self.separator = Separator(rnn_type, num_layers_rnn, rnn_input_size, rnn_hidden_size)


    def forward(self, waveform):
        # TODO: Check difference between Conv-STFT and torch.stft
        stft_mag, stft_phase = self.stft_obj.transform(waveform)
        stft_mag_dc, stft_phase_dc = stft_mag[:, 0:1, :], stft_phase[:, 0:1, :]
        stft_mag_no_dc, stft_phas_no_dc = stft_mag[:, 1:, :], stft_phase[:, 1:, :]  # Remove DC
        stft_real, stft_imag = stft_mag_no_dc * torch.cos(stft_phas_no_dc), stft_mag_no_dc * torch.sin(stft_phas_no_dc)
        stft_real, stft_imag = stft_real.unsqueeze(1), stft_imag.unsqueeze(1)

        encoded_real, encoded_imag, skip_list_real, skip_list_imag = self.encoder(stft_real, stft_imag)
        separated_real, separated_imag = self.separator(encoded_real, encoded_imag)
        mask_real, mask_imag = self.decoder(separated_real, separated_imag, skip_list_real, skip_list_imag)

        decoded_mag, decoded_phase = get_mask(mask_real, mask_imag, stft_real, stft_imag, self.mask_type)
        decoded_mag, decoded_phase = decoded_mag.squeeze(1), decoded_phase.squeeze(1)
        decoded_mag = torch.cat((stft_mag_dc, decoded_mag), dim=1)
        decoded_phase = torch.cat((stft_phase_dc, decoded_phase), dim=1)
        clean_waveform = self.stft_obj.inverse(decoded_mag.squeeze(1), decoded_phase.squeeze(1))
        return clean_waveform

def get_mask(mask_real, mask_imag, input_real, input_imag, mask_type):
    """ See paper to understand what each mask type means """
    if mask_type == 'E':
        mask_mag = torch.tanh(torch.sqrt(torch.pow(mask_real, 2) + torch.pow(mask_imag, 2)))
        mask_phase = torch.atan2(mask_imag, mask_real)
        input_mag = torch.sqrt(torch.pow(input_real, 2) + torch.pow(input_imag, 2))
        input_phase = torch.atan2(input_imag, input_real)

        output_mag = input_mag * mask_mag
        output_phase = mask_phase + input_phase
        return output_mag, output_phase

    elif mask_type == 'C':
        output_real = mask_real * input_real - mask_imag * input_imag
        output_imag = mask_imag * input_real + mask_real * input_imag

        output_mag = torch.sqrt(torch.pow(output_real, 2) + torch.pow(output_imag, 2))
        output_phase = torch.atan2(output_real, output_imag)
        return output_mag, output_phase

    elif mask_type == 'R':
        output_real = mask_real * input_real
        output_imag = mask_imag * input_imag

        output_mag = torch.sqrt(torch.pow(output_real, 2) + torch.pow(output_imag, 2))
        output_phase = torch.atan2(output_real, output_imag)
        return output_mag, output_phase
    else:
        print('Type Mask not supported or isn''t in original paper')
        assert 1 == 0


class Encoder(nn.Module):
    def __init__(self, num_convs, channel_sizes_encoder, freq_kernel_size, time_kernel_size,
                 stride, norm_type, dilation=1):
        super(Encoder, self).__init__()
        self.num_convs = num_convs
        self.conv_blocks = nn.ModuleList()
        # First conv has a input with one channel
        self.conv_blocks.append(ComplexConvNormAct(1, channel_sizes_encoder[0], freq_kernel_size, time_kernel_size,
                                                   stride, norm_type, dilation=dilation))
        for i in range(num_convs - 1):
            self.conv_blocks.append(ComplexConvNormAct(channel_sizes_encoder[i], channel_sizes_encoder[i + 1],
                                                       freq_kernel_size, time_kernel_size, stride, norm_type,
                                                       dilation=dilation))

    def forward(self, input_real, input_imag):
        skip_list_real = []  # used for skip connections
        skip_list_imag = []
        output_real = input_real
        output_imag = input_imag
        for i in range(self.num_convs):
            output_real, output_imag = self.conv_blocks[i](output_real, output_imag)
            skip_list_real.append(output_real)
            skip_list_imag.append(output_imag)

        skip_list_real.reverse(), skip_list_imag.reverse()  # Reverse the order for decoder
        return output_real, output_imag, skip_list_real, skip_list_imag


class Decoder(nn.Module):
    def __init__(self, num_convs, channel_sizes_decoder, freq_kernel_size, time_kernel_size,
                 stride, norm_type, dilation=1):
        super(Decoder, self).__init__()
        self.num_convs = num_convs
        self.conv_transp_blocks = nn.ModuleList()
        # Channel output dimension is half of the next item in the list because of skip connections
        for i in range(self.num_convs - 1):
            self.conv_transp_blocks.append(
                ComplexConvTransposedNormAct(channel_sizes_decoder[i], channel_sizes_decoder[i + 1] // 2,
                                             freq_kernel_size,
                                             time_kernel_size, stride, norm_type, dilation=dilation))
        self.conv_transp_blocks.append(
            ComplexConvTransposedNormAct(channel_sizes_decoder[-1], 1, freq_kernel_size, time_kernel_size, stride,
                                         norm_type, dilation=dilation))

    def forward(self, input_real, input_imag, skip_list_real, skip_list_imag):
        output_real = input_real
        output_imag = input_imag
        for i in range(self.num_convs):
            output_real_cat = torch.cat((output_real, skip_list_real[i]), dim=1)
            output_imag_cat = torch.cat((output_imag, skip_list_imag[i]), dim=1)
            output_real, output_imag = self.conv_transp_blocks[i](output_real_cat, output_imag_cat)
        return output_real, output_imag


class Separator(nn.Module):
    def __init__(self, rnn_type, num_layers, rnn_input_size, rnn_hidden_size):
        super(Separator, self).__init__()
        self.rnn = getattr(nn, rnn_type)(rnn_input_size, rnn_hidden_size, num_layers, batch_first=True,
                                         bidirectional=False)
        self.dense = nn.Linear(rnn_hidden_size, rnn_input_size)

    def forward(self, encoded_real, encoded_imag):
        #  Concat real and imaginary, currently no complex LSTM
        concat_input = torch.cat((encoded_real, encoded_imag), dim=1)
        B, N, F, T = concat_input.shape
        concat_input = concat_input.view(B, -1, T).permute(0, 2, 1)
        output_separator = self.dense(self.rnn(concat_input)[0]).permute(0, 2, 1)
        output_separator = output_separator.view(B, N, F, T)
        separated_real, separated_imag = output_separator[:, :N // 2, :, :], output_separator[:, N // 2:, :, :]
        return separated_real, separated_imag


class ComplexConvNormAct(nn.Module):
    def __init__(self, input_size, output_size, freq_kernel_size, time_kernel_size,
                 stride, norm_type, dilation=1):
        """
        :param input_size: number of channels in layer input
        :param output_size: number of channels in layer output
        :param norm_type: real_BN or complex_BN
        :param dilation: defualt is 1
        """
        super(ComplexConvNormAct, self).__init__()
        self.conv_real = CausalConv(input_size, output_size, freq_kernel_size, time_kernel_size, stride, dilation)
        self.conv_imag = CausalConv(input_size, output_size, freq_kernel_size, time_kernel_size, stride, dilation)
        self.norm = choose_norm(norm_type, output_size)
        self.act_real = nn.PReLU()
        self.act_imag = nn.PReLU()

    def forward(self, input_real, input_imag):
        output_real = self.conv_real(input_real) - self.conv_imag(input_imag)
        output_imag = self.conv_real(input_imag) + self.conv_imag(input_real)
        output_real, output_imag = self.norm(output_real, output_imag)
        output_real = self.act_real(output_real)
        output_imag = self.act_imag(output_imag)
        return output_real, output_imag


class ComplexConvTransposedNormAct(nn.Module):
    def __init__(self, input_size, output_size, freq_kernel_size, time_kernel_size,
                 stride, norm_type, dilation=1):
        """
        :param input_size: number of channels in layer input
        :param output_size: number of channels in layer output
        :param norm_type: real_BN or complex_BN
        :param dilation: defualt is 1
        """
        super(ComplexConvTransposedNormAct, self).__init__()
        self.conv_transp_real = SemiCausalConvTranspose(input_size, output_size, freq_kernel_size, time_kernel_size,
                                                        stride, dilation)
        self.conv_transp_imag = SemiCausalConvTranspose(input_size, output_size, freq_kernel_size, time_kernel_size,
                                                        stride, dilation)
        self.norm = choose_norm(norm_type, output_size)
        self.act_real = nn.PReLU()
        self.act_imag = nn.PReLU()

    def forward(self, input_real, input_imag):
        output_real = self.conv_transp_real(input_real) - self.conv_transp_imag(input_imag)
        output_imag = self.conv_transp_real(input_imag) + self.conv_transp_imag(input_real)
        output_real, output_imag = self.norm(output_real, output_imag)
        output_real = self.act_real(output_real)
        output_imag = self.act_imag(output_imag)
        return output_real, output_imag


class CausalConv(nn.Module):
    """Causal Convolution block."""

    def __init__(self, input_size, output_size, freq_kernel_size, time_kernel_size,
                 stride, dilation=1):
        super(CausalConv, self).__init__()
        padding_freq = (freq_kernel_size - 1) * dilation // 2  # Freq axis is not causal -> half padding and no chomp
        padding_time = (time_kernel_size - 1) * dilation
        conv2d = nn.Conv2d(input_size, output_size, (freq_kernel_size, time_kernel_size), stride,
                           (padding_freq, padding_time), dilation)
        chomp_time = Chomp2d(padding_time, 'conv')
        self.causal_conv = nn.Sequential(conv2d, chomp_time)

    def forward(self, input_tensor):
        return self.causal_conv(input_tensor)


class SemiCausalConvTranspose(nn.Module):
    """semi Causal Transposed Convolution block. looks one frame ahead"""

    def __init__(self, input_size, output_size, freq_kernel_size, time_kernel_size,
                 stride, dilation=1):
        super(SemiCausalConvTranspose, self).__init__()
        padding_freq = (freq_kernel_size - 1) * dilation // 2  # Freq axis is not causal -> half padding and no chomp
        padding_time = (time_kernel_size - 1) * dilation // 2  # In convTranspose pad half (semi causal, not causal)
        chomp_size = (time_kernel_size - 1) * dilation  # cut 1 frame
        output_padding = (1, 0)  # Fix output dimensions. only needed if we remove DC
        conv_trans_2d = nn.ConvTranspose2d(input_size, output_size, (freq_kernel_size, time_kernel_size), stride,
                                           (padding_freq, padding_time), output_padding, dilation=dilation)
        chomp_time = Chomp2d(chomp_size, 'deconv')
        self.semi_causal_conv_transpose = nn.Sequential(conv_trans_2d, chomp_time)

    def forward(self, input_tensor):
        return self.semi_causal_conv_transpose(input_tensor)


def choose_norm(norm_type, channel_size):
    # if norm_type == "complex_BN":
    #     return ComplexBatchNorm  # TODO: Implement Complex BN
    assert norm_type == 'real_BN'
    return RealBatchNorm(channel_size)


class RealBatchNorm(nn.Module):
    """ Regular Batch norm on each of the inputs
    Batch norm in train mode isn't causal - maybe a problem"""

    def __init__(self, channel_size):
        super(RealBatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(channel_size)

    def forward(self, real_input, imag_input):
        return self.batch_norm(real_input), self.batch_norm(imag_input)


class Chomp2d(nn.Module):
    """To ensure the output length is the same as the input. only chomps in time axis
    if causal - chomp from the front
    if semi causal (one frame look_ahead) chomp from the beginning"""

    def __init__(self, chomp_size, conv_type='conv'):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size
        self.conv_type = conv_type

    def forward(self, x):
        """
        Args:
            x: [B, N, F, T_padded]
        Returns:
            [B, N, F, T]
        """
        if self.conv_type == 'conv':
            return x[:, :, :, :-self.chomp_size].contiguous()
        else:
            assert self.conv_type == 'deconv'
        return x[:, :, :, self.chomp_size:].contiguous()


if __name__ == '__main__':
    from src.evaluate import calc_num_params

    torch.manual_seed(123)
    B, T = 4, 32000
    sample_rate = 16000
    enc_list = [16, 32, 64, 64, 128, 128]
    dec_list = [x * 2 for x in enc_list]
    dec_list.reverse()
    num_convs = 6

    fft_length = 512
    hop_size = int(6.25e-3 * sample_rate)
    window_length = int(25e-3 * sample_rate)
    window = 'hann'

    waveform = torch.randn((B, T)).float()
    stft_real_imag = torch.stft(waveform, fft_length, hop_size, window_length)
    stft_real, stft_imag = stft_real_imag[:, :, :, 0], stft_real_imag[:, :, :, 1]
    stft_real = stft_real[:, 1:, :].unsqueeze(1)  # Remove DC
    stft_imag = stft_imag[:, 1:, :].unsqueeze(1)  # Remove DC

    # enc = Encoder(6, enc_list, 5, 2, (2, 1), 'real_BN')
    # enc.eval()
    # output_real, output_imag, out_real_list, out_imag_list = enc(stft_real, stft_imag)
    # stft_real[:, :, :, 100:] = 0
    # output_real2, output_imag2, out_real_list2, out_imag_list2 = enc(stft_real, stft_imag)
    # print(stft_real.shape)
    # print(output_real.shape)
    # print(output_real2.shape)
    # print(torch.equal(output_real[:, :, :, 0:100], output_real2[:, :, :, 0:100]))

    # dec = Decoder(6, dec_list, 5, 2, (2, 1), 'real_BN')
    # dec.eval()
    # dec_real, dec_imag = dec(output_real2, output_imag2, out_real_list2, out_imag_list2)
    # output_real2[:, :, :, 100:] = 0
    # dec_real2, dec_imag2 = dec(output_real2, output_imag2, out_real_list2, out_imag_list2)
    # print(torch.equal(dec_real[:, :, :, 0:95], dec_real2[:, :, :, 0:95]))
    # print(stft_real.shape)
    # print(dec_real.shape)
    #

    # calc_num_params(enc)
    # calc_num_params(dec)

    model = DCCRN(fft_length, window_length, hop_size, window, num_convs, enc_list, dec_list, 5, 2, (2, 1), 1,
                  'real_BN', 'LSTM', 2, 'E')
    calc_num_params(model)
    bla2 = model(waveform)

    # # Test causality of conv layer
    # conv_layer = CausalConv(1, 32, 5, 2, (2, 1))
    # conv2_layer = CausalConv(32, 64, 5, 2, (2, 1))
    # bla = conv2_layer(conv_layer(stft_real))
    # stft_real[:, :, :, 100:] = 0
    # bla2 = conv2_layer(conv_layer(stft_real))
    # print(torch.equal(bla[:, :, :, 0:100], bla2[:, :, :, 0:100]))

    # # Test semi causality of Transposed Convolution
    # deconv_layer = SemiCausalConvTranspose(32, 1, 5, 2, (2, 1))
    # bla = conv_layer(stft_real)
    # bla3 = deconv_layer(bla)
    # print(stft_real.shape)
    # print(bla3.shape)
    # bla[:, :, :, 100:] = 0
    # bla4 = deconv_layer(bla)
    # print(torch.equal(bla3[:, :, :, 0:99], bla4[:, :, :, 0:99]))
