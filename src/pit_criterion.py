# Created on 2018/12
# Author: Kaituo XU

from itertools import permutations

import torch
import torch.nn.functional as F
from src.deepspeech_model import DeepSpeech
import scipy.signal
import torchaudio
from src.utils import arrange_batch, parse_audio

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths, device, features_model=None):
    """
    Args:
        source: [B, T], B is batch size
        estimate_source: [B, T]
        source_lengths: [B]
    """
    # max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
    #                                                   estimate_source,
    #                                                   source_lengths)
    # loss = 0 - torch.mean(max_snr)
    # reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    # return loss, max_snr, estimate_source, reorder_estimate_source
    assert features_model is not None
    loss = calc_deep_feature_loss(source, estimate_source, source_lengths, features_model, device)
    # loss2 = 0.0 - torch.mean(calc_si_sdr(source, estimate_source, source_lengths))
    import GPUtil
    GPUtil.showinitialization()
    return loss


def calc_si_sdr(source, estimate_source, source_lengths):
    """ SI-SDR for Speech Enhancement from paper https://arxiv.org/abs/1909.01019 """

    assert source.size() == estimate_source.size()
    B, T = source.size()

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1).float()  # [B, 1]
    mean_target = torch.sum(source, dim=1, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=1, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    #
    cross_energy = torch.sum(zero_mean_target * zero_mean_estimate, dim=1, keepdim=True)
    target_energy = torch.sum(zero_mean_target ** 2, dim=1, keepdim=True) + EPS
    estimate_energy = torch.sum(zero_mean_estimate ** 2, dim=1, keepdim=True) + EPS
    # si_sdr = 10 * torch.log10(cross_energy/ (target_energy * estimate_energy - cross_energy) + EPS)
    alpha = cross_energy / target_energy
    si_sdr = torch.sum((alpha * zero_mean_target) ** 2, dim=1, keepdim=True) / \
             torch.sum((alpha * zero_mean_target - zero_mean_estimate) ** 2, dim=1, keepdim=True)
    si_sdr = 10 * torch.log10(si_sdr)
    return si_sdr


def calc_deep_feature_loss(source, estimate_source, source_lengths, deep_features_model, device):
    """
    Calculates deep feature loss using the DeepSpeech2 ASR model.
    Code and model from: https://github.com/SeanNaren/deepspeech.pytorch
    Args:
        source: [B, T], B is batch size
        estimate_source: [B, T]
        source_lengths: [B], each item is between [0, T]
    """
    # TODO: Make sure that output sigal behaves like signal from librosa.load
    #   ie check Decoder output for clean signal
    B, T = source.size()
    # model_dir = "../egs/models/loss_models/librispeech_pretrained_v2.pth"
    # deep_features_model = DeepSpeech.load_model(model_dir)
    deep_features_model = deep_features_model.to(device)

    audio_conf = deep_features_model.audio_conf
    window_stride = audio_conf['window_stride']
    window_size = audio_conf['window_size']
    sample_rate = audio_conf['sample_rate']
    win_length =  int(sample_rate * window_size)
    windows = {'hamming':  torch.hamming_window(win_length).to(device), 'hann': torch.hann_window(win_length).to(device),
               'blackman': torch.blackman_window(win_length).to(device),
               'bartlett': torch.bartlett_window(win_length).to(device)}
    window = windows.get(audio_conf['window'], windows['hamming'])

    spect_source_list, spect_estimate_list = [], []
    for b in range(B):
        spect_source_list.append(parse_audio(source[b, :], sample_rate, window_size,
                                                         window_stride, window, device, normalize=True))
        spect_estimate_list.append(parse_audio(estimate_source[b, :], sample_rate, window_size,
                                                           window_stride, window, device, normalize=True))

    batch_estimate, batch_estimate_sizes = arrange_batch(spect_estimate_list, device)
    batch_source, batch_source_sizes = arrange_batch(spect_source_list, device )

    features_source, _ = deep_features_model(batch_source, batch_source_sizes)
    features_estimate, _ = deep_features_model(batch_estimate, batch_estimate_sizes)

    mse_loss = torch.nn.MSELoss()
    return mse_loss(features_estimate, features_source)


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    # s_target here is the real wave, not the s_target from the paper (S from paper)
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot.float()])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 2, 3, 32000  # In this case B is the number of batches
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    source_lengths = torch.LongTensor([T, T - 3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)

    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)
