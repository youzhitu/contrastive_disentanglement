""" feature processing """

import torch
import torch.nn as nn
import numpy as np
import torchaudio


class FeatureExtractionLayer(nn.Module):
    def __init__(self, feat_dim=80, spec_aug=False):
        super().__init__()
        self.mel_fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600,
            window_fn=torch.hamming_window, n_mels=feat_dim)

        self.spec_aug = spec_aug

        if self.spec_aug:
            self.spec_augment = SpecAugment(time_mask_len_max=5, freq_mask_len_max=8, mix=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, wav_len], wav_len is the No. of sample points
        Returns:
            Tensor, [batch_size, freq_dim, seq_len], seq_len is the No. of frames
        """

        with torch.no_grad():
            x = self.mel_fbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=2, keepdim=True)
            # x = self.instancenorm(x)

            if self.spec_aug:
                x = self.spec_augment(x)

        return x


class SpecAugment(nn.Module):
    def __init__(self, time_mask_len_max=5, freq_mask_len_max=10, mix=False, n_time_masks=1, n_freq_masks=1):
        super().__init__()
        self.time_mask_len_max = time_mask_len_max
        self.freq_mask_len_max = freq_mask_len_max
        self.mix = mix
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def mask_along_dim(self, x, dim):
        """
        Args:
            x: Tensor, [batch_size, freq_dim, seq_len]
            dim: int, 1 for freq dimension or 2 for temporal dimension
        Returns:
            x: Tensor, [batch_size, freq_dim, seq_len]
        """

        x_size = x.size()
        mask_len_max = self.freq_mask_len_max if dim == 1 else self.time_mask_len_max

        mask_len = torch.randint(1, mask_len_max, (1,), device=x.device).item()
        start = torch.randint(0, x_size[dim] - mask_len, (x_size[0], 1), device=x.device)
        mask_range = torch.arange(x_size[dim], device=x.device)
        mask = torch.logical_and(mask_range >= start, mask_range < (start + mask_len))
        mask = mask.unsqueeze(len(x_size) - dim)

        if self.mix:
            return ~mask * x + mask * (x + x[torch.randperm(x_size[0])]) / 2
        return x.masked_fill(mask, 0.)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, freq_dim, seq_len]
        Returns:
            x: Tensor, [batch_size, freq_dim, seq_len]
        """

        for _ in range(self.n_time_masks):
            x = self.mask_along_dim(x, dim=2)

        for _ in range(self.n_freq_masks):
            x = self.mask_along_dim(x, dim=1)

        return x
