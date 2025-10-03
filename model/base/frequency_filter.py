import torch
import torch.nn as nn


class Frequency_filter(nn.Module):
    """Lightweight Amplitude-Phase Masker (APM)"""

    def __init__(self, shape, reduction=8):

        super().__init__()
        print("#####################")

        bsz, channel, h, w = shape[0], shape[1], shape[2], shape[3]
        mask_shape = (bsz, channel, h, w)
        self.channel = channel
        self.mask_amplitude = nn.Parameter(torch.ones(mask_shape))
        self.mask_phase = nn.Parameter(torch.ones(mask_shape))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.phase_attn = nn.Sequential(
            nn.Linear(channel, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        # Extract amplitude and phase information
        freq_domain = torch.fft.fftshift(torch.fft.fft2(x))
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)

        # AM and PM
        mask_amplitude = torch.sigmoid(self.mask_amplitude)  # Constrain the value of masker to [0,1]
        mask_phase = torch.sigmoid(self.mask_phase)
        adjusted_amplitude = mask_amplitude * amplitude
        adjusted_phase = mask_phase * phase
        adjusted_freq = torch.polar(adjusted_amplitude, adjusted_phase)
        adjusted_x = torch.fft.ifft2(torch.fft.ifftshift(adjusted_freq)).real

        # Channel attention
        pooled_phase = self.avg_pool(phase).view(x.size(0), self.channel)
        phase_weights = self.phase_attn(pooled_phase).view(x.size(0), self.channel, 1, 1)

        # phase channel weighting
        enhanced_x = adjusted_x * phase_weights.expand_as(x)

        return enhanced_x
