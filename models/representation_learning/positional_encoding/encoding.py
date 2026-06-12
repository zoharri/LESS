"""
Positional encoding utilities for representation learning.

This module provides the FreqEncoder_torch class, which implements frequency-based
positional encoding. This encoding is used to inject spatial information into
neural network representations, allowing models to better understand spatial
relationships in the input data.
"""

import torch
from torch import nn


class FreqEncoder_torch(nn.Module):
    """
    Frequency-based positional encoder for spatial information.
    
    This module implements positional encoding using sinusoidal functions at
    different frequencies. It is commonly used in neural networks to encode
    spatial or temporal information into vector representations.
    
    The encoder can be configured with different frequency bands and can optionally
    include the original input in the output.
    """

    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns
        self.N_freqs = N_freqs

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, max_level=None, **kwargs):

        if max_level is None:
            max_level = self.N_freqs
        else:
            max_level = int(max_level * self.N_freqs)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(max_level):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        # append 0
        if self.N_freqs - max_level > 0:
            out.append(
                torch.zeros(*input.shape[:-1], (self.N_freqs - max_level) * 2 * input.shape[-1], device=input.device,
                            dtype=input.dtype))

        out = torch.cat(out, dim=-1)

        return out
