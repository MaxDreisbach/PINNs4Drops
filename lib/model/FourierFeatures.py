import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739

    Given an input of size [batches, num_input_channels, num_samples],
     returns a tensor of size [batches, mapping_size*2, num_samples].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        #self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.register_buffer('_B', torch.randn((num_input_channels, mapping_size)) * scale)

    def forward(self, x):
        assert x.dim() == 3, 'Expected 3D input (got {}D input)'.format(x.dim())

        batches, channels, num_samples = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, N] to [(B*N), C].
        x = x.squeeze(0)
        x = x.swapaxes(0, 1)
        #print('xyz before matmul: ', x.size())
        #print('B: ', self._B.size())

        x = x @ self._B.to(x.device)
        #print('xyz after matmul: ', x.size())
        #print('xyz after matmul: ', x)

        # From [(B*N), C] to [B, C, N]
        x = x.swapaxes(0, 1)
        x = x.unsqueeze(0)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)