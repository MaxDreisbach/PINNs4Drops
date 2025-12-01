import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConv1d(nn.Conv1d):
    """
    Applies a 1D convolution with layer-wise adaptive activation functions.
    Based on:
    Jagtap, A. D. et al. Locally adaptive activation functions with slope recovery
    for deep and physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple
        Size of the convolving kernel
    stride : int or tuple, optional
        Stride of the convolution. Default: 1
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: True
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function.
    adaptive_rate_scaler : float, optional
        Fixed scaling factor for adaptive activation functions.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, adaptive_rate=0.5, adaptive_rate_scaler=2.0, mode='layerwise'):
        super(AdaptiveConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             padding, dilation, groups, bias)

        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if mode == 'layerwise':
            # L-LAAF
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(1, 1, 1))
        elif mode == 'neuronwise':
            # N-LAAF
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(1, in_channels, 1))
        else:
            # default: no trainable activation
            self.A = self.adaptive_rate



    def forward(self, input):
        if self.adaptive_rate:
            #print('self.A: ', self.A)
            input = self.adaptive_rate_scaler * self.A * input
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return (super().extra_repr() + f', adaptive_rate={self.adaptive_rate is not None}, '
                                       f'adaptive_rate_scaler={self.adaptive_rate_scaler}')