import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaptiveLinear import AdaptiveLinear


class SurfaceClassifier_LAAF(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier_LAAF, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(AdaptiveLinear(
                    filter_channels[l],
                    filter_channels[l + 1],
                    adaptive_rate=0.1,
                    adaptive_rate_scaler=10.0))

                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(AdaptiveLinear(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        adaptive_rate=0.1,
                        adaptive_rate_scaler=10.0))
                else:
                    self.filters.append(AdaptiveLinear(
                        filter_channels[l],
                        filter_channels[l + 1],
                        adaptive_rate=0.1,
                        adaptive_rate_scaler=10.0))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, im_feat, x_feat, y_feat, z_feat, t_feat):
        '''
        calculates forward pass through neural network for sampled point coordinate, time and pixel-aligned feature vector
        inputs: image feature vector, x, y, z, t
        outputs: alpha, u, v, w, p
        '''
        # print('image network input size: ', im_feat.size())
        # print('x network input size: ', x_feat.size())
        # print('t network input size: ', t_feat.size())
        y = torch.cat([im_feat, x_feat, y_feat, z_feat, t_feat], 1)
        tmpy = torch.cat([im_feat, x_feat, y_feat, z_feat, t_feat], 1)
        y = y.squeeze(0)
        y = y.swapaxes(0, 1)
        tmpy = tmpy.squeeze(0)
        tmpy = tmpy.swapaxes(0, 1)
        #print('MLP input shape: ', y.size())
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                # y = F.leaky_relu(y)
                ''' Changed to tanh activation function for PINN'''
                # TODO: sine or GELU activation
                y = torch.tanh(y)
                # y = nn.GELU(y)
                # y = torch.sin(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = y.swapaxes(0, 1)
            y = y.unsqueeze(0)
            '''
            Normalisation of the labels for (u,v,w,p) 
            -> all outputs of MLP can have sigmoid activation function as labels are normalised to [0, 1]
            -> sigmoid forces output to be either 0 or 1 -> linear layer (no activation for velocity and exponential for p)
            
            Different activation functions for each output variable -> alpha -> sigmoid, (u,v,p) - None, p ->exponential
            (see Buhendwa et al. (2021) - https://doi.org/10.1016/j.mlwa.2021.100029)
            '''
            y = torch.cat(
                (nn.Sigmoid()(y[:, :1, :]), y[:, 1:2, :], y[:, 2:3, :], y[:, 3:4, :], torch.exp(y[:, 4:5, :])), dim=1)
            # print('MLP output: ', y)
            # print('MLP output shape: ', y.size())

        return y
