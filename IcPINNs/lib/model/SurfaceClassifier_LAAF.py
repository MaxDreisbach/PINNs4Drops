import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaptiveConv1d import AdaptiveConv1d


class SurfaceClassifier_LAAF(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None, LAAF_scale=2.0, LAAF_mode='layerwise'):
        super(SurfaceClassifier_LAAF, self).__init__()

        self.filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op
        self.LAAF_scale = LAAF_scale

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(AdaptiveConv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1,
                    adaptive_rate=1 / self.LAAF_scale,
                    adaptive_rate_scaler=self.LAAF_scale,
                    mode=LAAF_mode
                ))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            print('using skip connections in MLP')
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(AdaptiveConv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1,
                        adaptive_rate=1 / self.LAAF_scale,
                        adaptive_rate_scaler=self.LAAF_scale,
                        mode=LAAF_mode
                    ))
                else:
                    self.filters.append(AdaptiveConv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1,
                        adaptive_rate=1 / self.LAAF_scale,
                        adaptive_rate_scaler=self.LAAF_scale,
                        mode=LAAF_mode
                    ))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, im_feat, x_feat, y_feat, z_feat, t_feat):
        '''
        calculates forward pass through neural network for sampled point coordinate, time and pixel-aligned feature vector
        inputs: image feature vector, x, y, z, t
        outputs: alpha, u, v, w, p
        '''
        y = torch.cat([im_feat, x_feat, y_feat, z_feat, t_feat], 1)
        tmpy = torch.cat([im_feat, x_feat, y_feat, z_feat, t_feat], 1)
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = torch.tanh(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                #TODO: tmpy should not average the coordinates x,y,z,t; but instead take the coordinates of the centre step (however this should not be relevant as the calibs are expected to be the same, thus x,y,z,t are the same of all input image-coordinate pairs)
                tmpy = tmpy.view(
                    -1, self.num_views, tmpy.shape[1], tmpy.shape[2]
                ).mean(dim=1)

        if self.last_op:
            '''           
            Different activation functions for each output variable -> alpha -> sigmoid, (u,v,p) - None, p ->exponential
            (see Buhendwa et al. (2021) - https://doi.org/10.1016/j.mlwa.2021.100029)
            '''
            y = torch.cat(
                (nn.Sigmoid()(y[:, :1, :]), y[:, 1:2, :], y[:, 2:3, :], y[:, 3:4, :], torch.exp(y[:, 4:5, :])), dim=1)

        return y
