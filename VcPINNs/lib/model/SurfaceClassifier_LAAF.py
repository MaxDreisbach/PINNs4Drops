import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaptiveConv1d import AdaptiveConv1d


class SurfaceClassifier_LAAF(nn.Module):
    def __init__(self, filter_channels, temp_filters, num_views=1, no_residual=False, residual_tcn=False, last_op=None, LAAF_scale=2.0, LAAF_mode='layerwise'):
        super(SurfaceClassifier_LAAF, self).__init__()

        self.filters = []
        self.temp_filters = []
        self.num_views = num_views
        self.no_residual = no_residual
        self.residual_tcn = residual_tcn
        filter_channels = filter_channels
        temp_filters = temp_filters
        self.last_op = last_op
        self.LAAF_scale = LAAF_scale


        #TCN
        if not self.residual_tcn:
            print('Using TCN with outer skip connection to MLP and', self.num_views // 2, 'layers')
            for j in range(0, self.num_views // 2):
                self.temp_filters.append(nn.Conv1d(
                    in_channels=temp_filters[j],
                    out_channels=temp_filters[j + 1],
                    kernel_size=3,
                    padding=0))
                self.add_module("1D_temp_conv%d" % j, self.temp_filters[j])
        else:
            print('Using TCN with residual connections and', self.num_views // 2, 'layers')
            # padding to allow for concat with input features
            for j in range(0, self.num_views // 2):
                self.temp_filters.append(nn.Conv1d(
                    in_channels=temp_filters[j],
                    out_channels=temp_filters[j + 1],
                    kernel_size=3,
                    padding=1))
                self.add_module("1D_temp_conv%d" % j, self.temp_filters[j])


        # MLP
        if self.no_residual:
            print('Using MLP without skip connections')
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
            print('Using MLP with skip connections')
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
        y = im_feat
        tmpy = im_feat

        y = y.permute(2, 1, 0)
        tmpy_tcn = tmpy.permute(2, 1, 0)
        #print('reshaped image features for 1D temporal convolution layer ', y.size())

        # TCN layers -> aggregate spatial features in time
        for i, f in enumerate(self.temp_filters):
            y = self._modules["1D_temp_conv" + str(i)](y)

            if i % 2 == 1 and self.residual_tcn:
                # residual connection
                #print('residual connection at layer', str(i))
                y = y + tmpy_tcn
                y = torch.tanh(y)
                tmpy_tcn = y
            else:
                y = torch.tanh(y)

        y = y.permute(2, 1, 0)

        '''
        Stack spatio-temporal TCN features with spatial HG features of current frame 
        -> get accurate local spatial features for alpha and spatio-temporal features for u,v,w,p
        '''
        if self.residual_tcn:
            # only take center time step feature
            y = y[self.num_views // 2: self.num_views // 2 + 1, :, :]
            tmpy = tmpy[self.num_views // 2: self.num_views // 2 + 1, :, :]
        else:
            tmpy = tmpy[self.num_views // 2: self.num_views // 2 + 1, :, :]

        y = torch.cat([y, tmpy, x_feat, y_feat, z_feat, t_feat], 1)
        tmpy = y

        #print('input shape to MLP', y.size())

        # MLP layers -> compute predictions from spatio-temporal features
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

        if self.last_op:
            '''           
            Different activation functions for each output variable -> alpha -> sigmoid, (u,v,p) - None, p ->exponential
            (see Buhendwa et al. (2021) - https://doi.org/10.1016/j.mlwa.2021.100029)
            '''
            y = torch.cat(
                (nn.Sigmoid()(y[:, :1, :]), y[:, 1:2, :], y[:, 2:3, :], y[:, 3:4, :], torch.exp(y[:, 4:5, :])), dim=1)

        return y
