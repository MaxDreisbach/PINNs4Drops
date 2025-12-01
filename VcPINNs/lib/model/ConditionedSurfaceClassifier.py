import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedSurfaceClassifier(nn.Module):
    def __init__(self):
        super(ConditionedSurfaceClassifier, self).__init__()

        in_dim = 4
        out_dim = 5
        self.hidden_dim = 128
        self.feat_dim = 256

        self.hl1 = nn.Linear(in_dim, self.hidden_dim)
        self.hl2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hl3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, out_dim)

        self.activation = nn.GELU()
        #self.activation = torch.sin()

        # 2 * hidden layer output size for scale and shift
        self.layer_phi1 = nn.Linear(self.feat_dim, 2 * self.hidden_dim)
        self.layer_phi2 = nn.Linear(self.feat_dim, 2 * self.hidden_dim)
        self.layer_phi3 = nn.Linear(self.feat_dim, 2 * self.hidden_dim)


    def forward(self, im_feat, x_feat, y_feat, z_feat, t_feat):
        ''' Conditioned MLP
        - input: x, y, z, t + image features from Hourglas network
        - output: predicted alpha, u, v, w, p fields in 3D
        The MLP input layer receives x, y, z and t, the image features are used as conditional input at all hidden layers.
        The conditional input is used to scale and shift the output of the hidden layers before activation and is weighted
        linearly (layer_phi_i) before being applied to the output (possible extension by applying an activation after linear layers phi)
        References: Vivek Oommen, Crunch Group, Brown University
        CORAL - https://arxiv.org/pdf/2306.07266.pdf
        INFINITY - https://arxiv.org/pdf/2307.13538.pdf
        '''

        im_feat = torch.squeeze(im_feat, 0).reshape(-1, self.feat_dim)
        x_feat = torch.squeeze(x_feat, 0).reshape(-1, 1)
        y_feat = torch.squeeze(y_feat, 0).reshape(-1, 1)
        z_feat = torch.squeeze(z_feat, 0).reshape(-1, 1)
        t_feat = torch.squeeze(t_feat, 0).reshape(-1, 1)
        #print('image network input size: ', im_feat.size())
        #print('x network input size: ', x_feat.size())
        #print('t network input size: ', t_feat.size())
        y = torch.cat([x_feat, y_feat, z_feat, t_feat], 1)

        phi1 = self.layer_phi1(im_feat)
        # get output from hidden layer and apply scale and shift from phi
        o1 = self.hl1(y) * phi1[:, :self.hidden_dim] + phi1[:, self.hidden_dim:]
        #a1 = self.activation(o1)
        a1 = torch.sin(o1)

        phi2 = self.layer_phi2(im_feat)
        o2 = self.hl2(a1) * phi2[:, :self.hidden_dim] + phi2[:, self.hidden_dim:]
        #a2 = self.activation(o2)
        a2 = torch.sin(o2)

        phi3 = self.layer_phi3(im_feat)
        o3 = self.hl3(a2) * phi3[:, :self.hidden_dim] + phi3[:, self.hidden_dim:]
        #a3 = self.activation(o3)
        a3 = torch.sin(o3)

        o4 = self.out(a3)
        output = torch.cat((nn.Sigmoid()(o4[:, :1]), o4[:, 1:2], o4[:, 2:3], o4[:, 3:4], torch.exp(o4[:, 4:5])),
                      dim=1)
        #output = torch.cat((o4[:, :1], o4[:, 1:2], o4[:, 2:3], o4[:, 3:4], torch.exp(o4[:, 4:5])),
        #                   dim=1)
        #print('network output size: ', output.size())
        print('occupancy field mean: ', torch.mean(output[:, :1]).item(), 'max: ', torch.max(output[:, :1]).item(), 'min: ',
              torch.min(output[:, :1]).item())

        output = torch.transpose(output, 0, 1).reshape(1, 5, -1)

        return output
