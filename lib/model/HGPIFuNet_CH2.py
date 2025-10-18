import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

from torch.autograd import grad

from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier_CH2 import SurfaceClassifier_CH2
from .SurfaceClassifier_LAAF_CH2 import SurfaceClassifier_LAAF_CH2
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from ..geometry import project_velocity_vector_field
from ..plotting import plot_data_sample
from ..plotting import plot_im_feat


def normalize(data, min, max):
    '''min-max normalization'''
    return (data - min) / (max - min)


def de_norm(data, min, max):
    '''min-max normalization'''
    return data * (max - min) + min


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]


def write_log(value, name):
    log_file = 'log_' + str(name) + '.txt'
    with open(log_file, 'a') as outfile:
        outfile.write('{0},\n'.format(value.item()))


class HGPIFuNet_CH2(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''

    def __init__(self,
                 opt,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(HGPIFuNet_CH2, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'PF-IcPINNs-MIM'
        self.opt = opt
        self.root = self.opt.dataroot
        print(self.root)

        # for PINN (u,v,w,p) data loss term
        self.n_data = self.opt.n_data
        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)
        if opt.freeze_hourglas:
            for i, layer in enumerate(self.image_filter.children()):
                print('freezing hourglas layer ', i)
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        
        self.surface_classifier = SurfaceClassifier_LAAF_CH2(
            filter_channels=self.opt.mlp_dim_CH2,
            num_views=self.opt.num_views,
            no_residual=self.opt.no_residual,
            last_op=nn.Sigmoid())

        self.normalizer = DepthNormalizer(opt)

        # This is a list of [B x Feat_i x H x W] features
        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.pred = []
        # initialize normalized (& dimensionless) coordinates
        self.x_feat, self.y_feat, self.z_feat = [], [], []
        # initialize dimensionless coordinates
        self.x, self.y, self.z = [], [], []
        self.intermediate_preds_list = []
        self.intermediate_u = []

        # Read fluid properties
        with open(os.path.join(self.root, "flow_case.json"), "r") as f:
            flow_case = json.load(f)

        self.U_ref = flow_case["U_0"]  # impact velocity
        self.L_ref = flow_case["rp"]  # image reproduction scale -> domain size in image space
        self.rho_ref = flow_case["rho_1"]  # selected density of water
        self.sigma = flow_case["sigma"]  # surface tension
        self.rho_1 = flow_case["rho_1"]  # density of inside medium
        self.rho_2 = flow_case["rho_2"]  # density of outside medium
        self.mu_1 = flow_case["mu_1"]  # viscosity of inside medium
        self.mu_2 = flow_case["mu_2"]  # viscosity of outside medium
        self.g = flow_case["g"]  # gravity
        self.y_ground = flow_case["y_ground"]  # 60um from y0+eps#-10.75 + eps# in (256,256,256) image space

        # self.epsilon = flow_case["epsilon"] # capillary width
        # self.M_0 = flow_case["M_0"] # mobility
        # self.epsilon = 0.01  # following Qiu (2022) - https://doi.org/10.1063/5.0091063
        self.epsilon = nn.Parameter(self.opt.eps * torch.ones(1))  # initialize learnable interface thickness
        self.M_0 = (self.epsilon.item()) ** 2
        self.L_THRESH = 0.05
        print('epsilon: ', self.epsilon.item())
        print('M_0: ', self.M_0)

        # min-max normalization boundaries
        self.xmin = flow_case["x_norm_min"]
        self.xmax = flow_case["x_norm_max"]
        self.tmin = flow_case["t_norm_min"]
        self.tmax = flow_case["t_norm_max"]
        self.umin = flow_case["u_norm_min"]
        self.umax = flow_case["u_norm_max"]
        self.vmin = flow_case["v_norm_min"]
        self.vmax = flow_case["v_norm_max"]
        self.wmin = flow_case["w_norm_min"]
        self.wmax = flow_case["w_norm_max"]
        self.pmin = flow_case["p_norm_min"]
        self.pmax = flow_case["p_norm_max"]

        self.RBA_lr = 0.4  # learning rate for RBA Lagrange multipliers
        self.RBA_b = 0.8

        init_net(self)


    def nth_derivative(self,f, wrt, n):
        #f = f[:, :, self.n_data*2:]
        #wrt = wrt[:, :, self.n_data*2:]
        for i in range(n):
            grads = grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
            f = grads
            if grads is None:
                print(f"[Error] Gradient is None at order {i + 1}.")
                return torch.tensor(0., device=f.device)
        #return only the grads for the pde sampling points             
        return grads[:, :, self.n_data*2:]


    def diff_xyz_de_norm(self, data):
        return data / (self.xmax - self.xmin)


    def diff_t_de_norm(self, data):
        return data / (self.tmax - self.tmin)


    def get_non_dimensional_pred(self):
        # retrieve de-normalized data for u,v,w,p
        alpha = self.preds[:, 0, :]
        u = de_norm(self.pred[:, 1, :], self.umin, self.umax)
        v = de_norm(self.pred[:, 2, :], self.vmin, self.vmax)
        w = de_norm(self.pred[:, 3, :], self.wmin, self.wmax)
        p = de_norm(self.pred[:, 4, :], self.pmin, self.pmax)

        return torch.stack((alpha, u, v, w, p), dim=1)


    def get_dimensional_pred(self):
        # retrieve de-normalized data for u,v,w,p
        alpha = self.preds[:, 0, :]
        u = de_norm(self.pred[:, 1, :], self.umin, self.umax)
        v = de_norm(self.pred[:, 2, :], self.vmin, self.vmax)
        w = de_norm(self.pred[:, 3, :], self.wmin, self.wmax)
        p = de_norm(self.pred[:, 4, :], self.pmin, self.pmax)

        # retrieve dimensional data for u,v,w,p
        u_dim = u * self.U_ref
        v_dim = v * self.U_ref
        w_dim = w * self.U_ref
        p_dim = p * self.rho_ref * self.U_ref ** 2

        return torch.stack((alpha, u_dim, v_dim, w_dim, p_dim), dim=1)

    def get_phi_pred(self):
        # retrieve data for phi
        return self.pred[:, 5:6, :]


    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        :param im_feat_list: [1, 256, 128, 128]*4 - list with feature maps (4 feature maps from 4 hourglass modules)
        '''
        self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
        # If it is not in training, only produce the last im_feat
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]


    def query(self, points, calibs, transforms=None, labels=None, uvwp_points=None, residual_points=None, labels_u=None, labels_v=None, labels_w=None,
              labels_p=None, time_step=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt alpha field labeling
        :param labels_u: Optional [B, Res, N] gt u-velocity component labeling
        :param labels_v: Optional [B, Res, N] gt v-velocity component labeling
        :param labels_w: Optional [B, Res, N] gt w-velocity component labeling
        :param labels_p: Optional [B, Res, N] gt pressure field labeling
        :return: [B, Res, N] predictions for each point
        '''
        # collect sampling points for joint prediction -> later separated (after prediction) for loss computation
        if uvwp_points is not None:
            points = torch.cat([points, uvwp_points, residual_points], 2)

        if labels is not None:
            self.labels = labels

        if labels_u is not None and labels_v is not None and labels_w is not None:
            labels_u_proj, labels_w_proj = project_velocity_vector_field(labels_u, labels_w, calibs)

            # normalizing the label data
            labels_u_proj = normalize(labels_u_proj, self.umin, self.umax)
            labels_v = normalize(labels_v, self.vmin, self.vmax)
            labels_w_proj = normalize(labels_w_proj, self.wmin, self.wmax)
            # print('u field mean: ', labels_u.mean().item(), 'max: ', labels_u.max().item(), 'min: ', labels_u.min().item())
            # print('p field mean: ', labels_p.mean().item(), 'max: ', labels_p.max().item(), 'min: ', labels_p.min().item())

            self.labels_u = labels_u_proj
            self.labels_v = labels_v
            self.labels_w = labels_w_proj

        if labels_p is not None:
            labels_p = normalize(labels_p, self.pmin, self.pmax)
            self.labels_p = labels_p

        ''' time step label input added'''
        if time_step is not None:
            # expand time step label to size of sample points in order to match other parametric input
            time_step = time_step.repeat(1, points.size(-1))
            time_step = torch.unsqueeze(time_step, 0)
            time_norm = normalize(time_step, self.tmin, self.tmax)
            self.t = time_norm
            self.t.requires_grad = True

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        '''Flip y-axis - required as it was flipped in TrainDataset'''
        self.x_feat = xyz[:, :1, :]
        self.y_feat = -xyz[:, 1:2, :]
        self.z_feat = xyz[:, 2:3, :]
        self.x_feat.requires_grad = True
        self.y_feat.requires_grad = True
        self.z_feat.requires_grad = True

        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            # plot_im_feat(im_feat)
            image_feature = self.index(im_feat, xy)
            self.x = self.x_feat
            self.y = self.y_feat
            self.z = self.z_feat

            self.pred = self.surface_classifier(image_feature, self.x, self.y, self.z, self.t)

            ''' Masking output sampled points outside of image domain '''
            pred_occupancy = in_img[:, None].float() * self.pred[:, :1, :]
            self.intermediate_preds_list.append(pred_occupancy)

        self.preds = self.intermediate_preds_list[-1]
        self.pred_dimensional = self.get_dimensional_pred()


    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]


    def get_solid_domain_mask(self, points):
        y_coords = points[:, 1:2, :]  # Get y-coordinates
        mask = (y_coords >= self.y_ground).float()
        return mask


    def get_RBA_residual(self, residuals):
        res_max = torch.max(torch.squeeze(torch.abs(residuals)))
        lambda_k = self.RBA_b + self.RBA_lr * torch.abs(residuals) / (res_max.item() + 1e-8)
        # print('Residuals maximum: ', res_max.item())
        # print('Lagrange multipliers min: ', torch.min(lambda_k).item(), ' max: ', torch.max(lambda_k).item(), ' mean: ', torch.mean(lambda_k).item())
        # print('Lagrange multipliers: ', lambda_k)
        # print('Lagrange multipliers shape: ', lambda_k.size())

        return lambda_k * residuals, lambda_k


    def get_error(self):
        '''
        Calculates MSE-loss of occupancy field (alpha) data
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            # get prediction for observation points only
            pred_a = preds[:, :, :self.n_data]
            labels_a = self.labels[:, :, :self.n_data]
            error += self.error_term(pred_a, labels_a)
        error /= len(self.intermediate_preds_list)

        return error


    def get_velocity_loss(self, points):
        '''
        Calculates MSE-loss of velocity data sampling points
        '''
        if points is not None:
            n_start = self.n_data
            n_end = self.n_data * 2
            # get prediction for observation points
            pred_u = self.pred[:, 1, n_start:n_end]
            pred_v = self.pred[:, 2, n_start:n_end]
            pred_w = self.pred[:, 3, n_start:n_end]

            # Do not calculate loss for sample points below surface in solid domain and within grooves
            mask = self.get_solid_domain_mask(points)
            error_u = self.error_term(pred_u * mask, self.labels_u * mask)
            error_v = self.error_term(pred_v * mask, self.labels_v * mask)
            error_w = self.error_term(pred_w * mask, self.labels_w * mask)
        else:
            error_u = 0
            error_v = 0
            error_w = 0

        return error_u, error_v, error_w


    def get_pressure_loss(self, points):
        '''
        Calculates MSE-loss of pressure data sampling points
        '''
        if points is not None:
            n_start = self.n_data
            n_end = self.n_data * 2

            # get prediction for observation points
            pred_p = self.pred[:, 4, n_start:n_end]
            mask = self.get_solid_domain_mask(points)
            error_p = self.error_term(pred_p * mask, self.labels_p * mask)
        else:
            error_p = 0

        return error_p


    def get_pde_loss(self, points, l_eps=False):
        '''
        Calculates MSE-loss of the phase advection equation and the Navier-Stokes equations (continuity and momentum equations in x,y,z)
        '''
        #print('pred shape: ', self.pred.shape)
        
        n_start = self.n_data * 2
        
        # get prediction
        C = self.preds[:1, :1, n_start:]  # preds instead of pred to get masking of C
        u = self.pred[:1, 1:2, n_start:]
        v = self.pred[:1, 2:3, n_start:]
        w = self.pred[:1, 3:4, n_start:]
        p = self.pred[:1, 4:5, n_start:]
        phi = self.pred[:1, 5:6, n_start:]

        # get dimensional quantities
        # using de_norm() to transform alpha field prediction [0,1] to C prediction [-1,1]
        C = de_norm(C, -1.0, 1.0)
        u = de_norm(u, self.umin, self.umax)
        v = de_norm(v, self.vmin, self.vmax)
        w = de_norm(w, self.wmin, self.wmax)
        p = de_norm(p, self.pmin, self.pmax)
        # print('u field mean: ', u.mean().item(), 'max: ', u.max().item(), 'min: ', u.min().item())
        # print('p field mean: ', p.mean().item(), 'max: ', p.max().item(), 'min: ', p.min().item())

        # derivatives
        C_t = self.diff_t_de_norm(self.nth_derivative(C, wrt=self.t, n=1))
        C_x = self.nth_derivative(C, wrt=self.x, n=1)
        C_y = self.nth_derivative(C, wrt=self.y, n=1)
        C_z = self.nth_derivative(C, wrt=self.z, n=1)
        C_xx = self.nth_derivative(C_x, wrt=self.x, n=1)
        C_yy = self.nth_derivative(C_y, wrt=self.y, n=1)
        C_zz = self.nth_derivative(C_z, wrt=self.z, n=1)
        laplacian_C = C_xx + C_yy + C_zz

        phi_xx = self.nth_derivative(phi, wrt=self.x, n=2)
        phi_yy = self.nth_derivative(phi, wrt=self.y, n=2)
        phi_zz = self.nth_derivative(phi, wrt=self.z, n=2)
        laplacian_phi = phi_xx + phi_yy + phi_zz

        u_t = self.diff_t_de_norm(self.nth_derivative(u, wrt=self.t, n=1))
        u_x = self.nth_derivative(u, wrt=self.x, n=1)
        u_y = self.nth_derivative(u, wrt=self.y, n=1)
        u_z = self.nth_derivative(u, wrt=self.z, n=1)
        u_xx = self.nth_derivative(u_x, wrt=self.x, n=1)
        u_yy = self.nth_derivative(u_y, wrt=self.y, n=1)
        u_zz = self.nth_derivative(u_z, wrt=self.z, n=1)

        v_t = self.diff_t_de_norm(self.nth_derivative(v, wrt=self.t, n=1))
        v_x = self.nth_derivative(v, wrt=self.x, n=1)
        v_y = self.nth_derivative(v, wrt=self.y, n=1)
        v_z = self.nth_derivative(v, wrt=self.z, n=1)
        v_xx = self.nth_derivative(v_x, wrt=self.x, n=1)
        v_yy = self.nth_derivative(v_y, wrt=self.y, n=1)
        v_zz = self.nth_derivative(v_z, wrt=self.z, n=1)

        w_t = self.diff_t_de_norm(self.nth_derivative(w, wrt=self.t, n=1))
        w_x = self.nth_derivative(w, wrt=self.x, n=1)
        w_y = self.nth_derivative(w, wrt=self.y, n=1)
        w_z = self.nth_derivative(w, wrt=self.z, n=1)
        w_xx = self.nth_derivative(w_x, wrt=self.x, n=1)
        w_yy = self.nth_derivative(w_y, wrt=self.y, n=1)
        w_zz = self.nth_derivative(w_z, wrt=self.z, n=1)

        p_x = self.nth_derivative(p, wrt=self.x, n=1)
        p_y = self.nth_derivative(p, wrt=self.y, n=1)
        p_z = self.nth_derivative(p, wrt=self.z, n=1)

        # mixture density, viscosity
        rho_M = (1 + C) / 2 * self.rho_1 + (1 - C) / 2 * self.rho_2
        mu_M = (1 + C) / 2 * self.mu_1 + (1 - C) / 2 * self.mu_2

        # derivatives of viscosity mixture required for viscous term in NSE
        mu_x = (self.mu_1 - self.mu_2) * C_x / 2
        mu_y = (self.mu_1 - self.mu_2) * C_y / 2
        mu_z = (self.mu_1 - self.mu_2) * C_z / 2

        # get dimensionless numbers
        one_Re = mu_M / (self.rho_ref * self.U_ref * self.L_ref)  # 1/Re
        one_Re_x = mu_x / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dx)
        one_Re_y = mu_y / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dy)
        one_Re_z = mu_z / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dz)
        one_We = self.sigma / (self.rho_ref * self.U_ref ** 2 * self.L_ref)  # 1/We
        one_Fr2 = self.g * self.L_ref / self.U_ref ** 2  # 1/(Fr^2)

        ''' Cahn-Hilliard equation '''
        # ensure sensible range for learnable interface thickness epsilon (see Qiu (2022), Fink (2018), Samkhaniani (2021))
        self.epsilon.data.clamp_(min=2.2e-05, max=self.opt.eps)
        if l_eps:
            eps_loss = F.huber_loss(self.epsilon, 2.2e-05 * torch.ones_like(self.epsilon))
        else:
            eps_loss = F.huber_loss(self.epsilon, self.opt.eps * torch.ones_like(self.epsilon))
        self.M_0 = self.epsilon ** 2
        #print('M_0: ', self.M_0)

        # mixing energy density
        lambda_CH = (3 * np.sqrt(2) / 4) * self.sigma * self.epsilon
        # chemical potential
        ''' In HGPIFuNet_CH2 chemical potential is instead predicted by MLP
         in order to avoid gradient pathologies with fourth order derivative in Cahn Hilliard equation
         Idea: instead use below equation as additional loss term
        '''
        res_phi = phi - (lambda_CH / self.epsilon ** 2) * C * (C ** 2 - 1) - lambda_CH * laplacian_C
        #print('res_phi mean: ', res_phi.mean().item(), 'max: ', res_phi.max().item(), 'min: ', res_phi.min().item())

        # surface tension (sigma already contained in phi -> division)
        f_sigma_x = (one_We / self.sigma) * phi * C_x
        f_sigma_y = (one_We / self.sigma) * phi * C_y
        f_sigma_z = (one_We / self.sigma) * phi * C_z

        '''two-phase flow single-field Navier stokes equations in the phase intensive-form are considered here (see 
        Marschall 2011, pp 121ff) - The derivatives of the phase field in the unsteady and convective term result 
        to zero, as they yield in a term that is equal to the interface advection equation (similarly terms drop out 
        due to continuity) '''

        # calculate residual of momentum equations
        # calculate residual of momentum equations
        res_momentum_x = rho_M / self.rho_ref * (u_t + u * u_x + v * u_y + w * u_z) + p_x - one_Re * (
                u_xx + u_yy + u_zz) - 2 * one_Re_x * u_x - one_Re_y * (u_y + v_x) - one_Re_z * (
                                      u_z + w_x) - f_sigma_x

        res_momentum_y = rho_M / self.rho_ref * (v_t + u * v_x + v * v_y + w * v_z) + p_y - one_Re * (
                v_xx + v_yy + v_zz) - 2 * one_Re_y * v_y - one_Re_x * (u_y + v_x) - one_Re_z * (
                                  v_z + w_y) - f_sigma_y + rho_M / self.rho_ref * one_Fr2

        res_momentum_z = rho_M / self.rho_ref * (w_t + u * w_x + v * w_y + w * w_z) + p_z - one_Re * (
                w_xx + w_yy + w_zz) - 2 * one_Re_z * w_z - one_Re_y * (v_z + w_y) - one_Re_x * (
                                      u_z + w_x) - f_sigma_z


        ''' Phase field advection and continuity equation residual'''
        res_conti = u_x + v_y + w_z

        # The phase field equation
        res_phase_adv = C_t + u * C_x + v * C_y + w * C_z - self.M_0 * laplacian_phi

        ''' No residual calculation for sampling points within solid substrate -> Masking'''
        mask = self.get_solid_domain_mask(points)
        # zero_residual_points = (ground_mask == 0).sum()
        # print('no. of residual points on liquid-solid interface: %s -> nse residual set to zero' % zero_residual_points.item())
        res_momentum_x = res_momentum_x * mask
        res_momentum_y = res_momentum_y * mask
        res_momentum_z = res_momentum_z * mask
        res_phase_adv = res_phase_adv * mask
        res_phi = res_phi * mask
        res_conti = res_conti * mask

        # get RBA update with local Lagrange multipliers
        res_momentum_x, RBA_mom_x = self.get_RBA_residual(res_momentum_x)
        res_momentum_y, RBA_mom_y = self.get_RBA_residual(res_momentum_y)
        res_momentum_z, RBA_mom_z = self.get_RBA_residual(res_momentum_z)
        res_phase_adv, RBA_phase_adv = self.get_RBA_residual(res_phase_adv)
        res_phi, RBA_phi = self.get_RBA_residual(res_phi)
        res_conti, RBA_conti = self.get_RBA_residual(res_conti)

        # plot RBA
        # plot_data_sample(self.x, self.y, self.z, RBA_mom_x, 0.8, 1.2)
        # plot_data_sample(self.x, self.y, self.z, RBA_mom_y, 0.8, 1.2)
        # plot_data_sample(self.x, self.y, self.z, RBA_mom_z, 0.8, 1.2)
        # plot_data_sample(self.x, self.y, self.z, RBA_phase_adv, 0.8, 1.2)
        # plot_data_sample(self.x, self.y, self.z, RBA_conti, 0.8, 1.2)

        loss_momentum_x = F.mse_loss(res_momentum_x, torch.zeros_like(res_momentum_x))
        loss_momentum_y = F.mse_loss(res_momentum_y, torch.zeros_like(res_momentum_y))
        loss_momentum_z = F.mse_loss(res_momentum_z, torch.zeros_like(res_momentum_z))

        phase_adv_loss = F.mse_loss(res_phase_adv, torch.zeros_like(res_phase_adv))
        phi_loss = F.mse_loss(res_phi, torch.zeros_like(res_phi))

        conti_loss = F.mse_loss(res_conti, torch.zeros_like(res_conti))

        return conti_loss, phase_adv_loss, loss_momentum_x, loss_momentum_y, loss_momentum_z, eps_loss, phi_loss

    def forward(self, images, points, calibs, transforms=None, labels=None, uvwp_points=None, residual_points=None, labels_u=None,
                labels_v=None, labels_w=None, labels_p=None, time_step=None, get_PINN_loss=True):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, uvwp_points=uvwp_points, residual_points=residual_points, labels_u=labels_u,
                   labels_v=labels_v, labels_w=labels_w, labels_p=labels_p, time_step=time_step)

        # get the prediction
        res = self.get_preds()
        res_PINN = self.get_non_dimensional_pred()

        # get the data loss for alpha, (u,w,w) velocity components and pressure
        loss_data_alpha = self.get_error()
        loss_data_u, loss_data_v, loss_data_w = self.get_velocity_loss(points=uvwp_points)
        loss_data_p = self.get_pressure_loss(points=uvwp_points)

        # only learn interface thickness when alpha field is sufficiently converged
        if loss_data_alpha < self.L_THRESH:
            learn_eps = True
        else:
            learn_eps = False

        # get pde errors - do not call during inference (missing gradients for model in test mode)
        if get_PINN_loss:
            loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_eps, phi_loss = self.get_pde_loss(
                points=residual_points, l_eps=learn_eps)
        else:
            loss_conti = loss_data_alpha * 0
            loss_phase_conv = loss_data_alpha * 0
            loss_momentum_x = loss_data_alpha * 0
            loss_momentum_y = loss_data_alpha * 0
            loss_momentum_z = loss_data_alpha * 0
            return res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z

        return res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_eps, self.epsilon, phi_loss
