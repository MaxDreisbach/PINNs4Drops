import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import grad

from .BasePIFuNet import BasePIFuNet
from .SurfaceClassifier import SurfaceClassifier
from .DepthNormalizer import DepthNormalizer
from .HGFilters import *
from ..net_util import init_net
from ..geometry import project_velocity_vector_field


def normalize(data, min, max):
    return (data - min) / (max - min)


def de_norm(data, min, max):
    return data * (max - min) + min


# a very simple torch method to compute derivatives.
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads


def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]


def write_log(value, name):
    log_file = 'log_' + str(name) + '.txt'
    with open(log_file, 'a') as outfile:
        outfile.write('{0},\n'.format(value.item()))


class HGPIFuNet(BasePIFuNet):
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
        super(HGPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)

        self.name = 'hgpifu-PINN'

        self.opt = opt
        # for PINN (u,v,w,p) data loss term
        self.n_vel_pres_data = self.opt.n_vel_pres_data

        self.num_views = self.opt.num_views

        self.image_filter = HGFilter(opt)

        self.surface_classifier = SurfaceClassifier(
            filter_channels=self.opt.mlp_dim,
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

        init_net(self)

    def diff_xyz_de_norm(self, data):
        return data / (self.xmax - self.xmin)

    def diff_t_de_norm(self, data):
        return data / (self.tmax - self.tmin)

    def get_non_dimensional_pred(self):
        # retrieve de-normalized data for u,v,w,p
        alpha = self.preds[:, 0, :]
        u = de_norm(self.pred[:, 1, :], -5.0, 5.0)
        v = de_norm(self.pred[:, 2, :], -2.0, 4.5)
        w = de_norm(self.pred[:, 3, :], -5.0, 5.0)
        p = de_norm(self.pred[:, 4, :], -1.25, 4.0)

        return torch.stack((alpha, u, v, w, p), dim=1)

    def get_dimensional_pred(self):
        # retrieve de-normalized data for u,v,w,p
        alpha = self.preds[:, 0, :]
        u = de_norm(self.pred[:, 1, :], -5.0, 5.0)
        v = de_norm(self.pred[:, 2, :], -2.0, 4.5)
        w = de_norm(self.pred[:, 3, :], -5.0, 5.0)
        p = de_norm(self.pred[:, 4, :], -1.25, 4.0)

        # retrieve dimensional data for u,v,w,p
        u_dim = u * self.U_ref
        v_dim = v * self.U_ref
        w_dim = w * self.U_ref
        p_dim = p * self.rho_ref * self.U_ref**2

        return torch.stack((alpha, u_dim, v_dim, w_dim, p_dim), dim=1)

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

    def query(self, points, calibs, transforms=None, labels=None, labels_u=None, labels_v=None, labels_w=None,
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
        # TODO: read fluid properties and impact parameters from a .json dict
        # impact parameters
        U_0 = 0.62  # impact velocity
        D_0 = 2.1 / 10 ** 3  # Droplet diameter
        rp = 256 / 93.809 / 10 ** 3  # synthetic image reproduction scale
        rho_1 = 998.2  # density of inside medium (water)
        self.U_ref = U_0  # impact velocity
        self.L_ref = rp  # Droplet diameter or image reproduction scale
        self.rho_ref = rho_1  # selected density of water
        # min-max normalization boundaries
        self.xmax = 550.0
        self.xmin = -550.0
        self.tmax = 70.0
        self.tmin = 0

        if labels is not None:
            self.labels = labels

        if labels_u is not None and labels_v is not None and labels_w is not None:
            labels_u_proj, labels_w_proj = project_velocity_vector_field(labels_u, labels_w, calibs)

            # normalizing the label data
            labels_u_proj = normalize(labels_u_proj, -5.0, 5.0)
            labels_v = normalize(labels_v, -2.0, 4.5)
            labels_w_proj = normalize(labels_w_proj, -5.0, 5.0)
            # print('u field mean: ', labels_u.mean().item(), 'max: ', labels_u.max().item(), 'min: ', labels_u.min().item())
            # print('p field mean: ', labels_p.mean().item(), 'max: ', labels_p.max().item(), 'min: ', labels_p.min().item())

            self.labels_u = labels_u_proj
            self.labels_v = labels_v
            self.labels_w = labels_w_proj

        if labels_p is not None:
            labels_p = normalize(labels_p, -1.25, 4.0)
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
        x = xyz[:, :1, :]
        y = xyz[:, 1:2, :]
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        '''non-dimensionalize coordinates'''
        x_non_dim = x / self.L_ref
        y_non_dim = y / self.L_ref
        z_non_dim = z / self.L_ref

        '''Normalize data to [0,1] by min-max-normalization'''
        self.x_feat = normalize(x_non_dim, self.xmin, self.xmax)
        self.y_feat = normalize(y_non_dim, self.xmin, self.xmax)
        self.z_feat = normalize(z_non_dim, self.xmin, self.xmax)
        self.x_feat.requires_grad = True
        self.y_feat.requires_grad = True
        self.z_feat.requires_grad = True


        if self.opt.skip_hourglass:
            tmpx_local_feature = self.index(self.tmpx, xy)

        self.intermediate_preds_list = []

        for im_feat in self.im_feat_list:
            image_feature = self.index(im_feat, xy)

            self.x = self.x_feat
            self.y = self.y_feat
            self.z = self.z_feat

            # print('image feature size: ', image_feature.size())
            # print('x size: ', self.x.size())
            # print('t size: ', self.t.size())

            self.pred = self.surface_classifier(image_feature, self.x, self.y, self.z, self.t)

            ''' Masking output occupancy field -> always zero outside of shadowgraph contour, applied along depth dimension '''
            pred_occupancy = in_img[:, None].float() * self.pred[:, :1, :]
            self.intermediate_preds_list.append(pred_occupancy)

        self.preds = self.intermediate_preds_list[-1]
        self.pred_dimensional = self.get_dimensional_pred()
        # print('occupancy field mean: ', self.preds.mean().item(), 'max: ', self.preds.max().item(), 'min: ', self.preds.min().item())

    def get_im_feat(self):
        '''
        Get the image filter
        :return: [B, C_feat, H, W] image feature after filtering
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        Calculates MSE-loss of occupancy field (alpha) data
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term(preds, self.labels)
        error /= len(self.intermediate_preds_list)

        return error

    def get_velocity_loss(self):
        '''
        Calculates MSE-loss of velocity data sampling points
        '''
        if self.n_vel_pres_data >= self.pred.size(dim=2):
            self.n_vel_pres_data = self.pred.size(dim=2)

        pred_u = self.pred[:, 1, :self.n_vel_pres_data]
        pred_v = self.pred[:, 2, :self.n_vel_pres_data]
        pred_w = self.pred[:, 3, :self.n_vel_pres_data]
        error_u = self.error_term(pred_u, self.labels_u)
        error_v = self.error_term(pred_v, self.labels_v)
        error_w = self.error_term(pred_w, self.labels_w)

        return error_u + error_v + error_w

    def get_pressure_loss(self):
        '''
        Calculates MSE-loss of pressure data sampling points
        '''
        if self.n_vel_pres_data >= self.pred.size(dim=2):
            self.n_vel_pres_data = self.pred.size(dim=2)

        pred_p = self.pred[:, 4, :self.n_vel_pres_data]
        error_p = self.error_term(pred_p, self.labels_p)

        return error_p

    def detect_faulty_derivative(self, grad, name):
        error_log = 'error_log.txt'
        faulty_grad = 0
        alpha = self.pred[0, 0, :]

        if torch.any(grad.isnan()):
            print('at least one value of %s is nan' % name)
            print(grad[grad.isnan()])
            print('Occupancy field is: ', alpha[grad.isnan()])
            faulty_grad = 1
            # write error log
            with open(error_log, 'w') as outfile:
                outfile.write('at least one value of %s is nan \n' % name)
                outfile.write(str(grad[grad.isnan()]))
                outfile.write('\n Occupancy field is: %s' % alpha[grad.isnan()])

        if torch.any(grad.isinf()):
            print('at least one value of %s is inf' % name)
            print(grad[grad.isinf()])
            print('Occupancy field is: ', alpha[grad.isinf()])
            faulty_grad = 2
            with open(error_log, 'w') as outfile:
                outfile.write('at least one value of %s is inf \n' % name)
                outfile.write(str(grad[grad.isinf()]))
                outfile.write('\n Occupancy field is: %s' % alpha[grad.isnan()])

        # if not grad.all():
        #    print('at least one value of %s is zero' % name)
        #    print(grad[grad == 0])
        #    print('Occupancy field is: ', alpha[grad == 0])
        #    faulty_grad = 3

        return faulty_grad

    def get_pde_loss(self, points):
        '''
        Calculates MSE-loss of the phase advection equation and the Navier-Stokes equations (continuity and momentum equations in x,y,z)
        '''
        # get prediction
        alpha = self.preds[0, 0, :]  # preds instead of pred to get masking of alpha
        u = self.pred[0, 1, :]
        v = self.pred[0, 2, :]
        w = self.pred[0, 3, :]
        p = self.pred[0, 4, :]

        # get dimensional quantities
        u = de_norm(u, -5.0, 5.0)
        v = de_norm(v, -2.0, 4.5)
        w = de_norm(w, -5.0, 5.0)
        p = de_norm(p, -1.25, 4.0)
        #print('u field mean: ', u.mean().item(), 'max: ', u.max().item(), 'min: ', u.min().item())
        #print('p field mean: ', p.mean().item(), 'max: ', p.max().item(), 'min: ', p.min().item())

        # fluid parameters
        sigma = 0.071  # surface tension
        rho_1 = 998.2  # density of inside medium (water)
        rho_2 = 1.204  # density of outside medium (air)
        mu_1 = 1.0016 / 10 ** 3  # density of inside medium (water)
        mu_2 = 1.825 / 10 ** 5  # density of outside medium (air)
        g = 9.81  # gravity
        eps = 0.1  # offset from ground to catch solid-liquid interface
        y_ground = 1e-5 * 6 * 50000 - 2.75  # 60um from y0+eps#-10.75 + eps# in (256,256,256) image space
        # mixture density, viscosity
        rho_M = alpha * rho_1 + (1 - alpha) * rho_2
        mu_M = alpha * mu_1 + (1 - alpha) * mu_2

        # derivatives
        alpha_t = self.diff_t_de_norm(nth_derivative(alpha, wrt=self.t, n=1))
        alpha_x = self.diff_xyz_de_norm(nth_derivative(alpha, wrt=self.x, n=1))
        alpha_y = self.diff_xyz_de_norm(nth_derivative(alpha, wrt=self.y, n=1))
        alpha_z = self.diff_xyz_de_norm(nth_derivative(alpha, wrt=self.z, n=1))
        alpha_xx = self.diff_xyz_de_norm(nth_derivative(alpha_x, wrt=self.x, n=1))
        alpha_yy = self.diff_xyz_de_norm(nth_derivative(alpha_y, wrt=self.y, n=1))
        alpha_zz = self.diff_xyz_de_norm(nth_derivative(alpha_z, wrt=self.z, n=1))
        alpha_xy = self.diff_xyz_de_norm(nth_derivative(alpha_x, wrt=self.z, n=1))
        alpha_xz = self.diff_xyz_de_norm(nth_derivative(alpha_x, wrt=self.z, n=1))
        alpha_yz = self.diff_xyz_de_norm(nth_derivative(alpha_y, wrt=self.z, n=1))

        u_t = self.diff_t_de_norm(nth_derivative(u, wrt=self.t, n=1))
        u_x = self.diff_xyz_de_norm(nth_derivative(u, wrt=self.x, n=1))
        u_y = self.diff_xyz_de_norm(nth_derivative(u, wrt=self.y, n=1))
        u_z = self.diff_xyz_de_norm(nth_derivative(u, wrt=self.z, n=1))
        u_xx = self.diff_xyz_de_norm(nth_derivative(u_x, wrt=self.x, n=1))
        u_yy = self.diff_xyz_de_norm(nth_derivative(u_y, wrt=self.y, n=1))
        u_zz = self.diff_xyz_de_norm(nth_derivative(u_z, wrt=self.z, n=1))

        v_t = self.diff_t_de_norm(nth_derivative(v, wrt=self.t, n=1))
        v_x = self.diff_xyz_de_norm(nth_derivative(v, wrt=self.x, n=1))
        v_y = self.diff_xyz_de_norm(nth_derivative(v, wrt=self.y, n=1))
        v_z = self.diff_xyz_de_norm(nth_derivative(v, wrt=self.z, n=1))
        v_xx = self.diff_xyz_de_norm(nth_derivative(v_x, wrt=self.x, n=1))
        v_yy = self.diff_xyz_de_norm(nth_derivative(v_y, wrt=self.y, n=1))
        v_zz = self.diff_xyz_de_norm(nth_derivative(v_z, wrt=self.z, n=1))

        w_t = self.diff_t_de_norm(nth_derivative(w, wrt=self.t, n=1))
        w_x = self.diff_xyz_de_norm(nth_derivative(w, wrt=self.x, n=1))
        w_y = self.diff_xyz_de_norm(nth_derivative(w, wrt=self.y, n=1))
        w_z = self.diff_xyz_de_norm(nth_derivative(w, wrt=self.z, n=1))
        w_xx = self.diff_xyz_de_norm(nth_derivative(w_x, wrt=self.x, n=1))
        w_yy = self.diff_xyz_de_norm(nth_derivative(w_y, wrt=self.y, n=1))
        w_zz = self.diff_xyz_de_norm(nth_derivative(w_z, wrt=self.z, n=1))

        p_x = self.diff_xyz_de_norm(nth_derivative(p, wrt=self.x, n=1))
        p_y = self.diff_xyz_de_norm(nth_derivative(p, wrt=self.y, n=1))
        p_z = self.diff_xyz_de_norm(nth_derivative(p, wrt=self.z, n=1))

        # derivatives of viscosity mixture required for viscous term in NSE
        mu_x = (mu_1 - mu_2) * alpha_x
        mu_y = (mu_1 - mu_2) * alpha_y
        mu_z = (mu_1 - mu_2) * alpha_z

        f_alpha_t = self.detect_faulty_derivative(alpha_t, 'alpha_xx')
        f_alpha_xx = self.detect_faulty_derivative(alpha_xx, 'alpha_xx')
        f_alpha_yy = self.detect_faulty_derivative(alpha_yy, 'alpha_yy')
        f_alpha_zz = self.detect_faulty_derivative(alpha_zz, 'alpha_zz')
        f_alpha_xy = self.detect_faulty_derivative(alpha_xy, 'alpha_xy')
        f_alpha_xz = self.detect_faulty_derivative(alpha_xz, 'alpha_xz')
        f_alpha_yz = self.detect_faulty_derivative(alpha_yz, 'alpha_yz')

        # get dimensionless numbers
        one_Re = mu_M / (self.rho_ref * self.U_ref * self.L_ref)  # 1/Re
        one_Re_x = mu_x / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dx)
        one_Re_y = mu_y / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dy)
        one_Re_z = mu_z / (self.rho_ref * self.U_ref * self.L_ref)  # 1/(dRe/dz)
        one_We = sigma / (self.rho_ref * self.U_ref ** 2 * self.L_ref)  # 1/We
        one_Fr2 = g * self.L_ref / self.U_ref ** 2  # 1/(Fr^2)

        # surface tension term (requires error handling due to possible zero division)
        abs_interface_grad = torch.sqrt(
            torch.square(alpha_x) + torch.square(alpha_y) + torch.square(alpha_z) + np.finfo(float).eps)
        curvature = - ((alpha_xx + alpha_yy + alpha_zz) / abs_interface_grad
                       - (alpha_x ** 2 * alpha_xx + alpha_y ** 2 * alpha_yy + alpha_z ** 2 * alpha_zz +
                          2 * alpha_x * alpha_y * alpha_xy +
                          2 * alpha_x * alpha_z * alpha_xz +
                          2 * alpha_y * alpha_z * alpha_yz) / torch.pow(abs_interface_grad, 3))

        f_sigma_x = one_We * curvature * alpha_x
        f_sigma_y = one_We * curvature * alpha_y
        f_sigma_z = one_We * curvature * alpha_z

        ''' No residual calculation for sampling points within solid substrate -> Masking'''
        residual_mask = torch.zeros_like(curvature)
        for i in range(self.opt.num_sample_inout):
            # print(points[:, 1, i])
            if points[:, 1, i] >= y_ground:
                residual_mask[:, :, i] = 1
            else:
                residual_mask[:, :, i] = 0

        # zero_residual_points = (residual_mask == 0).sum()
        # print('no. of residual points on liquid-solid interface: %s -> nse residual set to zero' % zero_residual_points.item())

        '''two phase flow single-field navier stokes equations in the phase intensive-form are considered here (see 
        Marschall 2011, pp 121ff) - The derivatives of the phase field in the unsteady and convective term result 
        to zero, as they yield in a term that is equal to the interface advection equation (similarly terms drop out 
        due to continuity) '''
        # calculate residual of momentum equations
        res_momentum_x = rho_M / self.rho_ref * (u_t + u * u_x + v * u_y + w * u_z) + p_x - one_Re * (
                u_xx + u_yy + u_zz) - 2 * one_Re_x * u_x - one_Re_y * (u_y + v_x) - one_Re_z * (
                                      u_z + w_x) - f_sigma_x

        # check sign of gravity term
        res_momentum_y = rho_M / self.rho_ref * (v_t + u * v_x + v * v_y + w * v_z) + p_y - one_Re * (
                v_xx + v_yy + v_zz) - 2 * one_Re_y * u_y - one_Re_x * (u_y + v_x) - one_Re_z * (
                                  v_z + w_y) - f_sigma_y + rho_M / self.rho_ref * one_Fr2

        res_momentum_z = rho_M / self.rho_ref * (w_t + u * w_x + v * w_y + w * w_z) + p_z - one_Re * (
                w_xx + w_yy + w_zz) - 2 * one_Re_z * u_z - one_Re_y * (v_z + w_y) - one_Re_x * (
                                      u_z + w_x) - f_sigma_z

        loss_x = F.mse_loss(res_momentum_x * residual_mask, torch.zeros_like(res_momentum_x))
        loss_y = F.mse_loss(res_momentum_y * residual_mask, torch.zeros_like(res_momentum_y))
        loss_z = F.mse_loss(res_momentum_z * residual_mask, torch.zeros_like(res_momentum_z))
        nse_loss = loss_x + loss_y + loss_z

        ''' Phase field advection and continuity equation losses'''
        phase_adv_residual = alpha_t + u * alpha_x + v * alpha_y + w * alpha_z
        phase_adv_loss = F.mse_loss(phase_adv_residual * residual_mask, torch.zeros_like(phase_adv_residual))

        conti_residual = u_x + v_y + w_z
        conti_loss = F.mse_loss(conti_residual * residual_mask, torch.zeros_like(conti_residual))

        return conti_loss, phase_adv_loss, nse_loss

    def forward(self, images, points, calibs, transforms=None, labels=None, labels_u=None,
                labels_v=None, labels_w=None, labels_p=None, time_step=None, get_PINN_loss=True):
        # Get image feature
        self.filter(images)

        # Phase 2: point query
        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels, labels_u=labels_u,
                   labels_v=labels_v, labels_w=labels_w, labels_p=labels_p, time_step=time_step)

        # get the prediction
        res = self.get_preds()
        res_PINN = self.get_non_dimensional_pred()

        # get the data loss for alpha, (u,w,w) velocity components and pressure
        error_data = self.get_error()
        error_data_vel = self.get_velocity_loss()
        error_data_pres = self.get_pressure_loss()

        # get pde errors - do not call during inference (missing gradients for model in test mode)
        if get_PINN_loss:
            error_conti, error_phase_conv, error_nse = self.get_pde_loss(points=points)
        else:
            error_conti = error_data * 0
            error_phase_conv = error_data * 0
            error_nse = error_data * 0

        w_vel = self.opt.w_vel
        w_pres = self.opt.w_pres
        w_conti = self.opt.w_conti
        w_phase = self.opt.w_phase
        w_nse = self.opt.w_nse
        error_total = error_data + w_vel * error_data_vel + w_pres * error_data_pres + w_conti * error_conti + w_phase * error_phase_conv + w_nse * error_nse

        return res, res_PINN, error_total, error_data, w_vel * error_data_vel, w_pres * error_data_pres, w_conti * error_conti, w_phase * error_phase_conv, w_nse * error_nse
