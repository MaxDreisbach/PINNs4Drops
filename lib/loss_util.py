import torch
import torch.nn as nn
from lib.options import BaseOptions

# get options
opt = BaseOptions().parse()


def get_loss_weights_SoftAdapt(losses, losses_prev, beta=0.1, var_lw=False):
    ''' Implementation of SoftAdapt in two variants
    1) SoftAdapt based on loss ratios (https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/advanced_schemes.html)
    2) Loss weighted Softadapt according to original paper
    The second variant assigns smaller weights to lower loss terms, which makes sense if all losses are expected to reach a similar magnitude
    Since the PDE losses should be lower than the data losses the first variant is used as default
    Note: a small value (eps) is added to divisions for numerical stability'''
    eps = 10 ** (-8)

    losses = losses.detach().clone()
    losses_prev = losses_prev.detach().clone()

    if var_lw:
        diff_losses = losses - losses_prev + eps
        weights = nn.Softmax(dim=0)(beta * diff_losses - torch.max(diff_losses))
        losses_mean = torch.mean(torch.stack((losses, losses_prev), dim=1), dim=1)
        weights = losses_mean * weights / torch.sum((losses_mean * weights + eps))
    else:
        ratio_losses = losses / (losses_prev + eps)
        weights = nn.Softmax(dim=0)(ratio_losses - torch.max(ratio_losses))

    # NEW: make sure that alpha field loss does not drop - either assign max(weights) or 1/10 if w_alpha < 1/10
    w_alpha_lim = 0.1
    if weights[0] <= w_alpha_lim:
        # weights[0] = torch.max(weights)
        weights[0] = w_alpha_lim

    return weights * 10


def get_EWMA(loss, prev_EWMA, iteration, epoch, opt, beta=0.9):
    ''' calculates exponentially weighted moving average of input
    inputs: tensor of losses, tensor of previous EWMA
    output: current EWMA'''

    loss = loss.detach().clone()

    if iteration == 0 and epoch == 0:
        losses_EWMA = torch.ones_like(loss)
    elif iteration == 0 and epoch == opt.resume_epoch:
        losses_EWMA = torch.ones_like(loss)
    else:
        losses_EWMA = beta * prev_EWMA + (1 - beta) * loss

    return losses_EWMA


def assign_global_weights(l_a, l_u, l_v, l_w, l_p, l_c, l_ph, l_x, l_y, l_z, opt):
    ''' apply global loss weights to each loss term '''
    l_u = opt.weight_u * l_u
    l_v = opt.weight_v * l_v
    l_w = opt.weight_w * l_w
    l_p = opt.weight_p * l_p
    l_c = opt.weight_conti * l_c
    l_ph = opt.weight_phase * l_ph
    l_x = opt.weight_mom_x * l_x
    l_y = opt.weight_mom_y * l_y
    l_z = opt.weight_mom_z * l_z

    return torch.stack((l_a, l_u, l_v, l_w, l_p, l_c, l_ph, l_x, l_y, l_z), dim=0)



def assign_global_weights_CH2(l_a, l_u, l_v, l_w, l_p, l_c, l_ph, l_x, l_y, l_z, l_e, l_phi, opt):
    ''' apply global loss weights to each loss term (Cahn-Hilliard version)'''
    l_u = opt.weight_u * l_u
    l_v = opt.weight_v * l_v
    l_w = opt.weight_w * l_w
    l_p = opt.weight_p * l_p
    l_c = opt.weight_conti * l_c
    l_ph = opt.weight_phase * l_ph
    l_x = opt.weight_mom_x * l_x
    l_y = opt.weight_mom_y * l_y
    l_z = opt.weight_mom_z * l_z
    l_e = opt.weight_l_eps * l_e
    l_phi = opt.weight_phi * l_phi

    return torch.stack((l_a, l_u, l_v, l_w, l_p, l_c, l_ph, l_x, l_y, l_z, l_e, l_phi), dim=0)


def get_data_loss_onramp(losses, iteration, epoch, duration=1000):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    if iteration < duration and epoch == 0:
        losses_uvwp = losses_uvwp * (iteration / duration)

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_data_loss_cuton(losses, a_thresh=0.05):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    loss_a = loss_alpha.item()

    if loss_a > a_thresh:
        losses_uvwp = losses_uvwp * 0.01

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_pde_loss_onramp(losses, iteration, epoch, duration=5000):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    if iteration < duration and epoch == 0:
        losses_pde = losses_pde * (iteration / duration)

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_pde_loss_cuton(losses, a_thresh=0.05):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    loss_a = loss_alpha.item()

    if loss_a > a_thresh:
        losses_pde = losses_pde * 0

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_loss_weights_Kiani(losses):
    l_a = losses[:1]
    l_u = losses[1:2]
    l_v = losses[2:3]
    l_w = losses[3:4]
    l_p = losses[4:5]
    l_c = losses[5:6]
    l_ph = losses[6:7]
    l_x = losses[7:8]
    l_y = losses[8:9]
    l_z = losses[9:10]
    total_loss_uvwp = l_u + l_v + l_w + l_p
    total_loss_data = total_loss_uvwp + l_a
    total_loss_pde = l_c + l_ph + l_x + l_y + l_z
    eps = 10 ** (-8)

    # First update weights of data terms
    w_u = torch.clip(total_loss_uvwp / (l_u + eps), 0.01, 50.0)
    w_v = torch.clip(total_loss_uvwp / (l_v + eps), 0.01, 50.0)
    w_w = torch.clip(total_loss_uvwp / (l_w + eps), 0.01, 50.0)
    w_p = torch.clip(total_loss_uvwp / (l_p + eps), 0.01, 50.0)
    w_a = torch.clip(total_loss_data / (l_a + eps), 0.01, 50.0)

    # Then update PDE-loss weights
    w_c = torch.clip(total_loss_pde / (l_c + eps), 0.01, 50.0)
    w_ph = torch.clip(total_loss_pde / (l_ph + eps), 0.01, 50.0)
    w_x = torch.clip(total_loss_pde / (l_x + eps), 0.01, 50.0)
    w_y = torch.clip(total_loss_pde / (l_y + eps), 0.01, 50.0)
    w_z = torch.clip(total_loss_pde / (l_z + eps), 0.01, 50.0)

    return torch.squeeze(torch.stack((w_a, w_u, w_v, w_w, w_p, w_c, w_ph, w_x, w_y, w_z), dim=0))


def get_loss_weights_mag_CH2(losses):
    l_a = losses[:1]
    l_u = losses[1:2]
    l_v = losses[2:3]
    l_w = losses[3:4]
    l_p = losses[4:5]
    l_c = losses[5:6]
    l_ph = losses[6:7]
    l_x = losses[7:8]
    l_y = losses[8:9]
    l_z = losses[9:10]
    l_e = losses[10:11]
    l_phi = losses[11:12]
    total_loss_uvwp = l_u + l_v + l_w + l_p
    total_loss_data = total_loss_uvwp + l_a
    total_loss_pde = l_c + l_ph + l_x + l_y + l_z + l_e + l_phi
    eps = 10 ** (-8)

    # First update weights of data terms
    w_u = torch.clip(total_loss_uvwp / (l_u + eps), 0.01, 50.0)
    w_v = torch.clip(total_loss_uvwp / (l_v + eps), 0.01, 50.0)
    w_w = torch.clip(total_loss_uvwp / (l_w + eps), 0.01, 50.0)
    w_p = torch.clip(total_loss_uvwp / (l_p + eps), 0.01, 50.0)
    w_a = torch.clip(total_loss_data / (l_a + eps), 0.01, 50.0)

    # Then update PDE-loss weights
    w_c = torch.clip(total_loss_pde / (l_c + eps), 0.01, 50.0)
    w_ph = torch.clip(total_loss_pde / (l_ph + eps), 0.01, 50.0)
    w_x = torch.clip(total_loss_pde / (l_x + eps), 0.01, 50.0)
    w_y = torch.clip(total_loss_pde / (l_y + eps), 0.01, 50.0)
    w_z = torch.clip(total_loss_pde / (l_z + eps), 0.01, 50.0)
    w_l = torch.clip(total_loss_pde / (l_e + eps), 0.01, 50.0)
    w_phi = torch.clip(total_loss_pde / (l_phi + eps), 0.01, 50.0)

    return torch.squeeze(torch.stack((w_a, w_u, w_v, w_w, w_p, w_c, w_ph, w_x, w_y, w_z, w_l, w_phi), dim=0))