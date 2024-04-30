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
        #weights[0] = torch.max(weights)
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


def get_data_loss_onramp(losses, iteration, epoch, duration=1000):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    if iteration < duration and epoch == 0:
        losses_uvwp = losses_uvwp * (iteration / duration)

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_pde_loss_onramp(losses, iteration, epoch, duration=5000):
    loss_alpha = losses[:1]
    losses_uvwp = losses[1:5]
    losses_pde = losses[5:]

    if iteration < duration and epoch==0:
        losses_pde = losses_pde * (iteration / duration)

    return torch.cat((loss_alpha, losses_uvwp, losses_pde), dim=0)


def get_PDE_loss_scheduler(losses, losses_EWMA, iteration, it_start_pde_loss, epoch, ep_start_pde_loss):
    thres_alpha = 0.03

    losses_pde = losses[5:]
    l_alpha_EWMA = losses_EWMA[:1]

    if iteration == 0 and epoch == 0 or epoch == opt.resume_epoch:
        PDE_LOSS = False

    if l_alpha_EWMA <= thres_alpha and PDE_LOSS == False:
        it_start_pde_loss = iteration
        ep_start_pde_loss = epoch
        PDE_LOSS = True


    if PDE_LOSS == True:
        it_pde_anneal = iteration - it_start_pde_loss
        ''' learning rate annealing for pde loss terms'''
        if it_pde_anneal < 5000 and epoch == ep_start_pde_loss:
            losses_pde = losses_pde * (it_pde_anneal / 5000)
        else:
            '''Do nothing -> global weights for pde losses'''
    else:
        losses_pde = losses_pde * 0

    return losses_pde, it_start_pde_loss, ep_start_pde_loss


