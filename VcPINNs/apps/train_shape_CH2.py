import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.loss_util import *
from lib.data import *
from lib.model import *
from lib.model.HGPIFuNet_CH2 import HGPIFuNet_CH2
from lib.geometry import index

# get options
opt = BaseOptions().parse()


def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDataset(opt, phase='train')
    test_dataset = TrainDataset(opt, phase='test')
    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('validation data size: ', len(test_data_loader))

    print('no. of collocation points (PDE): ', opt.n_residual)
    print('no. of observation points (data): ', opt.n_data)

    # create net
    netG = HGPIFuNet_CH2(opt, projection_mode).to(device=cuda)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)

    lr = opt.learning_rate
    print('Using Network: ', netG.name)

    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        load_custom = True
        if not load_custom:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))
        else:
            checkpoint = torch.load(pt.load_netG_checkpoint_path, map_location=cuda)
            
            # Fix output layer (conv4 layer) size mismatch (5 -> 6 output neuron)
            old_weight = checkpoint['surface_classifier.conv4.weight']  # [5, 644, 1]
            old_bias = checkpoint['surface_classifier.conv4.bias']      # [5]
            
            new_weight = torch.zeros(6, 644, 1)
            new_bias = torch.zeros(6)
            
            new_weight[:5] = old_weight
            new_bias[:5] = old_bias
            
            # Random init for new neuron
            torch.nn.init.xavier_uniform_(new_weight[5:])
            
            checkpoint['surface_classifier.conv4.weight'] = new_weight
            checkpoint['surface_classifier.conv4.bias'] = new_bias
            
            # Add missing epsilon
            checkpoint['epsilon'] = netG.state_dict()['epsilon']
            
            netG.load_state_dict(checkpoint)


    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        load_custom = False
        if not load_custom:
            netG.load_state_dict(torch.load(model_path, map_location=cuda))
        else:
            print('add and initialize new learnable weights for output and epsilon...')
            checkpoint = torch.load(model_path, map_location=cuda)
            
            # Fix output layer (conv4 layer) size mismatch (5 -> 6 output neuron)
            old_weight = checkpoint['surface_classifier.conv4.weight']  # [5, 644, 1]
            old_bias = checkpoint['surface_classifier.conv4.bias']      # [5]
            
            new_weight = torch.zeros(6, 644, 1)
            new_bias = torch.zeros(6)
            
            new_weight[:5] = old_weight
            new_bias[:5] = old_bias
            
            # Random init for new neuron
            torch.nn.init.xavier_uniform_(new_weight[5:])
            
            checkpoint['surface_classifier.conv4.weight'] = new_weight
            checkpoint['surface_classifier.conv4.bias'] = new_bias
            
            # Add missing epsilon
            checkpoint['epsilon'] = netG.state_dict()['epsilon']
            
            netG.load_state_dict(checkpoint)

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    'NEW: log train error'
    loss_log = os.path.join(opt.checkpoints_path, opt.name, opt.name + '_train_loss.txt')
    with open(loss_log, 'w') as outfile:
        outfile.write("Training losses\n")
    'log loss weights'
    weight_log = os.path.join(opt.checkpoints_path, opt.name, opt.name + '_loss_weights.txt')
    with open(weight_log, 'w') as outfile:
        outfile.write("Loss weights\n")

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch, 0)
    losses_EWMA_prev = None
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            sample_tensor_uvwp = train_data['samples_uvwp'].to(device=cuda)
            sample_tensor_residual = train_data['samples_residual'].to(device=cuda)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
                sample_tensor_uvwp = reshape_sample_tensor(sample_tensor_uvwp, opt.num_views)
                sample_tensor_residual = reshape_sample_tensor(sample_tensor_residual, opt.num_views)

            label_tensor = train_data['labels'].to(device=cuda)
            ''' NEW: PINN read time step in TrainDataLoader and hand over to network here'''
            time_step_label = train_data['time_step'].to(device=cuda)
            label_tensor_u = train_data['labels_u'].to(device=cuda)
            label_tensor_v = train_data['labels_v'].to(device=cuda)
            label_tensor_w = train_data['labels_w'].to(device=cuda)
            label_tensor_p = train_data['labels_p'].to(device=cuda)

            iter_data_time = time.time()
            res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_eps, epsilon, loss_phi = netG.forward(
                image_tensor, sample_tensor, calib_tensor, labels=label_tensor, uvwp_points=sample_tensor_uvwp,
                residual_points=sample_tensor_residual, labels_u=label_tensor_u,
                labels_v=label_tensor_v, labels_w=label_tensor_w, labels_p=label_tensor_p, time_step=time_step_label,
                get_PINN_loss=True)

            ''' For now add loss for chemical potential to loss term for Cahn-Hilliard equation, but save phi loss for plotting'''
            loss_phi_log = loss_phi
            # apply global loss weights to each loss term
            losses = assign_global_weights_CH2(loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z, loss_eps, loss_phi, opt)
            #print('losses before weighting: ', losses)

            # learning rate onramp for u,v,w,p data loss terms
            # this is done because the other losses overweight the alpha loss in early training otherwise
            losses = get_data_loss_onramp(losses, train_idx, epoch, duration=opt.onramp)
            losses = get_pde_loss_onramp(losses, train_idx, epoch, duration=opt.onramp)

            if opt.loss_softadapt:
                ''' Calculate loss weights with SoftAdapt on EWMA of losses 
                refresh every 100 iterations'''
                if train_idx == 0 and (epoch == 0 or epoch == opt.resume_epoch):
                    losses_EWMA = losses
                    loss_weights = torch.ones_like(losses) * 1.0
                if train_idx % 10 == 0:
                    losses_EWMA = get_EWMA(losses, losses_EWMA, train_idx, epoch, opt)
                    #print([f"{x:.3e}" for x in losses_EWMA.cpu().numpy()])
                if train_idx % 1000 == 0: #100 or 1000
                    if losses_EWMA_prev is None:
                        losses_EWMA_prev = losses_EWMA
                    else:
                        loss_weights = get_loss_weights_SoftAdapt(losses_EWMA, losses_EWMA_prev)
                        losses_EWMA_prev = losses_EWMA
                losses = loss_weights * losses
            elif opt.loss_inverse_weight:
                ''' Calculate loss weights with inverse relative magnitude
                refresh every iteration'''
                losses_EWMA = torch.zeros_like(losses)
                if train_idx <= 0 and epoch == 0:
                    loss_weights = torch.ones_like(losses) * 1.0
                else:
                    loss_weights = get_loss_weights_Kiani(losses)
                losses = loss_weights * losses

            # backpropagation step
            loss_total = torch.sum(losses)
            optimizerG.zero_grad()
            loss_total.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                loss_log_s = 'Name: {0} | Epoch: {1} | {2}/{3} | L_t: {4:.06f} | L_a: {5:.06f} | L_u: {6:.06f} | L_v: {7:.06f} | L_w: {8:.06f} | L_p: {9:.06f} | L_c: {10:.9f} | L_ph: {11:.9f} | L_mom_x: {12:.9f} | L_mom_y: {13:.9f} | L_mom_z: {14:.9f} | LR: {15:.06f} | alpha_EWMA: {16:.06f} | Sigma: {17:.02f} | loss eps: {18:.09f} | epsilon: {19:.09f} | loss phi: {20:.09f} | dataT: {21:.05f} | netT: {22:.05f} | ETA: {23:02d}:{24:02d}\n'.format(
                    opt.name, epoch, train_idx, len(train_data_loader), loss_total.item(),
                    loss_data_alpha.item(), loss_data_u.item(), loss_data_v.item(), loss_data_w.item(),
                    loss_data_p.item(), loss_conti.item(),
                    loss_phase_conv.item(), loss_momentum_x.item(), loss_momentum_y.item(), loss_momentum_z.item(),
                    lr, losses_EWMA[:1].item(), opt.sigma, loss_eps, epsilon.item(), loss_phi_log.item(), iter_data_time - iter_start_time,
                                                        iter_net_time - iter_data_time, int(eta // 60),
                    int(eta - 60 * (eta // 60)))
                print(loss_log_s)
                with open(loss_log, 'a') as outfile:
                    outfile.write(loss_log_s)

                weight_log_s = 'w_a: {0:.06f} | w_u: {1:.06f}| w_v: {2:.06f} | w_w: {3:.06f} | w_p: {4:.06f} | w_cont: {5:.06f} | w_phase: {6:.06f} | w_nse_x: {7:.06f} | w_nse_y: {8:.9f} | w_nse_z: {9:.9f} | w_eps: {10:.9f} | w_phi: {11:.9f}\n'.format(
                    loss_weights[0].item(), loss_weights[1].item(), loss_weights[2].item(), loss_weights[3].item(),
                    loss_weights[4].item(), loss_weights[5].item(), loss_weights[6].item(), loss_weights[7].item(),
                    loss_weights[8].item(), loss_weights[9].item(), loss_weights[10].item(), loss_weights[11].item())
                with open(weight_log, 'a') as outfile:
                    outfile.write(weight_log_s)

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            iter_data_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        IOU_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.resume_epoch) + '_IOU.txt')
        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (validation) ...')
                test_errors = calc_error(opt, netG, cuda, test_dataset, 100)
                str_err_test = 'Val | MSE_a: {0:06f} | MSE_u: {1:06f} | MSE_v: {2:06f} | MSE_w: {3:06f} | MSE_p: {4:06f} | MSE_c: {' \
                               '5:06f} | MSE_ph: {6:06f} | MSE_nse_x: {7:06f} | MSE_nse_y: {8:06f} | MSE_nse_z: {9:06f} | IOU: {10:06f} | prec: {11:06f} | recall: {' \
                               '12:06f}\n'.format(*test_errors)
                print(str_err_test)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_test)
                MSE_a, MSE_u, MSE_v, MSE_w, MSE_p, MSE_c, MSE_ph, MSE_nse_x, MSE_nse_y, MSE_nse_z, IOU, prec, recall = test_errors
                test_losses['MSE(val)'] = MSE_a
                test_losses['IOU(val)'] = IOU
                test_losses['prec(val)'] = prec
                test_losses['recall(val)'] = recall

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_errors = calc_error(opt, netG, cuda, train_dataset, 100)
                train_dataset.is_train = True
                str_err_train = 'Train | MSE_a: {0:06f} | MSE_u: {1:06f} | MSE_v: {2:06f} | MSE_w: {3:06f} | MSE_p: {4:06f} | MSE_c: {' \
                                '5:06f} | MSE_ph: {6:06f} | MSE_nse_x: {7:06f} | MSE_nse_y: {8:06f} | MSE_nse_z: {9:06f} | IOU: {10:06f} | prec: {11:06f} | recall: {' \
                                '12:06f}\n'.format(*train_errors)
                print(str_err_train)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_train)
                MSE_a, MSE_u, MSE_v, MSE_w, MSE_p, MSE_c, MSE_ph, MSE_nse_x, MSE_nse_y, MSE_nse_z, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE_a
                test_losses['IOU(train)'] = IOU
                test_losses['prec(train)'] = prec
                test_losses['recall(train)'] = recall

            if not opt.no_gen_mesh:
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = random.choice(test_dataset)
                    save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, test_data['name'])
                    gen_mesh(opt, netG, cuda, test_data, save_path)

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = random.choice(train_dataset)
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    gen_mesh(opt, netG, cuda, train_data, save_path)
                train_dataset.is_train = True


if __name__ == '__main__':
    train(opt)
