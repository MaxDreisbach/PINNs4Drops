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
from lib.data import *
from lib.model import *
from lib.model.HGPIFuNet_CH import HGPIFuNet_CH
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

    # create net
    #netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    netG = HGPIFuNet_CH(opt, projection_mode).to(device=cuda)
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    # optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate, amsgrad=True)

    # optimizer variants for partly frozen network (hourglas feature extraction frozen)
    # optimizerG = torch.optim.RMSprop(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.learning_rate, momentum=0, weight_decay=0)
    # optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.learning_rate, amsgrad=True)

    lr = opt.learning_rate
    print('Using Network: ', netG.name)

    def set_train():
        netG.train()

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    if opt.continue_train:
        if opt.resume_epoch < 0:
            model_path = '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)
        netG.load_state_dict(torch.load(model_path, map_location=cuda))

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

    # training
    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch, 0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor = train_data['labels'].to(device=cuda)
            ''' NEW: PINN read time step in TrainDataLoader and hand over to network here'''
            time_step_label = train_data['time_step'].to(device=cuda)
            label_tensor_u = train_data['labels_u'].to(device=cuda)
            label_tensor_v = train_data['labels_v'].to(device=cuda)
            label_tensor_w = train_data['labels_w'].to(device=cuda)
            label_tensor_p = train_data['labels_p'].to(device=cuda)

            iter_data_time = time.time()
            res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z = netG.forward(
                image_tensor, sample_tensor, calib_tensor, labels=label_tensor, labels_u=label_tensor_u,
                labels_v=label_tensor_v, labels_w=label_tensor_w, labels_p=label_tensor_p, time_step=time_step_label,
                get_PINN_loss=True)

            # apply weighting to loss terms
            loss_data_u = opt.weight_u * loss_data_u
            loss_data_v = opt.weight_v * loss_data_v
            loss_data_w = opt.weight_w * loss_data_w
            loss_data_p = opt.weight_p * loss_data_p
            loss_conti = opt.weight_conti * loss_conti
            loss_phase_conv = opt.weight_phase * loss_phase_conv
            loss_momentum_x = opt.weight_mom_x * loss_momentum_x
            loss_momentum_y = opt.weight_mom_y * loss_momentum_y
            loss_momentum_z = opt.weight_mom_z * loss_momentum_z

            ''' learning rate annealing for u,v,w,p data loss terms
                        # this is done because the other losses overweight the alpha loss in early training otherwise'''
            if train_idx < 1000 and epoch == 0:
                loss_data_u = loss_data_u * (train_idx / 1000)
                loss_data_v = loss_data_w * (train_idx / 1000)
                loss_data_w = loss_data_w * (train_idx / 1000)
                loss_data_p = loss_data_p * (train_idx / 1000)

            ''' turn on PDE losses after initializing network only with alpha field for a few iterations
            calculate exponentially weighted moving average of alpha loss and 
            active PDE losses after alpha loss has dropped below threshold thres_alpha'''
            thres_alpha = 0.03
            beta = 0.9

            if train_idx == 0 and epoch == 0:
                l_alpha_EWMA = 0.25
                PDE_LOSS = False

            if train_idx == 0 and epoch==opt.resume_epoch:
                l_alpha_EWMA = 0.0300001
                PDE_LOSS = False

            l_alpha_EWMA = beta * l_alpha_EWMA + (1 - beta) * loss_data_alpha

            if l_alpha_EWMA <= thres_alpha and PDE_LOSS == False:
                it_start_pde_loss = train_idx
                PDE_LOSS = True

            if PDE_LOSS == True:
                it_pde_anneal = train_idx - it_start_pde_loss
                ''' learning rate annealing for pde loss terms'''
                if it_pde_anneal < 5000:
                    loss_conti = loss_conti * (it_pde_anneal / 5000)
                    loss_phase_conv = loss_phase_conv * (it_pde_anneal / 5000)
                    loss_momentum_x = loss_momentum_x * (it_pde_anneal / 5000)
                    loss_momentum_y = loss_momentum_y * (it_pde_anneal / 5000)
                    loss_momentum_z = loss_momentum_z * (it_pde_anneal / 5000)
                else:
                    '''Do nothing -> global weights for pde losses'''
            else:
                loss_conti = loss_conti * 0
                loss_phase_conv = loss_phase_conv * 0
                loss_momentum_x = loss_momentum_x * 0
                loss_momentum_y = loss_momentum_y * 0
                loss_momentum_z = loss_momentum_z * 0

            loss_total = loss_data_alpha + loss_data_u + loss_data_v + loss_data_w + loss_data_p + loss_conti + loss_phase_conv + loss_momentum_x + loss_momentum_y + loss_momentum_z

            optimizerG.zero_grad()
            loss_total.backward()
            optimizerG.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                loss_log_s = 'Name: {0} | Epoch: {1} | {2}/{3} | L_t: {4:.06f} | L_a: {5:.06f} | L_u: {6:.06f} | L_v: {7:.06f} | L_w: {8:.06f} | L_p: {9:.06f} | L_c: {10:.9f} | L_ph: {11:.9f} | L_mom_x: {12:.9f} | L_mom_y: {13:.9f} | L_mom_z: {14:.9f} | LR: {15:.06f} | alpha_EWMA: {16:.06f} | Sigma: {17:.02f} | dataT: {18:.05f} | netT: {19:.05f} | ETA: {20:02d}:{21:02d}\n'.format(
                    opt.name, epoch, train_idx, len(train_data_loader), loss_total.item(),
                    loss_data_alpha.item(), loss_data_u.item(), loss_data_v.item(), loss_data_w.item(),
                    loss_data_p.item(), loss_conti.item(),
                    loss_phase_conv.item(), loss_momentum_x.item(), loss_momentum_y.item(), loss_momentum_z.item(),
                    lr, l_alpha_EWMA.item(), opt.sigma, iter_data_time - iter_start_time,
                                                        iter_net_time - iter_data_time, int(eta // 60),
                    int(eta - 60 * (eta // 60)))
                print(loss_log_s)
                with open(loss_log, 'a') as outfile:
                    outfile.write(loss_log_s)

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                torch.save(netG.state_dict(), '%s/%s/netG_latest' % (opt.checkpoints_path, opt.name))
                torch.save(netG.state_dict(), '%s/%s/netG_epoch_%d' % (opt.checkpoints_path, opt.name, epoch))

            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                r = res[0].cpu()
                points = sample_tensor[0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())

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
                str_err_test = 'eval val MSE_t: {0:06f} | MSE_a: {1:06f}| MSE_v: {2:06f}| MSE_p: {3:06f}| MSE_c: {' \
                               '4:06f}| MSE_ph: {5:06f}| MSE_nse: {6:06f} | IOU: {7:06f} | prec: {8:06f} | recall: {' \
                               '9:06f}\n'.format(*test_errors)
                print(str_err_test)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_test)
                MSE, MSE_a, MSE_v, MSE_p, MSE_c, MSE_ph, MSE_nse, IOU, prec, recall = test_errors
                test_losses['MSE(val)'] = MSE
                test_losses['IOU(val)'] = IOU
                test_losses['prec(val)'] = prec
                test_losses['recall(val)'] = recall

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_errors = calc_error(opt, netG, cuda, train_dataset, 100)
                train_dataset.is_train = True
                str_err_train = 'eval train MSE_t: {0:06f} | MSE_a: {1:06f}| MSE_v: {2:06f}| MSE_p: {3:06f}| MSE_c: {' \
                                '4:06f}| MSE_ph: {5:06f}| MSE_nse: {6:06f} | IOU: {7:06f} | prec: {8:06f} | recall: {' \
                                '9:06f}\n'.format(*train_errors)
                print(str_err_train)
                with open(IOU_log, 'a') as outfile:
                    outfile.write(str_err_train)
                MSE, MSE_a, MSE_v, MSE_p, MSE_c, MSE_ph, MSE_nse, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE
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
