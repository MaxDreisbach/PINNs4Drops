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
import csv

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.plotting import *
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
                                  batch_size=1, shuffle =True,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('validation data size: ', len(test_data_loader))

    # create net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    print('Using Network: ', netG.name)

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.load_netG_checkpoint_path is not None:
        print('loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

    os.makedirs('%s/%s/%s' % (opt.checkpoints_path, opt.name, 'pred_fields'), exist_ok=True)
    os.makedirs('%s/%s/%s' % (opt.results_path, opt.name, 'pred_fields'), exist_ok=True)


    ''' actual plotting happens here'''
    with torch.no_grad():
        set_eval()
        print('calc error (validation) for steps ...')
        plot_timesteps = [0, 5, 10, 15, 50, 97, 103, 140]
        #plot_timesteps = [97, 103]

        dataset = []
        for idx in plot_timesteps:
            data = test_dataset[idx]
            print(data['name'])
            dataset.append(data)

        test_errors = calc_error(opt, netG, cuda, dataset, len(plot_timesteps), slice_dim='z', ds='test',
                                 plot_results=True)
        test_errors = calc_error(opt, netG, cuda, dataset, len(plot_timesteps), slice_dim='x', ds='test',
                                 plot_results=True)


        print('calc error (train) ...')
        train_dataset.is_train = False
        plot_timesteps = [0, 45, 90, 135, 450, 900]
        dataset = []
        for idx in plot_timesteps:
            data = train_dataset[idx]
            print(data['name'])
            dataset.append(data)

        test_errors = calc_error(opt, netG, cuda, dataset, len(plot_timesteps), slice_dim='z', ds='test',
                                 plot_results=True)
        test_errors = calc_error(opt, netG, cuda, dataset, len(plot_timesteps), slice_dim='x', ds='test',
                                 plot_results=True)


if __name__ == '__main__':
    train(opt)