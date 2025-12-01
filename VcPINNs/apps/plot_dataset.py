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
from lib.data.EvalDatasetPlot import *
from lib.model import *
from lib.plotting import *
from lib.geometry import index

# get options
opt = BaseOptions().parse()

SLICE_DIM1 = 'z'
SLICE_DIM2 = 'x'

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = EvalDataset2(opt, phase='test', slice_dim=SLICE_DIM1)
    test_dataset = EvalDataset2(opt, phase='test', slice_dim=SLICE_DIM2)

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=1, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
                                   
    print('dataset size: ', len(train_data_loader))

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
                                  
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)
                                  
    plot_all_fields(opt, train_dataset, 711, slice_dim=SLICE_DIM1, ds='test', plot_results=True)
    plot_all_fields(opt, test_dataset, 711, slice_dim=SLICE_DIM2, ds='test', plot_results=True)



def plot_all_fields(opt, dataset, num_tests, slice_dim='z', ds='test', plot_results=False):
    # Read fluid properties and simulation domain
    with open(os.path.join(opt.dataroot, "flow_case.json"), "r") as f:
        flow_case = json.load(f)

    # non-dimensionalize the label data
    U_ref = flow_case["U_0"]  # impact velocity
    L_ref = flow_case["rp"]  # image reproduction scale -> domain size
    rho_ref = flow_case["rho_1"]  # density of liquid phase (water)
    

    if num_tests > len(dataset):
        num_tests = len(dataset)
    for idx in range(num_tests):
        idx = idx + 304
        data = dataset[idx]

        # retrieve the data
        name = data['name']
        print('Processing:', name)

        image_tensor = data['img']
        calib_tensor = data['calib']
        sample_tensor = data['samples'].unsqueeze(0)
        #sample_tensor_uvwp = data['samples_uvwp'].to(device=cuda).unsqueeze(0)
        #sample_tensor_residual = data['samples_residual'].to(device=cuda).unsqueeze(0)
        sample_tensor_uvwp = None
        sample_tensor_residual = None

        if opt.num_views > 1:
            sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

        label_tensor = data['labels'].unsqueeze(0)
        label_tensor_u = data['labels_u'].unsqueeze(0)
        label_tensor_v = data['labels_v'].unsqueeze(0)
        label_tensor_w = data['labels_w'].unsqueeze(0)
        label_tensor_p = data['labels_p'].unsqueeze(0)
        time_step_label = data['time_step']

        if opt.num_views > 1:
            # pick middle frame
            calib_tensor = calib_tensor[opt.num_views // 2: opt.num_views // 2 + 1, :, :]
            #print('calib_tensor shape: ', calib_tensor.size())

        labels_u_proj, labels_w_proj = project_velocity_vector_field(label_tensor_u, label_tensor_w,
                                                                     calib_tensor)
        ground_plot = -10000                                      
        y = sample_tensor[0, 1, :]
        label_tensor = label_tensor[0, 0, y > ground_plot]

        
        if sample_tensor_uvwp is not None:
            y = sample_tensor_uvwp[0, 1, :]
        else:
            y = sample_tensor[0, 1, :]
        labels_u_proj = labels_u_proj[0, y > ground_plot]
        label_tensor_v = label_tensor_v[0, y > ground_plot]
        labels_w_proj = labels_w_proj[0, y > ground_plot]
        label_tensor_p = label_tensor_p[0, y > ground_plot]
        sample_x = sample_tensor[0, 0, y > ground_plot]
        sample_y = sample_tensor[0, 1, y > ground_plot]
        sample_z = sample_tensor[0, 2, y > ground_plot]
                                                    

        # retrieve dimensional data for u,v,w,p
        EVAL_DROPLET_ONLY = False
        if EVAL_DROPLET_ONLY:
            u_pred = label_tensor * labels_u_proj * U_ref
            v_pred = label_tensor * label_tensor_v * U_ref
            w_pred = label_tensor * labels_w_proj * U_ref
            p_pred = label_tensor * label_tensor_p * rho_ref * U_ref ** 2
            pred_dim = torch.stack((label_tensor, u_pred, v_pred, w_pred, p_pred))

            u_lab = label_tensor * labels_u_proj * U_ref
            v_lab = label_tensor * label_tensor_v * U_ref
            w_lab = label_tensor * labels_w_proj * U_ref
            p_lab = label_tensor * label_tensor_p * rho_ref * U_ref ** 2
        else:
            u_pred = labels_u_proj * U_ref
            v_pred = label_tensor_v * U_ref
            w_pred = labels_w_proj * U_ref
            p_pred = label_tensor_p * rho_ref * U_ref ** 2
            pred_dim = torch.stack((label_tensor, u_pred, v_pred, w_pred, p_pred))

            u_lab = labels_u_proj * U_ref
            v_lab = label_tensor_v * U_ref
            w_lab = labels_w_proj * U_ref
            p_lab = label_tensor_p * rho_ref * U_ref ** 2
            

        res = pred_dim[0, :]

        print('u field mean: ', u_lab.mean().item(), 'max: ', u_lab.max().item(), 'min: ', u_lab.min().item())
        print('v field mean: ', v_lab.mean().item(), 'max: ', v_lab.max().item(), 'min: ', v_lab.min().item())
        print('w field mean: ', w_lab.mean().item(), 'max: ', w_lab.max().item(), 'min: ', w_lab.min().item())
        print('p field mean: ', p_lab.mean().item(), 'max: ', p_lab.max().item(), 'min: ', p_lab.min().item())

        if plot_results:
            sample_tensor = torch.stack((sample_x, sample_y, sample_z))
            # plot compound prediction (pressure contours, alpha contour line, velocity vector field)
            plot_compound(opt, sample_tensor, pred_dim, label_tensor, u_lab, v_lab, w_lab, p_lab, slice_dim, 'compound',
                                 'pres', name, ds)

            # plot error in alpha field               
            plot_contour(opt, sample_tensor, label_tensor, label_tensor, slice_dim, 'alpha', 'alpha', name, ds)

            #plot velocity errors
            plot_contour_w_alpha(opt, sample_tensor, u_pred, res, u_lab, label_tensor, slice_dim, 'u',
                                 'vel', name, ds)
            plot_contour_w_alpha(opt, sample_tensor, v_pred, res, v_lab, label_tensor, slice_dim, 'v',
                                 'vel', name, ds)
            plot_contour_w_alpha(opt, sample_tensor, w_pred, res, w_lab, label_tensor, slice_dim, 'w',
                                 'vel', name, ds)

            #plot error in pressure field
            plot_contour_w_alpha(opt, sample_tensor, p_pred, res, p_lab, label_tensor, slice_dim, 'p',
                                 'pres', name, ds)

            # plot 3D-contours
            #plot_iso_surface(opt, sample_tensor, res_PINN[0], 'alpha', name, ds)
            #plot_iso_surface(opt, sample_tensor, res_PINN[1], 'u', name, ds)
            #plot_iso_surface(opt, sample_tensor, res_PINN[2], 'v', name, ds)
            #plot_iso_surface(opt, sample_tensor, res_PINN[3], 'w', name, ds)
            #plot_iso_surface(opt, sample_tensor, res_PINN[4], 'p', name, ds)


if __name__ == '__main__':
    train(opt)
