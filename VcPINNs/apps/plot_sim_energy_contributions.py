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
from lib.data.SimDataset import *
from lib.model import *
from lib.plotting import *
from lib.geometry import index
from lib.physics_util import *


# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = SimDataset(opt, phase='train')
    test_dataset = SimDataset(opt, phase='test')

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
    
    #calc_energy_contrib(opt, train_dataset, 1014, process_val_data=True)
    calc_energy_contrib(opt, test_dataset, 304, process_val_data=True)



def calc_energy_contrib(opt, dataset, num_tests, process_val_data=True):
    # Read fluid properties and simulation domain
    with open(os.path.join(opt.dataroot, "flow_case.json"), "r") as f:
        flow_case = json.load(f)

    U_ref = flow_case["U_0"]  # impact velocity
    L_ref = flow_case["rp"]  # image reproduction scale -> domain size
    rho_1 = flow_case["rho_1"]  # density of inside medium
    rho_2 = flow_case["rho_2"]  # density of outside medium
    sigma = flow_case["sigma"]  # surface tension
    g = flow_case["g"]  # gravity

    # Read experimental conditions
    with open(os.path.join(opt.test_folder_path, "exp_case.json"), "r") as f:
        exp_case = json.load(f)

    y_ground = exp_case["ground"]
    contact_angle = exp_case["theta_eq"]    

    if num_tests > len(dataset):
        num_tests = len(dataset)
    for idx in range(num_tests):
        #idx = idx + 50
        # retrieve the data
        data = dataset[idx]
        name = data['name']
        print('Processing:', name)
        save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])

        image_tensor = data['img']
        calib_tensor = data['calib']
        coords = data['coords']
        verts = data['verts']
        faces = data['faces']
        time_tensor = data['time_step']
        labels = data['labels'].unsqueeze(0)
        labels_u = data['labels_u'].unsqueeze(0)
        labels_v = data['labels_v'].unsqueeze(0)
        labels_w = data['labels_w'].unsqueeze(0)
        labels_p = data['labels_p'].unsqueeze(0)
        
        resolution = coords.shape[1:4]
        sdf = labels.reshape(resolution).numpy()
        u = labels_u.reshape(resolution).numpy()
        v = labels_v.reshape(resolution).numpy()
        w = labels_w.reshape(resolution).numpy()
        p = labels_p.reshape(resolution).numpy()
        #print("mean velocity:", np.mean(np.sqrt(u**2 + v**2 + w**2)))

        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for image in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[image].detach().cpu().clone().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        calc_energies = True
        if calc_energies:
            E_surf, E_kin, E_pot = calculate_energy_contributions(opt, coords, sdf, u, v, w, verts, faces, U_ref, L_ref, rho_1, rho_2, sigma, g, y_ground, contact_angle)
      
            'log energy contributions'
            energies_log = os.path.join(opt.name + '_energy_contributions.txt')
            log_message = 'Name: {0} | E_surf: {1} | E_kin: {2} | E_pot: {3} |\n'.format(
                save_path[:-4], E_surf, E_kin, E_pot)
            print(log_message)
            with open(energies_log, 'a') as outfile:
                outfile.write(log_message)


if __name__ == '__main__':
    train(opt)
