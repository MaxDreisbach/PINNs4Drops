import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *
from lib.model.HGPIFuNet_CH2 import HGPIFuNet_CH2

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options
opt = BaseOptions().parse()

class Evaluator:
    def __init__(self, opt, projection_mode='orthogonal'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        #netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        netG = HGPIFuNet_CH2(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            print('loading for net G ...', opt.load_netG_checkpoint_path)
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_image(self, image_paths, mask_paths, time_paths, id):
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1

        # Name
        img_name = os.path.splitext(os.path.basename(image_paths[id]))[0]

        # get previous, current and next image with respective mask
        if self.opt.num_views == 3:
            mask_paths = mask_paths[id-1:id+2]
            image_paths = image_paths[id - 1:id + 2]
        elif self.opt.num_views == 5:
            mask_paths = mask_paths[id - 2:id + 3]
            image_paths = image_paths[id - 2:id + 3]
        elif self.opt.num_views == 7:
            mask_paths = mask_paths[id - 3:id + 4]
            image_paths = image_paths[id - 3:id + 4]
        else:
            mask_paths = mask_paths[id - 1:id + 2]
            image_paths = image_paths[id - 1:id + 2]

        image_list = []
        mask_list = []
        calib_list = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            print(image_path)
            # Mask
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()

            # image
            image = Image.open(image_path).convert('RGB')
            image = self.to_tensor(image)
            image = mask.expand_as(image) * image

            # calib
            calib = torch.Tensor(projection_matrix).float()

            mask_list.append(mask)
            image_list.append(image)
            calib_list.append(calib)

        image_tensor = torch.stack(image_list, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)
        calib_tensor = torch.stack(calib_list, dim=0)

        #time step
        time_path = time_paths[id]
        with open(time_path) as f:
            t = f.readline().strip('\n')
        timestep = torch.tensor([float(t)])

        return {
            'name': img_name,
            'calib': calib_tensor,
            'b_min': B_MIN,
            'b_max': B_MAX,
            'img': image_tensor,
            'mask': mask_tensor,
            'time_step': timestep
        }

    def eval(self, data, gen_3D_iso=False, use_octree=False, gen_vel_pres=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            if self.netC:
                gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            else:
                gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree, gen_vel_pres=gen_vel_pres, gen_3D_iso=gen_3D_iso)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_images.sort()
    #test_images = test_images[14:]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]
    test_time_labels = [f[:-4] + '_time.txt' for f in test_images]

    print("number of images to be reconstructed: ", len(test_masks))

    for id in range(len(test_images)):
        # spare out first and last image for temporal processing
        if opt.num_views == 3:
            if id == 0 or id == len(test_images) - 1:
                print('skipping time step ', id)
                continue
            else:
                data = evaluator.load_image(test_images, test_masks, test_time_labels, id)
                evaluator.eval(data, gen_3D_iso=opt.plot3D, use_octree=False, gen_vel_pres=True)
        elif opt.num_views == 5:
            if id <= 1 or id >= len(test_images) - 2:
                print('skipping time step ', id)
                continue
            else:
                data = evaluator.load_image(test_images, test_masks, test_time_labels, id)
                evaluator.eval(data, gen_3D_iso=opt.plot3D, use_octree=False, gen_vel_pres=True)
        elif opt.num_views == 7:
            if id <= 2 or id >= len(test_images) - 3:
                print('skipping time step ', id)
                continue
            else:
                data = evaluator.load_image(test_images, test_masks, test_time_labels, id)
                evaluator.eval(data, gen_3D_iso=opt.plot3D, use_octree=False, gen_vel_pres=True)
        else:
            if id == 0 or id == len(test_images) - 1:
                continue
            else:
                data = evaluator.load_image(test_images, test_masks, test_time_labels, id)
                evaluator.eval(data, gen_3D_iso=opt.plot3D, use_octree=False, gen_vel_pres=True)


