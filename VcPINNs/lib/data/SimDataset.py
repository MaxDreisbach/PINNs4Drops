from torch.utils.data import Dataset
import numpy as np
import os
import random
import json
import time
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from natsort import natsorted
from lib.sample_util import *
from lib.sdf import *

log = logging.getLogger('trimesh')
log.setLevel(40)
PLOTTING = False
PLOT_SAMPLING = False
PLOT_INPUT = False


class SimDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'
        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        # Path setup
        self.root = self.opt.dataroot
        if self.opt.RGB:
            self.RENDER = os.path.join('../PIFu-master/train_data_DFS2023C', 'RENDER')
        else:
            self.RENDER = os.path.join(self.root, 'RENDER')

        self.MASK = os.path.join('../PINN-PIFu/train_data_DFS2024D/MASK')
        self.OBJ = os.path.join('../PIFu-master/train_data_DFS2023C', 'GEO', 'OBJ')
        self.PARAM = os.path.join('../PIFu-master/train_data_DFS2023C/PARAM')
        if self.is_train:
            print('Path setup: \n', self.RENDER, ' \n', self.MASK, ' \n', self.OBJ, ' \n', self.PARAM)

        self.UV_MASK = os.path.join('../PIFu-master/train_data_DFS2023C/UV_MASK')
        self.UV_NORMAL = os.path.join('../PIFu-master/train_data_DFS2023C/UV_NORMAL')
        self.UV_RENDER = os.path.join('../PIFu-master/train_data_DFS2023C/UV_RENDER')
        self.UV_POS = os.path.join('../PIFu-master/train_data_DFS2023C/UV_POS')

        self.VEL = os.path.join('../PINN-PIFu/train_data_DFS2024D', 'VEL')
        self.PRES = os.path.join('../PINN-PIFu/train_data_DFS2024D', 'PRES')
        self.TIME = os.path.join('../PINN-PIFu/train_data_DFS2024D', 'TIME')

        self.B_MIN = np.array([-1, -1, -1])
        self.B_MAX = np.array([1, 1, 1])

        # for PINN (u,v,w,p) data loss term
        self.n_data = self.opt.n_data
        self.n_residual = self.opt.n_residual
        self.small_data_partition = self.opt.small_data_partition

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        # NEW: Changed for single image processing
        self.yaw_list = list(range(0, 360, 10))
        # self.yaw_list = [0]
        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = []


    def get_subjects(self):
        all_subjects = os.listdir(self.RENDER)
        val_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        eval_subjects = np.loadtxt(os.path.join(self.root, 'eval.txt'), dtype=str)
        test_subjects = np.loadtxt(os.path.join(self.root, 'test.txt'), dtype=str)

        if len(val_subjects) == 0:
            return natsorted(list(set(all_subjects) - set(test_subjects)))

        if self.is_train:
            return natsorted(list(set(all_subjects)))
        else:
            return natsorted(list(eval_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)


    def get_subjects_list(self, sid, num_views):
        '''
        fetches adjacent time steps in the whole dataset for a given id in the training dataset
        '''

        assert num_views % 2 == 1, "num_views must be an odd integer."
        all_subjects = natsorted(os.listdir(self.RENDER))
        subject_main = self.subjects[sid]
        global_id = all_subjects.index(subject_main)
        len_subjects = 1014 # hardcoded length for structured DS


        if global_id <= 254:
            step = 10 #time step 0.015ms (Fink 2018 data)
        else:
            step = 1 #time step 0.15ms


        half_window = num_views // 2
        start_idx = global_id - half_window * step
        end_idx = global_id + half_window * step

        # Clamp start and end if they go out of bounds
        #if start_idx < 0:
        #    start_idx = 0
        #    end_idx = start_idx + (num_views - 1) * step
        #if end_idx >= len_subjects:
        #    end_idx = len_subjects - 1
        #    start_idx = end_idx - (num_views - 1) * step
        #    if start_idx < 0:  # In case total_subjects is too small
        #        start_idx = 0

        id_list = list(range(start_idx, end_idx + 1, step))
        subject_list = []
        for id in id_list:
            subject_list.append(all_subjects[id])

        #print(subject_list)
        return subject_list


    def get_render(self, subject, num_views, sid, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch = self.pitch_list[pid]
        yaw = self.yaw_list[yid]
        subject_list = self.get_subjects_list(sid, num_views)
        # print(subject_list)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        # augmentation parameters for all consecutive frames
        rand_scale = random.uniform(0.9, 1.1)
        rand_x = random.uniform(-0.1, 0.1)
        rand_y = random.uniform(-0.1, 0.1)

        for subject in subject_list:
            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (yaw, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.png' % (yaw, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (yaw, pitch, 0))

            # loading calibration data
            param = np.load(param_path, allow_pickle=True)
            # pixel unit / world unit
            ortho_ratio = param.item().get('ortho_ratio')
            # world unit / model unit
            scale = param.item().get('scale')
            # camera center world coordinate
            center = param.item().get('center')
            # model rotation
            R = param.item().get('R')

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            scale_intrinsic = np.identity(4)
            scale_intrinsic[0, 0] = scale / ortho_ratio
            '''Y-axis flipped for gridsample in index() in geometry.py -> needs to be flipped back later 
            to keep PINN coordinates consistent'''
            scale_intrinsic[1, 1] = -scale / ortho_ratio
            scale_intrinsic[2, 2] = scale / ortho_ratio
            # Match image pixel space to image uv space
            uv_intrinsic = np.identity(4)
            uv_intrinsic[0, 0] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[1, 1] = 1.0 / float(self.opt.loadSize // 2)
            uv_intrinsic[2, 2] = 1.0 / float(self.opt.loadSize // 2)
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if self.opt.random_flip and np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if self.opt.random_scale:
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if self.opt.random_trans:
                    dx = int(round(rand_x * (w - tw)))
                    dy = int(round(rand_y * (h - th)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
                trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                if self.opt.aug_blur > 0.00001:
                    blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                    render = render.filter(blur)

            intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }


    def select_sampling_method(self, subject):
        '''
        returns samples and labels for (alpha,u,v,w,p) in B_MIN,B_MAX - [256,256,256] domain
        '''
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)

        mesh = trimesh.load(os.path.join(self.OBJ, subject, '%s.obj' % subject))
        # verts transformed into computational domain, i.e. ([-128:128, -28:228, -128:128] -> [-1:1, -1:1, -1:1])
        verts = (mesh.vertices - 125.25) / 128
        faces = mesh.faces

        coords, mat = create_grid(self.opt.resolution, self.opt.resolution, self.opt.resolution,
                                  self.B_MIN, self.B_MAX, transform=None)
                                                       
        samples = coords.reshape(3, -1)
        samples = torch.Tensor(samples).float()
        samplesT = samples.T

        # Load (u,v,w,p) fields for PINN
        # transformation (rotation, translation, scaling) is handled in HGPIFuNet Projection -> geometry.py
        u_grid = np.load(os.path.join(self.VEL, subject, 'u_train.npy'), allow_pickle=True)
        v_grid = np.load(os.path.join(self.VEL, subject, 'v_train.npy'), allow_pickle=True)
        w_grid = np.load(os.path.join(self.VEL, subject, 'w_train.npy'), allow_pickle=True)
        p_grid = np.load(os.path.join(self.PRES, subject, 'p_train.npy'), allow_pickle=True)
        # for validation only
        c_grid = np.load(os.path.join(self.VEL, subject, 'c_train.npy'), allow_pickle=True)
        
        # Read fluid properties and simulation domain
        with open(os.path.join(self.root, "flow_case.json"), "r") as f:
            flow_case = json.load(f)

        # Creating new grid to map to - set the limits (x_min,x_max,etc.) and scale according with mesh processing
        x_min = flow_case["x_min"]/128
        x_max = flow_case["x_max"]/128
        y_min = (flow_case["y_min"] - 125.25)/128
        y_max = (flow_case["y_max"] - 125.25)/128
        y_ground = (flow_case["y_ground"] - 125.25)/128  # 60um from y0
        z_min = flow_case["z_min"]/128
        z_max = flow_case["z_max"]/128
        x = np.linspace(x_min, x_max, flow_case["x_res"])
        y = np.linspace(y_min, y_max, flow_case["y_res"])
        z = np.linspace(z_min, z_max, flow_case["z_res"])
        grid_points = (x, y, z)

        # actual interpolation from grid to continuous point cloud
        samplesT = np.transpose(samples, (1, 0))
        labels_u = interpn(grid_points, u_grid, samplesT[:self.n_data, :], bounds_error=False, fill_value=float('nan'))
        labels_v = interpn(grid_points, v_grid, samplesT[:self.n_data, :], bounds_error=False, fill_value=float('nan'))
        labels_w = interpn(grid_points, w_grid, samplesT[:self.n_data, :], bounds_error=False, fill_value=float('nan'))
        labels_p = interpn(grid_points, p_grid, samplesT[:self.n_data, :], bounds_error=False, fill_value=float('nan'))
        # labels_c = interpn(grid_points, c_grid, samplesT[:self.n_vel_pres_data, :], bounds_error=False, fill_value=0)
        
        # Utility: print NaN stats for numpy arrays
        def print_nan_stats_np(array, name):
            total = array.size
            nan_count = np.isnan(array).sum()
            valid_count = total - nan_count
            print(f"[DEBUG] {name}:")
            print(f"    Total elements       : {total}")
            print(f"    NaN elements         : {nan_count}")
            print(f"    Valid elements       : {valid_count}")
            print(f"    Valid %              : {100.0 * valid_count / total:.2f}%")
            print("-" * 40)
                
        # DEBUG: Check how many NaNs resulted from out-of-bounds sampling
        #print_nan_stats_np(labels_u, "labels_u")
        #print_nan_stats_np(labels_v, "labels_v")
        #print_nan_stats_np(labels_w, "labels_w")
        #print_nan_stats_np(labels_p, "labels_p")
        
        # rotate vector field to match PINN domain (x,y,z) -> (y,z,x)
        # Careful! this depends on the definition of the KOS and might be different for other datasets
        labels_u_r = labels_u
        labels_v_r = labels_w
        labels_w_r = labels_v

        # from C [-1, 1] -> alpha [0, 1]
        labels = (labels_c + 1) / 2
        
        # Plotting for debug
        PLOTTING = False
        #PLOTTING = True
        if PLOTTING:
            mask = (labels > 0.5)
            #print(mask.shape)
            #print(samples.shape)
            #print(labels_u_r.shape)
            c_drop = labels_c[mask]
            u_drop = labels_u_r[mask]
            v_drop = labels_v_r[mask]
            w_drop = labels_w_r[mask]
            p_drop = labels_p[mask]
            s_drop = samples[:, mask]
        
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            x = s_drop[0, :]
            y = s_drop[1, :]
            z = s_drop[2, :]
        
            plot_half = True
            if plot_half:
                x = x[z < 0]
                y = y[z < 0]
                c_drop = c_drop[z < 0]
                u_drop = u_drop[z < 0]
                v_drop = v_drop[z < 0]
                w_drop = w_drop[z < 0]
                p_drop = p_drop[z < 0]
                z = z[z < 0]
        
            # mappable = ax.scatter(x, y, z, s=10, c=labels_p, vmin=-500, vmax=500, cmap='viridis')
            # mappable = ax.scatter(x, y, z, s=10, c=labels_u, cmap='coolwarm')
            mappable = ax.scatter(x, y, z, s=10, c=u_drop, cmap='coolwarm')
            plt.colorbar(mappable)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            ax.set_box_aspect((1, 1, 0.5))
            plt.show()
        
        ''' time step read in'''
        timestep_path = os.path.join(self.TIME, subject, 'time_step.txt')
        with open(timestep_path) as f:
            t = f.readline().strip('\n')

        timestep = torch.tensor([float(t)])

        del mesh
        return {
            'verts': verts,
            'faces': faces,
            'coords': coords,
            'labels': torch.tensor(labels, dtype=torch.float32),
            'labels_u': torch.tensor(labels_u_r, dtype=torch.float32),
            'labels_v': torch.tensor(labels_v_r, dtype=torch.float32),
            'labels_w': torch.tensor(labels_w_r, dtype=torch.float32),
            'labels_p': torch.tensor(labels_p, dtype=torch.float32),
            'time_step': timestep
        }


    def get_item(self, index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        subject = self.subjects[sid]

        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, sid=sid, yid=yid, pid=pid,
                                      random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        if PLOT_SAMPLING:
            debug_sampling_points(render_data, sample_data)

        if PLOT_INPUT:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                img = np.uint8((np.transpose(render_data['img'][i].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[i].imshow(img, cmap='gray', interpolation='bicubic')
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_title(f'Image {i + 1}')  # Optional: title for each subplot

            plt.tight_layout()
            plt.show()

        return res


    def __getitem__(self, index):
        return self.get_item(index)
