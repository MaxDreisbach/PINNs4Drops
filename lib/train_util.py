import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .geometry import *
from .plotting import *
import cv2
from PIL import Image
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt


def reshape_multiview_tensors(image_tensor, calib_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1],
        image_tensor.shape[2],
        image_tensor.shape[3],
        image_tensor.shape[4]
    )
    calib_tensor = calib_tensor.view(
        calib_tensor.shape[0] * calib_tensor.shape[1],
        calib_tensor.shape[2],
        calib_tensor.shape[3]
    )

    return image_tensor, calib_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(opt, net, cuda, data, save_path, use_octree=True, gen_vel_pres=False):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    time_tensor = data['time_step'].to(device=cuda)

    #get non-dimensional time
    # Read fluid properties and simulation domain
    with open(os.path.join(opt.dataroot, "flow_case.json"), "r") as f:
        flow_case = json.load(f)

    # non-dimensionalize the label data
    U_ref = flow_case["U_0"]  # impact velocity
    L_ref = flow_case["rp"]  # image reproduction scale -> domain size
    timestep_dimless = time_tensor / (L_ref / U_ref)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']

    save_img_path = save_path[:-4] + '.png'
    save_img_list = []
    for v in range(image_tensor.shape[0]):
        save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

    verts, faces, coords, sdf, u, v, w, p, _, _ = reconstruction(
        net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree, time_step=timestep_dimless)

    if gen_vel_pres:
        #plot_contour_eval(coords, opt, u, 'x', 'u', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, v, 'x', 'v', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, w, 'x', 'w', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, p, 'x', 'p', 'pres', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, u, 'y', 'u', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, v, 'y', 'v', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, w, 'y', 'w', 'vel', save_path[:-4], 'test')
        #plot_contour_eval(coords, opt, p, 'y', 'p', 'pres', save_path[:-4], 'test')
        plot_contour_eval(coords, opt, u, sdf, 'z', 'u', 'vel', save_path[:-4])
        plot_contour_eval(coords, opt, v, sdf, 'z', 'v', 'vel', save_path[:-4])
        plot_contour_eval(coords, opt, w, sdf, 'z', 'w', 'vel', save_path[:-4])
        plot_contour_eval(coords, opt, p, sdf, 'z', 'p', 'pres', save_path[:-4])
        #gen_vtk_prediction(coords, u, 'u', save_path[:-4])
        #gen_vtk_prediction(coords, v, 'v', save_path[:-4])
        #gen_vtk_prediction(coords, w, 'w', save_path[:-4])
        #gen_vtk_prediction(coords, p, 'p', save_path[:-4])

    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
    uv = xyz_tensor[:, :2, :]
    color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
    color = color * 0.5 + 0.5
    save_obj_mesh_with_color(save_path, verts, faces, color)

def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt


def calc_error(opt, net, cuda, dataset, num_tests, slice_dim='z', ds='test', plot_results=False):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_arr, error_data_arr, error_vel_arr, error_pres_arr, error_conti_arr, error_phase_arr, error_nse_arr, IOU_arr, prec_arr, recall_arr = [], [], [], [], [], [], [], [], [], []
        for idx in range(num_tests):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            name = data['name']
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
            label_tensor_u = data['labels_u'].to(device=cuda).unsqueeze(0)
            label_tensor_v = data['labels_v'].to(device=cuda).unsqueeze(0)
            label_tensor_w = data['labels_w'].to(device=cuda).unsqueeze(0)
            label_tensor_p = data['labels_p'].to(device=cuda).unsqueeze(0)
            time_step_label = data['time_step'].to(device=cuda)

            res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, labels_u=label_tensor_u,
                   labels_v=label_tensor_v, labels_w=label_tensor_w, labels_p=label_tensor_p, time_step=time_step_label, get_PINN_loss=False)

            loss = loss_data_alpha + loss_data_u + loss_data_v + loss_data_w + loss_data_p + loss_conti + loss_phase_conv + loss_momentum_x +loss_momentum_z +loss_momentum_z
            IOU, prec, recall = compute_acc(res, label_tensor)

            if plot_results:
                labels_u_proj, labels_w_proj = project_velocity_vector_field(label_tensor_u, label_tensor_w,
                                                                             calib_tensor)
                # plot compound prediction (pressure contours, alpha contour line, velocity vector field)
                plot_compound(opt, sample_tensor, res_PINN, label_tensor[0, 0, :], labels_u_proj[0, :], label_tensor_v[0, :], label_tensor_w[0, :], label_tensor_p[0, :], slice_dim, 'compound',
                                     'pres', name, ds)

                # plot error in alpha field
                plot_contour(opt, sample_tensor, res_PINN[0, 0, :], label_tensor[0, 0, :], slice_dim, 'alpha', 'alpha', name, ds)

                #plot velocity errors
                plot_contour_w_alpha(opt, sample_tensor, res_PINN[0, 1, :], res[0, 0, :], labels_u_proj[0, :], label_tensor[0, 0, :], slice_dim, 'u',
                                     'vel', name, ds)
                plot_contour_w_alpha(opt, sample_tensor, res_PINN[0, 2, :], res[0, 0, :], label_tensor_v[0, :], label_tensor[0, 0, :], slice_dim, 'v',
                                     'vel', name, ds)
                plot_contour_w_alpha(opt, sample_tensor, res_PINN[0, 3, :], res[0, 0, :], labels_w_proj[0, :], label_tensor[0, 0, :], slice_dim, 'w',
                                     'vel', name, ds)

                #plot error in pressure field
                plot_contour_w_alpha(opt, sample_tensor, res_PINN[0, 4, :], res[0, 0, :], label_tensor_p[0, :], label_tensor[0, 0, :], slice_dim, 'p',
                                     'pres', name, ds)

                # plot 3D-contours
                #plot_iso_surface(opt, sample_tensor, res_PINN[0, 0, :], 'alpha', name, ds)
                #plot_iso_surface(opt, sample_tensor, res_PINN[0, 1, :], 'u', name, ds)
                #plot_iso_surface(opt, sample_tensor, res_PINN[0, 2, :], 'v', name, ds)
                #plot_iso_surface(opt, sample_tensor, res_PINN[0, 3, :], 'w', name, ds)
                #plot_iso_surface(opt, sample_tensor, res_PINN[0, 4, :], 'p', name, ds)

            print('{0}/{1}: {6} | Loss: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f} a_MSE: {7:06f} u_MSE: {8:06f} v_MSE: {9:06f} w_MSE: {10:06f} p_MSE: {11:06f}'.format(idx, num_tests, loss.item(), IOU.item(), prec.item(), recall.item(), name, loss_data_alpha.item(), loss_data_u.item(), loss_data_v.item(), loss_data_w.item(), loss_data_p.item()))
            error_arr.append(loss.item())
            error_data_arr.append(loss_data_alpha.item())
            loss_data_vel = loss_data_u + loss_data_v + loss_data_w
            error_vel_arr.append(loss_data_vel.item())
            error_pres_arr.append(loss_data_p.item())
            error_conti_arr.append(loss_conti.item())
            error_phase_arr.append(loss_phase_conv.item())
            loss_momentum = loss_momentum_x + loss_momentum_y + loss_momentum_z
            error_nse_arr.append(loss_momentum.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())
    print('All samples: {0}'.format(IOU_arr))
    return np.average(error_arr), np.average(error_data_arr), np.average(error_vel_arr), np.average(error_pres_arr), np.average(error_conti_arr), np.average(error_phase_arr), np.average(error_nse_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_error_color(opt, netG, netC, cuda, dataset, num_tests):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)

