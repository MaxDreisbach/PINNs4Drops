import torch
import numpy as np
from .mesh_util import *
from .sample_util import *
from .physics_util import *
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


def gen_mesh(opt, net, cuda, data, save_path, use_octree=False, gen_vel_pres=False, gen_3D_iso=False, process_val_data=False):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    time_tensor = data['time_step'].to(device=cuda)
    b_min = data['b_min']
    b_max = data['b_max']

    # Read fluid properties and simulation domain
    with open(os.path.join(opt.dataroot, "flow_case.json"), "r") as f:
        flow_case = json.load(f)

    #U_ref = flow_case["U_0"]  # impact velocity
    #L_ref = flow_case["rp"]  # image reproduction scale -> domain size
    rho_1 = flow_case["rho_1"]  # density of inside medium
    rho_2 = flow_case["rho_2"]  # density of outside medium
    sigma = flow_case["sigma"]  # surface tension
    g = flow_case["g"]  # gravity

    # Read experimental conditions
    with open(os.path.join(opt.test_folder_path, "exp_case.json"), "r") as f:
        exp_case = json.load(f)

    y_ground = exp_case["ground"]
    contact_angle = exp_case["theta_eq"]
    U_ref = exp_case["U_0"]  # impact velocity
    L_ref = exp_case["rp"]  # image reproduction scale -> domain size
    
    if process_val_data:
        y_ground = 0.140625
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib_tensor = torch.Tensor(projection_matrix).float().unsqueeze(0)
        calib_tensor = calib_tensor.repeat(opt.num_views, 1, 1).to(device=cuda)
        b_min = np.array([-1, -1, -1])
        b_max = np.array([1, 1, 1])
        # already dimless from dataloader
        timestep_dimless = time_tensor
    else:
        #get non-dimensional time
        timestep_dimless = time_tensor / (L_ref / U_ref)

    net.filter(image_tensor)

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
        if 'FDM' in opt.test_folder_path:
            plot_vel = 'vel_FDM'
            plot_pres = 'pres_FDM'
        else:
            plot_vel = 'vel_PDMS'
            plot_pres = 'pres_PDMS'
    
        plot_contour_eval(coords, opt, u, sdf, 'x', 'u', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, v, sdf, 'x', 'v', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, w, sdf, 'x', 'w', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, p, sdf, 'x', 'p', y_ground, plot_pres, save_path[:-4])
        plot_contour_eval(coords, opt, u, sdf, 'z', 'u', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, v, sdf, 'z', 'v', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, w, sdf, 'z', 'w', y_ground, plot_vel, save_path[:-4])
        plot_contour_eval(coords, opt, p, sdf, 'z', 'p', y_ground, plot_pres, save_path[:-4])

    calc_energies = True
    if calc_energies:
        E_surf, E_kin, E_pot = calculate_energy_contributions(opt, coords, sdf, u, v, w, verts, faces, U_ref, L_ref, rho_1, rho_2, sigma, g, y_ground, contact_angle, plot_diagnostic=False)

        'log energy contributions'
        energies_log = os.path.join(opt.name + '_energy_contributions.txt')
        log_message = 'Name: {0} | E_surf: {1} | E_kin: {2} | E_pot: {3} |\n'.format(
            save_path[:-4], E_surf, E_kin, E_pot)
        print(log_message)
        with open(energies_log, 'a') as outfile:
            outfile.write(log_message)

        # plot 3D-contours
    if gen_3D_iso:
        # only get prediction in liquid phase
        u = np.multiply(u, sdf)
        v = np.multiply(v, sdf)
        w = np.multiply(w, sdf)
        p = np.multiply(p, sdf)
        plot_iso_surface_eval(opt, coords, p, 'p [Pa]', 'p', 'exp')
        plot_iso_surface_eval(opt, coords, sdf, 'Phase [-]', 'vol_frac', 'exp')
        plot_iso_surface_eval(opt, coords, u, 'u [m/s]', 'u', 'exp')
        plot_iso_surface_eval(opt, coords, v, 'v [m/s]', 'v', 'exp')
        plot_iso_surface_eval(opt, coords, w, 'w [m/s]', 'w', 'exp')
        #gen_vtk_prediction(coords, u, 'u', save_path[:-4])

    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
    uv = xyz_tensor[:, :2, :]
    color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
    color = color * 0.5 + 0.5
    save_obj_mesh_with_color(save_path, verts, faces, color)


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


def calc_error(opt, net, cuda, dataset, num_tests, slice_dim='z', ds='train', plot_results=False, eval_droplet_only=False):
    # Read fluid properties and simulation domain
    with open(os.path.join(opt.dataroot, "flow_case.json"), "r") as f:
        flow_case = json.load(f)

    # non-dimensionalize the label data
    U_ref = flow_case["U_0"]  # impact velocity
    L_ref = flow_case["rp"]  # image reproduction scale -> domain size
    rho_ref = flow_case["rho_1"]  # density of liquid phase (water)

    if num_tests > len(dataset):
        num_tests = len(dataset)
        
        
    #num_tests = 1
        
    with torch.no_grad():
        error_alpha_arr, error_u_arr, error_v_arr, error_w_arr, error_pres_arr, error_conti_arr, error_phase_arr, error_nse_x_arr, error_nse_y_arr, error_nse_z_arr, IOU_arr, prec_arr, recall_arr = [], [], [], [], [], [], [], [], [], [], [], [], []
        Erel_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.name) + '_' + ds + '_uvp_L1_L2_rel.txt')
        Eabs_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.name) + '_' + ds + '_uvp_L1_L2_abs.txt')
        U_mean_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.name) + '_' + ds + '_U_mean.txt')

        # Initialize error collection for average calculation over entire dataset
        total_abs_error_u = total_square_error_u = total_abs_lab_u = total_square_lab_u = total_elements_u = 0.0
        total_abs_error_v = total_square_error_v = total_abs_lab_v = total_square_lab_v = total_elements_v = 0.0
        total_abs_error_w = total_square_error_w = total_abs_lab_w = total_square_lab_w = total_elements_w = 0.0
        total_abs_error_p = total_square_error_p = total_abs_lab_p = total_square_lab_p = total_elements_p = 0.0
        total_abs_error_alpha = total_square_error_alpha = total_abs_lab_alpha = total_square_lab_alpha = total_elements_alpha = 0.0
        for idx in range(num_tests):
            #data = dataset[idx * len(dataset) // num_tests]
            # skip to time steps after 20
            data = dataset[idx+8]
            
            #data = dataset[19] # get sample 0069

            # retrieve the data
            name = data['name']
            print('Processing:', name)

            image_tensor = data['img'].to(device=cuda)
            calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            #sample_tensor_uvwp = data['samples_uvwp'].to(device=cuda).unsqueeze(0)
            #sample_tensor_residual = data['samples_residual'].to(device=cuda).unsqueeze(0)
            sample_tensor_uvwp = None
            sample_tensor_residual = None

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)

            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)
            label_tensor_u = data['labels_u'].to(device=cuda).unsqueeze(0)
            label_tensor_v = data['labels_v'].to(device=cuda).unsqueeze(0)
            label_tensor_w = data['labels_w'].to(device=cuda).unsqueeze(0)
            label_tensor_p = data['labels_p'].to(device=cuda).unsqueeze(0)
            time_step_label = data['time_step'].to(device=cuda)

            res, res_PINN, loss_data_alpha, loss_data_u, loss_data_v, loss_data_w, loss_data_p, loss_conti, loss_phase_conv, loss_momentum_x, loss_momentum_y, loss_momentum_z = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor, uvwp_points=sample_tensor_uvwp, residual_points=sample_tensor_residual, labels_u=label_tensor_u,
                   labels_v=label_tensor_v, labels_w=label_tensor_w, labels_p=label_tensor_p, time_step=time_step_label, get_PINN_loss=False)
                   

            loss = loss_data_alpha + loss_data_u + loss_data_v + loss_data_w + loss_data_p + loss_conti + loss_phase_conv + loss_momentum_x + loss_momentum_z + loss_momentum_z

            IOU, prec, recall = compute_acc(res[:, :, :opt.n_data], label_tensor)

            if opt.num_views > 1:
                # pick middle frame
                calib_tensor = calib_tensor[opt.num_views // 2: opt.num_views // 2 + 1, :, :]
                #print('calib_tensor shape: ', calib_tensor.size())

            labels_u_proj, labels_w_proj = project_velocity_vector_field(label_tensor_u, label_tensor_w,
                                                                         calib_tensor)
            ground_plot = 0                                  
            y = sample_tensor[0, 1, :]
            label_tensor = label_tensor[0, 0, y > ground_plot]
            res = res[:, :, :opt.n_data]
            res = res[0, 0, y > ground_plot]
            
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
            if sample_tensor_uvwp is not None:
                res_PINN = res_PINN[:, :, opt.n_data:2*opt.n_data]
                res_PINN = res_PINN[0, :, y > ground_plot]
            else:
                res_PINN = res_PINN[0, :, y > ground_plot]
                
            
            #ground_offset = 1
            #labels_u_proj[sample_y < ground_offset] = 0
            #label_tensor_v[sample_y < ground_offset] = 0
            #labels_w_proj[sample_y < ground_offset] = 0
            #label_tensor_p[sample_y < ground_offset] = 0
            #res_PINN[:, sample_y < ground_offset] = 0
                

            # Create valid mask for in-bounds (non-NaN)
            valid_mask = ~torch.isnan(labels_u_proj) & \
                         ~torch.isnan(label_tensor_v) & \
                         ~torch.isnan(labels_w_proj) & \
                         ~torch.isnan(label_tensor_p)
                         
            # Apply the mask to predictions and labels
            labels_u_proj = labels_u_proj[valid_mask]
            label_tensor_v = label_tensor_v[valid_mask]
            labels_w_proj = labels_w_proj[valid_mask]
            label_tensor_p = label_tensor_p[valid_mask]
            res_PINN = res_PINN[:, valid_mask]
            
            # Optional: apply same mask to coordinates if needed
            sample_x = sample_x[valid_mask]
            sample_y = sample_y[valid_mask]
            sample_z = sample_z[valid_mask]
            
            print(f"Valid samples after NaN masking: {valid_mask.sum().item()} / {valid_mask.numel()}")
                                                                      

            # retrieve dimensional data for u,v,w,p
            if eval_droplet_only:
                #Consider zero calculation outside droplet 
                print('Evaluate only fields inside droplet')
                mask = (label_tensor[valid_mask] > 0.5)
                #print('shape before: ', res_PINN[1].shape)
                
                u_pred = res_PINN[1][mask] * U_ref
                v_pred = res_PINN[2][mask] * U_ref
                w_pred = res_PINN[3][mask] * U_ref
                p_pred = res_PINN[4][mask] * rho_ref * U_ref ** 2
                pred_dim = torch.stack((res_PINN[0][mask], u_pred, v_pred, w_pred, p_pred))
    
                u_lab = labels_u_proj[mask] * U_ref
                v_lab = label_tensor_v[mask] * U_ref
                w_lab = labels_w_proj[mask] * U_ref
                p_lab = label_tensor_p[mask] * rho_ref * U_ref ** 2
                #print('shape before: ', u_pred.shape)
            else:
                print('Evaluate fields in complete domain')
                u_pred = res_PINN[1] * U_ref
                v_pred = res_PINN[2] * U_ref
                w_pred = res_PINN[3] * U_ref
                p_pred = res_PINN[4] * rho_ref * U_ref ** 2
                pred_dim = torch.stack((res_PINN[0], u_pred, v_pred, w_pred, p_pred))
    
                u_lab = labels_u_proj * U_ref
                v_lab = label_tensor_v * U_ref
                w_lab = labels_w_proj * U_ref
                p_lab = label_tensor_p * rho_ref * U_ref ** 2

            print('u field mean: ', u_lab.mean().item(), 'max: ', u_lab.max().item(), 'min: ', u_lab.min().item())
            print('v field mean: ', v_lab.mean().item(), 'max: ', v_lab.max().item(), 'min: ', v_lab.min().item())
            print('w field mean: ', w_lab.mean().item(), 'max: ', w_lab.max().item(), 'min: ', w_lab.min().item())
            print('p field mean: ', p_lab.mean().item(), 'max: ', p_lab.max().item(), 'min: ', p_lab.min().item())

            if plot_results:
                sample_tensor = torch.stack((sample_x, sample_y, sample_z))
                # plot compound prediction (pressure contours, alpha contour line, velocity vector field)
                plot_compound_crop(opt, sample_tensor, pred_dim, label_tensor, u_lab, v_lab, w_lab, p_lab, slice_dim, 'compound',
                                     'pres', name, ds)
                
                #plot_compound(opt, sample_tensor, pred_dim, label_tensor, u_lab, v_lab, w_lab, p_lab, slice_dim, 'compound',
                #                     'pres', name, ds)

                # plot error in alpha field               
                plot_contour(opt, sample_tensor, res_PINN[0], label_tensor, slice_dim, 'alpha', 'alpha', name, ds)

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

            calc_vel_press_error = True
            if calc_vel_press_error:
                a_abs_err, a_square_err, a_abs_lab, a_square_lab, a_num = compute_errors(res[0], label_tensor, norm='mean')
                u_abs_err, u_square_err, u_abs_lab, u_square_lab, u_num = compute_errors(u_pred, u_lab, norm='mean')
                v_abs_err, v_square_err, v_abs_lab, v_square_lab, v_num = compute_errors(v_pred, v_lab, norm='mean')
                w_abs_err, w_square_err, w_abs_lab, w_square_lab, w_num = compute_errors(w_pred, w_lab, norm='mean')
                p_abs_err, p_square_err, p_abs_lab, p_square_lab, p_num = compute_errors(p_pred, p_lab, norm='mean')
                
                U_mean = torch.mean((u_lab**2 + v_lab**2 + w_lab**2)**0.5)

                str_err_1 = '{0}/{1}: {2} | U_mean: {3:06f}\n'.format(idx, num_tests, name, U_mean.item())
                print(str_err_1)
                with open(U_mean_log, 'a') as outfile:
                    outfile.write(str_err_1)
                    
                
                # Careful here! Calculate relative errors for each time step, later global errors need to be calculated over whole dataset, not as average of local errors
                l1_a, l1_rel_a, l2_a, l2_rel_a = compute_global_errors(a_abs_err, a_square_err, a_abs_lab, a_square_lab, a_num)
                l1_u, l1_rel_u, l2_u, l2_rel_u = compute_global_errors(u_abs_err, u_square_err, u_abs_lab, u_square_lab, u_num)
                l1_v, l1_rel_v, l2_v, l2_rel_v = compute_global_errors(v_abs_err, v_square_err, v_abs_lab, v_square_lab, v_num)
                l1_w, l1_rel_w, l2_w, l2_rel_w = compute_global_errors(w_abs_err, w_square_err, w_abs_lab, w_square_lab, w_num)
                l1_p, l1_rel_p, l2_p, l2_rel_p = compute_global_errors(p_abs_err, p_square_err, p_abs_lab, p_square_lab, p_num)
     
                    
                str_err_2 = '{0}/{1}: {2} (relative)| L1_u: {3:06f} | L2_u: {4:06f} | L1_v: {5:06f} | L2_v: {6:06f} | L1_w: {7:06f} | L2_w: {8:06f} | L1_p: {9:06f} | L2_p: {10:06f} | IOU: {11:06f}\n'.format(
                    idx, num_tests, name, l1_rel_u, l2_rel_u, l1_rel_v, l2_rel_v, l1_rel_w, l2_rel_w, l1_rel_p, l2_rel_p, IOU.item())
                print(str_err_2)
                with open(Erel_log, 'a') as outfile:
                    outfile.write(str_err_2)
                    
                str_err_3 = '{0}/{1}: {2} (absolute)| L1_u: {3:06f} | L2_u: {4:06f} | L1_v: {5:06f} | L2_v: {6:06f} | L1_w: {7:06f} | L2_w: {8:06f} | L1_p: {9:06f} | L2_p: {10:06f} | IOU: {11:06f}\n'.format(
                    idx, num_tests, name, l1_u, l2_u, l1_v, l2_v, l1_w, l2_w, l1_p, l2_p, IOU.item())
                print(str_err_3)
                with open(Eabs_log, 'a') as outfile:
                    outfile.write(str_err_3)

                # accumulate errors
                total_abs_error_alpha += a_abs_err
                total_square_error_alpha += a_square_err
                total_abs_lab_alpha += a_abs_lab
                total_square_lab_alpha += a_square_lab
                total_elements_alpha += a_num

                total_abs_error_u += u_abs_err
                total_square_error_u += u_square_err
                total_abs_lab_u += u_abs_lab
                total_square_lab_u += u_square_lab
                total_elements_u += u_num

                total_abs_error_v += v_abs_err
                total_square_error_v += v_square_err
                total_abs_lab_v += v_abs_lab
                total_square_lab_v += v_square_lab
                total_elements_v += v_num

                total_abs_error_w += w_abs_err
                total_square_error_w += w_square_err
                total_abs_lab_w += w_abs_lab
                total_square_lab_w += w_square_lab
                total_elements_w += w_num

                total_abs_error_p += p_abs_err
                total_square_error_p += p_square_err
                total_abs_lab_p += p_abs_lab
                total_square_lab_p += p_square_lab
                total_elements_p += p_num

            error_alpha_arr.append(l2_rel_a)
            error_u_arr.append(l2_rel_u)
            error_v_arr.append(l2_rel_v)
            error_w_arr.append(l2_rel_w)
            error_pres_arr.append(l2_rel_p)
            error_conti_arr.append(loss_conti.item())
            error_phase_arr.append(loss_phase_conv.item())
            error_nse_x_arr.append(loss_momentum_x.item())
            error_nse_y_arr.append(loss_momentum_y.item())
            error_nse_z_arr.append(loss_momentum_z.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    # Compute dataset-wide metrics
    
    if calc_vel_press_error:
        l1_alpha, rel_l1_alpha, l2_alpha, rel_l2_alpha = compute_global_errors(
            total_abs_error_alpha, total_square_error_alpha, total_abs_lab_alpha, total_square_lab_alpha,
            total_elements_alpha)
        l1_u, rel_l1_u, l2_u, rel_l2_u = compute_global_errors(
            total_abs_error_u, total_square_error_u, total_abs_lab_u, total_square_lab_u, total_elements_u)
        l1_v, rel_l1_v, l2_v, rel_l2_v = compute_global_errors(
            total_abs_error_v, total_square_error_v, total_abs_lab_v, total_square_lab_v, total_elements_v)
        l1_w, rel_l1_w, l2_w, rel_l2_w = compute_global_errors(
            total_abs_error_w, total_square_error_w, total_abs_lab_w, total_square_lab_w, total_elements_w)
        l1_p, rel_l1_p, l2_p, rel_l2_p = compute_global_errors(
            total_abs_error_p, total_square_error_p, total_abs_lab_p, total_square_lab_p, total_elements_p)
            
        E_global_log = os.path.join(opt.checkpoints_path, opt.name, str(opt.name) + '_' + ds + '_uvp_L1_L2_global.txt')
        
        str_err_4 = (f"\n--- Global Dataset Errors ---\n"
        f"Alpha: L1_mean={l1_alpha:.6f}, L2_mean={l2_alpha:.6f}, rel_L1={rel_l1_alpha:.6f}, rel_L2={rel_l2_alpha:.6f}\n"
        f"U:     L1_mean={l1_u:.6f}, L2_mean={l2_u:.6f}, rel_L1={rel_l1_u:.6f}, rel_L2={rel_l2_u:.6f}\n"
        f"V:     L1_mean={l1_v:.6f}, L2_mean={l2_v:.6f}, rel_L1={rel_l1_v:.6f}, rel_L2={rel_l2_v:.6f}\n"
        f"W:     L1_mean={l1_w:.6f}, L2_mean={l2_w:.6f}, rel_L1={rel_l1_w:.6f}, rel_L2={rel_l2_w:.6f}\n"
        f"P:     L1_mean={l1_p:.6f}, L2_mean={l2_p:.6f}, rel_L1={rel_l1_p:.6f}, rel_L2={rel_l2_p:.6f}\n"
        )
        
        print(str_err_4)
        
        with open(E_global_log, 'a') as outfile:
            outfile.write(str_err_4)

    return np.average(error_alpha_arr), np.average(error_u_arr), np.average(error_v_arr), np.average(error_w_arr), np.average(error_pres_arr), np.average(error_conti_arr), np.average(error_phase_arr), np.average(error_nse_x_arr), np.average(error_nse_y_arr), np.average(error_nse_z_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

