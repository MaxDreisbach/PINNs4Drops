from skimage import measure
import numpy as np
import torch
import time
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
from .plotting import *
import matplotlib
import matplotlib.pyplot as plt

PLOTTING = False


def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None, time_step=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points, time_step, idx):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor, time_step=time_step)
        #pred = net.get_preds()[0][idx]
        ''' Modification for PINN to allow evaluation of all fields: alpha + u,v,w,p'''
        pred = net.get_preds_dimensional()[0][idx]
        #print('idx=(a,u,v,p):', idx ,'pred shape: ', pred.size())
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    time_log = 'time.txt'
    net_start_time = time.time()
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=0)
        u = eval_grid_octree(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=1)
        v = eval_grid_octree(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=2)
        w = eval_grid_octree(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=3)
        p = eval_grid_octree(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=4)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=0)
        u = eval_grid(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=1)
        v = eval_grid(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=2)
        w = eval_grid(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=3)
        p = eval_grid(coords, eval_func, num_samples=num_samples, time_step=time_step, idx=4)

    net_end_time = time.time()
    print('network time: {0}\n'.format(net_end_time - net_start_time))

    # Plotting for debug
    if PLOTTING:
        plot_contour_grid(sdf, sdf, 'z', 64, 'alpha')
        plot_contour_grid(u, u, 'z', 64, 'vel')
        plot_contour_grid(v, v, 'z', 64, 'vel')
        plot_contour_grid(w, w, 'z', 64, 'vel')
        plot_contour_grid(p, p, 'z', 64, 'pres')

    # Finally we do marching cubes
    mc_start_time = time.time()
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T

        mc_end_time = time.time()
        print('marching cubes time: {0}\n'.format(mc_end_time - mc_start_time))
        with open(time_log, 'a') as outfile:
            outfile.write('{0},\n'.format(mc_end_time - mc_start_time))

        return verts, faces, coords, u, v, w, p, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
