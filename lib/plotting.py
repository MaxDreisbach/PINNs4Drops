import copy
import os
from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from scipy.interpolate import interpn
from scipy.interpolate import griddata
import pyvista

def plot_contour_w_alpha(opt, samples, preds, alpha, labels, labels_alpha, plane_dim, name, type, sample_name, dataset_type):
    font_size = 24.0

    sample = samples.detach().cpu().numpy()
    sample = sample.T
    label = labels.detach().cpu().numpy()
    label_alpha = labels_alpha.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()
    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]


    def interpolate_grid(pred, sample, X, Y, Z):
        pred_interpn = griddata(sample, pred, (X, Y, Z), method='linear')
        pred_linear = griddata(sample, pred, (X, Y, Z), method='nearest')
        pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]
        return pred_interpn

    pred_interpn = interpolate_grid(pred, sample, X, Y, Z)
    alpha_interpn = griddata(sample, alpha, (X, Y, Z), method='nearest')
    label_interpn = interpolate_grid(label, sample, X, Y, Z)
    label_alpha_interpn = griddata(sample, label_alpha, (X, Y, Z), method='nearest')

    # mask out prediction in solid domain
    ground = 28
    pred_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    alpha_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    label_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    label_alpha_interpn[:, :int(ground * opt.resolution / 256), :] = 0

    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
        alpha_plot = np.squeeze(alpha_interpn[0, :, :])
        alpha_gt_plot = np.squeeze(label_alpha_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
        alpha_plot = np.squeeze(alpha_interpn[:, 0, :].T)
        alpha_gt_plot = np.squeeze(label_alpha_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)
        alpha_plot = np.squeeze(alpha_interpn[:, :, 0].T)
        alpha_gt_plot = np.squeeze(label_alpha_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)

    x, y = np.meshgrid(np.arange(opt.resolution)/opt.resolution, np.arange(opt.resolution)/opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 100)
        cbar_ticks = [0.0, 0.5, 1.0]
        cbar_label = '$\alpha$ [-]'
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.0, 1.0, 100)
        cbar_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        levels_err = np.linspace(0.0, 0.25, 100)
        cbar_ticks_err = [0.0, 0.0625, 0.125, 0.1875, 0.25]
        cbar_label = '$%s$ [m/s]' %name
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-150.0, 600, 100)
        cbar_ticks = [-150.0, 0.0, 150, 300, 450, 600]
        levels_err = np.linspace(0.0, 200, 100)
        cbar_ticks_err = [0, 50, 100, 150, 200]
        cbar_label = '$p$ [Pa]'
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.85))
    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap, linestyles='none')
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap, linestyles='none')
    p3 = axs[2].contourf(x, y, err_plot, levels=levels_err, cmap='viridis', linestyles='none')


    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs[0].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a2 = axs[1].contour(x, y, alpha_gt_plot, levels=levels_alpha, colors='k')
    a3 = axs[2].contour(x, y, alpha_gt_plot, levels=levels_alpha, colors='k')

    axs[0].set_ylabel('$y$', fontsize=font_size)
    axs[0].set_xlabel('$x$', fontsize=font_size)
    axs[1].set_xlabel('$x$', fontsize=font_size)
    axs[2].set_xlabel('$x$', fontsize=font_size)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)

    axs[0].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[1].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[2].tick_params(axis ='both', which ='major', labelsize = font_size)

    axs[0].set_title('$%s_{\mathrm{R}}$' %name, fontsize=font_size, y=1, pad=20)
    axs[1].set_title('$%s_{\mathrm{GT}}$' %name, fontsize=font_size, y=1, pad=20)
    axs[2].set_title('$%s_{\mathrm{err}}$' %name, fontsize=font_size, y=1, pad=20)

    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs[0], location='bottom')
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom')
    cbar3 = fig.colorbar(p3, ax=axs[2], location='bottom')
    cbar1.ax.tick_params(labelsize=font_size, which='major', width=1.5, length=6)
    cbar2.ax.tick_params(labelsize=font_size, which='major')
    cbar3.ax.tick_params(labelsize=font_size, which='major')
    cbar1.set_ticks(cbar_ticks)
    cbar2.set_ticks(cbar_ticks)
    cbar3.set_ticks(cbar_ticks_err)
    cbar1.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
    cbar2.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
    cbar3.set_label(cbar_label, fontsize=font_size, labelpad=0.1)

    filename = 'results/' + opt.name + '/pred_fields/'  + dataset_type + '_' + sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)


def plot_contour_w_alpha_res_gt(opt, samples, preds, alpha, labels_alpha, labels, plane_dim, name, type, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    label_p = labels.detach().cpu().numpy()
    label_alpha = labels_alpha.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    alpha = alpha.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]


    def interpolate_grid(pred, sample, X, Y, Z):
        pred_interpn = griddata(sample, pred, (X, Y, Z), method='linear')
        pred_nearest = griddata(sample, pred, (X, Y, Z), method='nearest')
        pred_interpn[np.isnan(pred_interpn)] = pred_nearest[np.isnan(pred_interpn)]
        return pred_interpn

    pred_interpn = interpolate_grid(pred, sample, X, Y, Z)
    alpha_interpn = interpolate_grid(alpha, sample, X, Y, Z)
    label_p_interpn = interpolate_grid(label_p, sample, X, Y, Z)
    label_alpha_interpn = interpolate_grid(label_alpha, sample, X, Y, Z)

    # mask out prediction in solid domain
    ground = 28
    pred_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    alpha_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    label_p_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    label_alpha_interpn[:, :int(ground * opt.resolution / 256), :] = 0


    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_p_interpn[0, :, :])
        alpha_plot = np.squeeze(alpha_interpn[0, :, :])
        alpha_gt_plot = np.squeeze(label_alpha_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_p_interpn[:, 0, :].T)
        alpha_plot = np.squeeze(alpha_interpn[:, 0, :].T)
        alpha_gt_plot = np.squeeze(label_alpha_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_p_interpn[:, :, 0].T)
        alpha_plot = np.squeeze(alpha_interpn[:, :, 0].T)
        alpha_gt_plot = np.squeeze(label_alpha_interpn[:, :, 0].T)

    x, y = np.meshgrid(np.arange(opt.resolution)/opt.resolution, np.arange(opt.resolution)/opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 100)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 100)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.6, 3.0, 10)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    #plt.rcParams['figure.constrained_layout.use'] = True

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 6.45))
    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)


    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs[0].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a2 = axs[1].contour(x, y, alpha_gt_plot, levels=levels_alpha, colors='k')

    axs[0].set_ylabel('$y$', fontsize=16)
    axs[0].set_xlabel('$x$', fontsize=16)
    axs[1].set_xlabel('$x$', fontsize=16)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)

    axs[0].tick_params(axis ='both', which ='major', labelsize = 20, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 20, pad = 10)

    axs[0].set_title('$%s_{\mathrm{R}}$' %name, fontsize=20, y=1, pad=20)
    axs[1].set_title('$%s_{\mathrm{GT}}$' %name, fontsize=20, y=1, pad=20)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs[0], location='bottom')
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom')
    cbar1.ax.tick_params(labelsize=20, which='major', width=1.5, length=6)
    cbar2.ax.tick_params(labelsize=20, which='major')
    cbar1.ax.locator_params(nbins=5)
    cbar2.ax.locator_params(nbins=5)
    plt.tight_layout(w_pad=4.5)

    filename = 'results/'+ opt.name + '/' + dataset_type + '_' + sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)
    

def plot_compound(opt, samples, res_PINN, labels_alpha, labels_u, labels_v, labels_w, labels_p, plane_dim, name, type, sample_name, dataset_type):
    font_size = 24.0

    sample = samples.detach().cpu().numpy()
    sample = sample.T
    label_alpha = labels_alpha.detach().cpu().numpy()
    label_u = labels_u.detach().cpu().numpy()
    label_v = labels_v.detach().cpu().numpy()
    label_w = labels_w.detach().cpu().numpy()
    label_p = labels_p.detach().cpu().numpy()
    res_PINN = res_PINN.detach().cpu().numpy()
    pred_alpha = res_PINN[0, :]
    pred_u = res_PINN[1, :]
    pred_v = res_PINN[2, :]
    pred_w = res_PINN[3, :]
    pred_p = res_PINN[4, :]


    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]

    def interpolate_grid(pred, sample, X, Y, Z):
        pred_interpn = griddata(sample, pred, (X, Y, Z), method='linear')
        pred_nearest = griddata(sample, pred, (X, Y, Z), method='nearest')
        pred_interpn[np.isnan(pred_interpn)] = pred_nearest[np.isnan(pred_interpn)]
        return pred_interpn

    alpha_interpn = griddata(sample, pred_alpha, (X, Y, Z), method='nearest')
    u_interpn = interpolate_grid(pred_u, sample, X, Y, Z)
    v_interpn = interpolate_grid(pred_v, sample, X, Y, Z)
    w_interpn = interpolate_grid(pred_w, sample, X, Y, Z)
    p_interpn = interpolate_grid(pred_p, sample, X, Y, Z)
    label_alpha_interpn = griddata(sample, label_alpha, (X, Y, Z), method='nearest')
    label_u_interpn = interpolate_grid(label_u, sample, X, Y, Z)
    label_v_interpn = interpolate_grid(label_v, sample, X, Y, Z)
    label_w_interpn = interpolate_grid(label_w, sample, X, Y, Z)
    label_p_interpn = interpolate_grid(label_p, sample, X, Y, Z)


    # mask out prediction in solid domain
    ground = 28
    alpha_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    u_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    v_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    w_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    p_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    label_alpha_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    label_u_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    label_v_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    label_w_interpn[:, :int(28 * opt.resolution / 256), :] = 0
    label_p_interpn[:, :int(28 * opt.resolution / 256), :] = 0

    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:1j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]

    if plane_dim == 'x':
        alpha_plot = np.squeeze(alpha_interpn[0, :, :])
        u_plot = np.squeeze(u_interpn[0, :, :])
        v_plot = np.squeeze(v_interpn[0, :, :])
        w_plot = np.squeeze(w_interpn[0, :, :])
        p_plot = np.squeeze(p_interpn[0, :, :])
        label_alpha_plot = np.squeeze(label_alpha_interpn[0, :, :])
        label_u_plot = np.squeeze(label_u_interpn[0, :, :])
        label_v_plot = np.squeeze(label_v_interpn[0, :, :])
        label_w_plot = np.squeeze(label_w_interpn[0, :, :])
        label_p_plot = np.squeeze(label_p_interpn[0, :, :])
    if plane_dim == 'y':
        alpha_plot = np.squeeze(alpha_interpn[:, 0, :].T)
        u_plot = np.squeeze(u_interpn[:, 0, :].T)
        v_plot = np.squeeze(v_interpn[:, 0, :].T)
        w_plot = np.squeeze(w_interpn[:, 0, :].T)
        p_plot = np.squeeze(p_interpn[:, 0, :].T)
        label_alpha_plot = np.squeeze(label_alpha_interpn[:, 0, :].T)
        label_u_plot = np.squeeze(label_u_interpn[:, 0, :].T)
        label_v_plot = np.squeeze(label_v_interpn[:, 0, :].T)
        label_w_plot = np.squeeze(label_w_interpn[:, 0, :].T)
        label_p_plot = np.squeeze(label_p_interpn[:, 0, :].T)
    if plane_dim == 'z':
        alpha_plot = np.squeeze(alpha_interpn[:, :, 0].T)
        u_plot = np.squeeze(u_interpn[:, :, 0].T)
        v_plot = np.squeeze(v_interpn[:, :, 0].T)
        w_plot = np.squeeze(w_interpn[:, :, 0].T)
        p_plot = np.squeeze(p_interpn[:, :, 0].T)
        label_alpha_plot = np.squeeze(label_alpha_interpn[:, :, 0].T)
        label_u_plot = np.squeeze(label_u_interpn[:, :, 0].T)
        label_v_plot = np.squeeze(label_v_interpn[:, :, 0].T)
        label_w_plot = np.squeeze(label_w_interpn[:, :, 0].T)
        label_p_plot = np.squeeze(label_p_interpn[:, :, 0].T)


    x, y = np.meshgrid(np.arange(opt.resolution)/opt.resolution, np.arange(opt.resolution)/opt.resolution)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    levels = np.linspace(-150, 600, 100)
    cbar_ticks = [-150.0, 0.0, 150, 300, 450, 600]

    colormap = 'viridis'

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 6.85))
    p1 = axs[0].contourf(x, y, p_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, label_p_plot, levels=levels, cmap=colormap)

    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs[0].contour(x, y, alpha_plot, levels=levels_alpha, colors='k')
    a2 = axs[1].contour(x, y, label_alpha_plot, levels=levels_alpha, colors='k')

    skip = (slice(None, None, 10), slice(None, None, 10))
    if plane_dim == 'x':
        q1 = axs[0].quiver(x[skip], y[skip], w_plot[skip], v_plot[skip], scale=7.5, scale_units='inches', color='black')
        q2 = axs[1].quiver(x[skip], y[skip], label_w_plot[skip], label_v_plot[skip], scale=7.5, scale_units='inches', color='black')
    if plane_dim == 'y':
        q1 = axs[0].quiver(x[skip], y[skip], u_plot[skip], w_plot[skip], scale=7.5, scale_units='inches', color='black')
        q2 = axs[1].quiver(x[skip], y[skip], label_u_plot[skip], label_w_plot[skip], scale=7.5, scale_units='inches', color='black')
    if plane_dim == 'z':
        q1 = axs[0].quiver(x[skip], y[skip], u_plot[skip], v_plot[skip], scale=7.5, scale_units='inches', color='black')
        q2 = axs[1].quiver(x[skip], y[skip], label_u_plot[skip], label_v_plot[skip], scale=7.5, scale_units='inches', color='black')

    axs[0].set_ylabel('$y$', fontsize=font_size)
    axs[0].set_xlabel('$x$', fontsize=font_size)
    axs[1].set_xlabel('$x$', fontsize=font_size)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)

    axs[0].tick_params(axis ='both', which ='major', labelsize = font_size, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = font_size, pad = 10)
    axs[0].set_title('$\mathrm{prediction}$', fontsize=font_size, y=1, pad=20)
    axs[1].set_title('$\mathrm{ground} \: \mathrm{truth}$', fontsize=font_size, y=1, pad=20)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs[0], location='bottom')
    cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom')
    cbar1.ax.tick_params(labelsize=font_size, which='major', width=1.5, length=6)
    cbar2.ax.tick_params(labelsize=font_size, which='major')
    cbar1.set_ticks(cbar_ticks)
    cbar2.set_ticks(cbar_ticks)
    cbar1.set_label('$p$ [Pa]', fontsize=font_size, labelpad=0.1)
    cbar2.set_label('$p$ [Pa]', fontsize=font_size, labelpad=0.1)

    filename = 'results/' + opt.name + '/pred_fields/'  + dataset_type + '_' + sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)
    

def plot_contour_grid(pred, label, plane_dim, res, type):
    # interpolate point cloud to 2D-plane
    ind = res // 2

    if plane_dim == 'x':
        var_plot = pred[ind, :, :].T
        gt_plot = label[ind, :, :].T
    if plane_dim == 'y':
        var_plot = pred[:, ind, :].T
        gt_plot = label[:, ind, :].T
    if plane_dim == 'z':
        var_plot = pred[:, :, ind].T
        gt_plot = label[:, :, ind].T

    if type == 'alpha':
        v_min, v_max = 0, 1
        colormap = 'RdBu_r'
    if type == 'vel':
        v_min, v_max = -2, 2
        colormap = 'RdBu_r'
    if type == 'pres':
        v_min, v_max = -250, 1000
        colormap = 'viridis'

    err_plot = np.absolute(var_plot - gt_plot)

    fig, axs = plt.subplots(1, 3, figsize=(17,6))
    p1 = axs[0].contourf(var_plot, cmap=colormap)
    p2 = axs[1].contourf(gt_plot, cmap=colormap)
    p3 = axs[2].contourf(err_plot, cmap=colormap)
    axs[0].contourf(var_plot, cmap=colormap)
    axs[1].contourf(gt_plot, cmap=colormap)
    axs[2].contourf(err_plot, cmap=colormap)

    axs[0].set_ylabel('y', fontsize=16)
    axs[0].set_xlabel('x', fontsize=16)
    axs[1].set_xlabel('x', fontsize=16)
    axs[2].set_xlabel('x', fontsize=16)
    axs[0].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[2].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[0].set_title(r'$U_{\mathrm{R}}$', fontsize=16, y=1, pad=20 )
    axs[1].set_title(r'$U_{\mathrm{GT}}$', fontsize=16, y=1, pad=20)
    axs[2].set_title(r'$U_{\mathrm{err}}$', fontsize=16, y=1, pad=20)

    x = np.arange(0, 512, 100)
    y = np.arange(0, 512, 100)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)
    plt.tight_layout(w_pad=4.5)
    plt.axis('equal')

    fig.colorbar(p1, ax=axs[0], location='bottom')
    fig.colorbar(p2, ax=axs[1], location='bottom')
    fig.colorbar(p3, ax=axs[2], location='bottom')
    plt.show()
    plt.savefig('modelA.pdf')


def plot_velocity_field(samples, preds, labels_u, labels_v, labels_w):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    sample_tensor = samples.detach().cpu().numpy()
    labels_u = labels_u.detach().cpu().numpy()
    labels_v = labels_v.detach().cpu().numpy()
    labels_w = labels_w.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()
    x = np.array(sample_tensor[0, 0, :])
    y = np.array(sample_tensor[0, 1, :])
    z = np.array(sample_tensor[0, 2, :])
    u = labels_u
    v = labels_v
    w = labels_w
    u_pred = pred[0, 1, :]
    v_pred = pred[0, 2, :]
    v_pred = pred[0, 3, :]

    # colorbar
    c = np.sqrt(np.abs(v) ** 2 + np.abs(u) ** 2 + np.abs(w) ** 2)
    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.jet(c)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x[:1000], y[:1000], z[:1000], u[:1000], v[:1000], w[:1000], arrow_length_ratio=0.5, length=10.0)
    #qq = ax.quiver(x[:1000], y[:1000], z[:1000], u[:1000], v[:1000], w[:1000], colors=c, arrow_length_ratio=0.3, length=5.0)
    #plt.colorbar(qq, cmap=plt.cm.jet)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    plt.show()
    plt.close(fig)


def plot_iso_surface(opt, samples, preds, name, sample_name, dataset_type):
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    OFF_SCREEN = False
    if OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    sample = samples.detach().cpu().numpy()
    sample = sample.T
    pred = preds.detach().cpu().numpy()
    

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -128:128:grid_res]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = pred_interpn.ravel(order='F')

    vmin = pred.min()
    vmax = pred.max()
    labels = dict(zlabel='Z', xlabel='X', ylabel='Y')
    contours = mesh.contour(np.linspace(vmin, vmax, 10))

    camera = pyvista.Camera()
    camera.position = (700.0, 100.0, 700.0)
    camera.focal_point = (5.0, 20.0, 5.0)

    if OFF_SCREEN:
        p = pyvista.Plotter(off_screen=True)
    else:
        p = pyvista.Plotter()

    p.add_mesh(mesh.outline(), color="k")
    p.add_mesh(contours, opacity=0.25, clim=[vmin, vmax])
    p.show_grid(**labels)
    p.add_axes(**labels)

    p.camera = camera

    if OFF_SCREEN:
        filename = 'results/' + dataset_type + '_' + sample_name + '_' + name + '_pred_3d.svg'
        p.save_graphic(filename)
        # p.screenshot(filename, transparent_background=True)
        p.close()
    else:
        p.show()
        

def plot_iso_surface_eval(opt, samples, preds, name, sample_name, dataset_type):
    # If environment variable PYVISTA_OFF_SCREEN is set to true save a png
    # otherwise create interactive plot
    
    print(name, preds.min(), preds.max())
    
    OFF_SCREEN = False
    if OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)

    
    if sample_name == 'vol_frac':
        vmin = 0
        vmax = 1
        num_contours = 5
        colormap = 'jet'
    elif sample_name == 'u':
        vmin = -0.5
        vmax = 0.5
        num_contours = 16
        colormap = 'seismic'
        #preds = np.ma.masked_where((preds >= -0.05) | (preds < 0.05), preds)
    elif sample_name == 'v':
        vmin = -0.7
        vmax = 0.7
        num_contours = 30
        colormap = 'seismic'
        #preds = np.ma.masked_where((preds >= -0.05) | (preds < 0.05), preds)
    elif sample_name == 'w':
        vmin = -0.4
        vmax = 0.4
        num_contours = 16
        colormap = 'seismic'
        #preds = np.ma.masked_where((preds >= -0.05) | (preds < 0.05), preds)
    if sample_name == 'p':
        vmin = -50
        vmax = 400
        num_contours = 14 #12
        colormap = 'jet'
        
    X = samples[0, :, :, :]
    Y = samples[1, :, :, :]
    Z = samples[2, :, :, :]
    
    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data[name] = preds.ravel(order='F')

    bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    mesh = mesh.clip_box(bounds)

    labels = dict(zlabel='Z', xlabel='X', ylabel='Y')
    contours = mesh.contour(np.linspace(vmin, vmax, num_contours))

    pyvista.set_plot_theme('document')
    camera = pyvista.Camera()
    camera.position = (-350.0/128, -100.0/128, 700.0/128)
    camera.focal_point = (5.0/128, 20.0/128, 5.0/128)
    camera.view_up = ([0, 0, 1])
    camera.roll = 180

    if OFF_SCREEN:
        p = pyvista.Plotter(off_screen=True)
    else:
        p = pyvista.Plotter()

    # clip half of the domain
    if sample_name == 'u':
        contours = contours.clip(normal="z")
    else:
        contours = contours.clip(normal="x")

    #p.add_mesh(mesh.outline(), color="k", show_edges=False)
    p.add_mesh(contours, opacity=1.0, clim=[vmin, vmax], cmap=colormap)
    #p.show_grid(**labels)
    p.add_axes(**labels)
    #p.camera = camera

    # Configure the font for LaTeX
    pyvista.global_theme.font.family = 'times'
    pyvista.global_theme.font.size = 24.0
    pyvista.global_theme.font.title_size = 24.0  # Larger title
    pyvista.global_theme.font.label_size = 24.0  # Larger label font


    if OFF_SCREEN:
        filename = 'results/' + dataset_type + '_' + sample_name + '_'  + '_pred_3d.svg'
        p.save_graphic(filename)
        # p.screenshot(filename, transparent_background=True)
        p.close()
    else:
        filename = 'results/' + dataset_type + '_' + sample_name + '_'  + '_pred_3d.svg'
        p.save_graphic(filename)
        p.show()


def gen_vtk_prediction(coords, preds, name, sample_name):
    X = coords[0, :, :, :]
    Y = coords[1, :, :, :]
    Z = coords[2, :, :, :]

    mesh = pyvista.StructuredGrid(X, Y, Z)
    mesh.point_data['values'] = preds.ravel(order='F')

    meshname = sample_name + '_' + name + '_pred_3d.vtk'
    mesh.save(meshname)

def plot_contour_eval(coords, opt, preds, alpha, plane_dim, name, type, sample_name):
    # interpolate point cloud to 2D-plane
    ind = opt.resolution // 2
    
    # mask out prediction in solid domain
    ground = 17 # for FDM
    ground_index = int(ground * opt.resolution / 256)
    preds[:, :int(ground_index * opt.resolution / 256), :] = 0

    if plane_dim == 'x':
        var_plot = preds[ind, :, :]
        alpha_plot = alpha[ind, :, :]
    elif plane_dim == 'y':
        var_plot = preds[:, ind, :].T
        alpha_plot = alpha[:, ind, :].T
    elif plane_dim == 'z':
        var_plot = preds[:, :, ind].T
        alpha_plot = alpha[:, :, ind].T
    else:
        raise ValueError("plane_dim must be 'x', 'y', or 'z'")

    if type == 'alpha':
        vmin, vmax = 0.0, 1.0
        levels = np.linspace(0, 1.0, 100)
        cbar_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
        colormap = 'RdBu_r'
    elif type == 'vel':
        vmin, vmax = -0.7, 0.7
        levels = np.linspace(-0.7, 0.7, 100)
        cbar_ticks = [-0.5, -0.25, 0, 0.25, 0.5]
        colormap = 'RdBu_r'
    elif type == 'pres':
        vmin, vmax = -150, 600
        levels = np.linspace(-150, 600, 100)
        cbar_ticks = [-150.0, 0.0, 150, 300, 450, 600]
        colormap = 'viridis'
    else:
        raise ValueError("Invalid type. Choose from 'alpha', 'vel', or 'pres'.")

    # Clip the data to colormap range
    var_plot = np.clip(var_plot, vmin, vmax)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    #plt.rcParams['figure.constrained_layout.use'] = True

    x, y = np.meshgrid(np.arange(opt.resolution) / opt.resolution, np.arange(opt.resolution) / opt.resolution)

    fig, axs = plt.subplots(figsize=(4.75, 6.85))
    #p1 = axs.contourf(
    #    x, y, var_plot,
    #    levels=levels,
    #    cmap=colormap,
    #    antialiased=False,
    #    linewidths=0)
    
    p1 = axs.imshow(
        var_plot,
        origin='lower',
        extent=[0, 1, 0, 1],
        vmin=vmin,
        vmax=vmax,
        cmap=colormap,
        interpolation='bilinear'
    )

    levels_alpha = np.linspace(0.5, 1.0, 2)
    a1 = axs.contour(x, y, alpha_plot, levels=levels_alpha, colors='k')

    axs.set_ylabel('$y$', fontsize=16)
    axs.set_xlabel('$x$', fontsize=16)

    x = np.arange(0.0, 1.0 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs.set_xticks(x)
    axs.set_yticks(y)
    axs.tick_params(axis='both', which='major', labelsize=20)
    axs.set_title('$%s_{\mathrm{R}}$' % name, fontsize=20, pad=20)
    
    #plt.tight_layout(w_pad=0.5)
    axs.set_aspect('equal', adjustable="datalim")
    cbar1 = fig.colorbar(p1, ax=axs, location='bottom')
    cbar1.ax.tick_params(labelsize=20, which='major', width=1.5, length=6)
    cbar1.set_ticks(cbar_ticks)
    #cbar1.ax.locator_params(nbins=5)

    filename = sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    #plt.savefig(filename, dpi=300)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    # plt.show()
    plt.close(fig)


def plot_data_sample(x, y, z, labels, vmin, vmax):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # x = samples[0, 0, :].detach().cpu().numpy()
    X = x.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Z = z.detach().cpu().numpy()
    var_plot = labels[0, :].detach().cpu().numpy()

    mappable = ax.scatter(X, Y, Z, s=5, c=var_plot, vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar(mappable)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def plot_contour(opt, samples, preds, labels, plane_dim, name, type, sample_name, dataset_type):
    font_size = 24.0
    sample = samples.detach().cpu().numpy()
    sample = sample.T
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    if plane_dim == 'x':
        X, Y, Z = np.mgrid[-0.1:0.1:3j, -28:228:grid_res, -128:128:grid_res]
    if plane_dim == 'y':
        X, Y, Z = np.mgrid[-128:128:grid_res, 7.9:8.1:1j, -128:128:grid_res]
    if plane_dim == 'z':
        X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.1:0.1:1j]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_nearest = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_nearest[np.isnan(pred_interpn)]
    pred_interpn = griddata(sample, pred, (X,Y,Z), method='nearest')

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_nearest = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_nearest[np.isnan(label_interpn)]
    label_interpn = griddata(sample, label, (X, Y, Z), method='nearest')

    # mask out prediction in solid domain
    ground = 28
    pred_interpn[:, :int(ground * opt.resolution / 256), :] = 0
    label_interpn[:, :int(ground * opt.resolution / 256), :] = 0


    if plane_dim == 'x':
        var_plot = np.squeeze(pred_interpn[0, :, :])
        gt_plot = np.squeeze(label_interpn[0, :, :])
    if plane_dim == 'y':
        var_plot = np.squeeze(pred_interpn[:, 0, :].T)
        gt_plot = np.squeeze(label_interpn[:, 0, :].T)
    if plane_dim == 'z':
        var_plot = np.squeeze(pred_interpn[:, :, 0].T)
        gt_plot = np.squeeze(label_interpn[:, :, 0].T)

    err_plot = np.absolute(gt_plot - var_plot)
    x, y = np.meshgrid(np.arange(opt.resolution) / opt.resolution, np.arange(opt.resolution) / opt.resolution)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 9)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 10)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-0.6, 3.0, 10)
        colormap = 'viridis'

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    if plane_dim == 'y':
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.85))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.85))

    p1 = axs[0].contourf(x, y, var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(x, y, gt_plot, levels=levels, cmap=colormap)
    p3 = axs[2].contourf(x, y, err_plot, levels=levels, cmap=colormap)
    axs[0].set_xbound(lower=0.0, upper=1.0)
    axs[1].set_xbound(lower=0.0, upper=1.0)
    axs[2].set_xbound(lower=0.0, upper=1.0)

    if plane_dim == 'x':
        axs[0].set_ylabel('$y$', fontsize=font_size)
        axs[0].set_xlabel('$z$', fontsize=font_size)
        axs[1].set_xlabel('$z$', fontsize=font_size)
        axs[2].set_xlabel('$z$', fontsize=font_size)
    if plane_dim == 'y':
        axs[0].set_ylabel('$z$', fontsize=font_size)
        axs[0].set_xlabel('$x$', fontsize=font_size)
        axs[1].set_xlabel('$x$', fontsize=font_size)
        axs[2].set_xlabel('$x$', fontsize=font_size)
    if plane_dim == 'z':
        axs[0].set_ylabel('$y$', fontsize=font_size)
        axs[0].set_xlabel('$x$', fontsize=font_size)
        axs[1].set_xlabel('$x$', fontsize=font_size)
        axs[2].set_xlabel('$x$', fontsize=font_size)

    x = np.arange(0.2, 0.8 + 0.001, 0.2)
    y = np.arange(0.0, 1.0 + 0.001, 0.2)
    axs[0].set_xticks(x)
    axs[0].set_yticks(y)
    axs[1].set_xticks(x)
    axs[1].set_yticks(y)
    axs[2].set_xticks(x)
    axs[2].set_yticks(y)


    axs[0].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[1].tick_params(axis ='both', which ='major', labelsize = font_size)
    axs[2].tick_params(axis ='both', which ='major', labelsize = font_size)

    axs[0].set_title(r'$\alpha_{\mathrm{R}}$', fontsize=40, y=1, pad=20)
    axs[1].set_title(r'$\alpha_{\mathrm{GT}}$', fontsize=40, y=1, pad=20)
    axs[2].set_title(r'$\alpha_{\mathrm{err}}$', fontsize=40, y=1, pad=20)

    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    
    cbar_label = '$\phi$ [-]'
    cbar_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    if plane_dim == 'y':
        cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=font_size, which='major')
        cbar2.set_ticks(cbar_ticks)
    else:
        cbar1 = fig.colorbar(p2, ax=axs[0], location='bottom', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=font_size, which='major')
        cbar1.set_ticks(cbar_ticks)
        
        cbar2 = fig.colorbar(p2, ax=axs[1], location='bottom', fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=font_size, which='major')
        cbar2.set_ticks(cbar_ticks)
        
        cbar3 = fig.colorbar(p2, ax=axs[2], location='bottom', fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=font_size, which='major')
        cbar3.set_ticks(cbar_ticks)
        cbar1.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
        cbar2.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
        cbar3.set_label(cbar_label, fontsize=font_size, labelpad=0.1)
        

    filename = 'results/' + opt.name + '/pred_fields/'  + dataset_type + '_' + sample_name + '_' + name + '_' + plane_dim + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()
    plt.close(fig)


def plot_im_feat(im_feat):
    print(im_feat.shape)
    feature_map = im_feat.detach().cpu().numpy()


    # Reshape the tensor to (4, 4, 128, 128)
    tensor_reshaped = feature_map[0, :, :, :].reshape(16, 16, 128, 128)

    # Concatenate along the rows (vertical direction) first
    vert_concatenate = np.concatenate(np.split(tensor_reshaped, 16, axis=0), axis=2)

    # Concatenate along the columns (horizontal direction)
    image_2d = np.concatenate(np.split(vert_concatenate, 16, axis=1), axis=3)

    image_2d = image_2d[0, 0, :, :]

    # Display the result
    plt.imshow(image_2d)
    plt.title('Combined 2D Image')
    plt.show()


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,H)
    """
    #grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    grayscale_im = np.max(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    #heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=2))
    #no_trans_heatmap = no_trans_heatmap.filter(ImageFilter.GaussianBlur(radius=2))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def plot_saliency_map(grads, train_data, opt):
    img_int = 1.25
    dir_path = './results/' + opt.name + '/saliency_map/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    yid = str(train_data['yid'] * 10)
    name = str(train_data['name'])
    print('timestep: ', name, 'rotation angle: ', yid)


    # Normalize gradients
    grads = grads - grads.min()
    grads /= grads.max()
    grayscale_grads = convert_to_grayscale(grads)

    raw_img = np.uint8((np.transpose(train_data['raw_img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(np.uint8(raw_img)).convert('RGB')

    img = np.uint8((np.transpose(train_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    cam = (grayscale_grads - np.min(grayscale_grads)) / (
                np.max(grayscale_grads) - np.min(grayscale_grads))  # Normalize between 0-1
    cam = np.uint8(cam[0, :, :] * 255)  # Scale between 0-255 to visualize

    # Grayscale activation map
    # cmap = 'seismic'
    # cmap = 'bwr'
    cmap = 'coolwarm'
    heatmap, heatmap_on_image = apply_colormap_on_image(img, cam, cmap)

    heatmap_on_image = np.array(heatmap_on_image)
    heatmap = np.array(heatmap)
    img = np.array(img)
    raw_img = np.array(raw_img)
    grads = np.transpose(grads, (1, 2, 0))
    grayscale_grads = np.transpose(grayscale_grads, (1, 2, 0))
    
    img =  np.uint8(np.clip(img * img_int, a_min=0, a_max=255))
    raw_img =  np.uint8(np.clip(raw_img * img_int, a_min=0, a_max=255))

    # Plotting

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.5))
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(10, 4.25))
    ax1, ax2, ax3 = axes.flatten()
    # Note: grads are in BGR format
    p1 = ax1.imshow(heatmap / 255,
                    vmin=0, vmax=1, cmap=cmap)
    p2 = ax2.imshow(raw_img)
    px = ax3.imshow(heatmap / 255,
                    vmin=0, vmax=1)

    # ax1.set_title(r'saliency map', fontsize=16, y=1, pad=20)
    # ax2.set_title(r'input image', fontsize=16, y=1, pad=20)
    # ax3.set_title(r'image and map', fontsize=16, y=1, pad=20)

    cbar1 = fig.colorbar(p1, ax=ax1, location='bottom')
    cbar2 = fig.colorbar(p2, ax=ax2, location='bottom')
    cbar3 = fig.colorbar(px, ax=ax3, location='bottom')
    cbar1.ax.tick_params(labelsize=16, which='major')
    cbar2.ax.tick_params(labelsize=16, which='major')
    cbar3.ax.tick_params(labelsize=16, which='major')
    px.set_cmap(cmap)
    cbar2.remove()

    p3 = ax3.imshow(heatmap_on_image)
    cbar_ticks = [0.0, 0.5, 1.0]
    cbar1.set_ticks(cbar_ticks)
    cbar3.set_ticks(cbar_ticks)
    # cbar1.ax.locator_params(nbins=5)
    # cbar3.ax.locator_params(nbins=5)
    fig.execute_constrained_layout()

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    filename = dir_path + name + '_' + yid + '_smap.pdf'
    plt.savefig(filename, format='pdf', dpi=1200)

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    # ax1, ax2, ax3 = axes.flatten()
    # Note: grads are in BGR format
    # ax1.imshow(grads[:, :, 2])
    # ax2.imshow(grads[:, :, 1])
    # ax3.imshow(grads[:, :, 0])
    # fig.suptitle('Gradients of prediction wrt input image, red (left), green (middle), blue channel (right)')
    # plt.show()

    # filename = dir_path + name + '_' + yid + '_grads.pdf'
    # plt.savefig(filename, format='pdf', dpi=1200)
    plt.close(fig)


def calculate_image_gradients(img):
    # Extract the blue channel
    blue_channel = img[:, :, 0]  # OpenCV uses BGR format, so blue is at index 0

    # Calculate gradients using Sobel operator on the blue channel
    grad_x = cv2.Sobel(blue_channel, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x direction
    grad_y = cv2.Sobel(blue_channel, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y direction

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))  # Convert to uint8

    # Create a color map for better visualization
    color_map = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)

    # Overlay the gradient onto the original image
    overlay = cv2.addWeighted(img, 0.6, color_map, 0.4, 0)

    return overlay


def plot_Score_CAM(cam, train_data, layer, opt):
    img_int = 1.5
    dir_path = './results/' + opt.name + '/Score_CAM/'

    # overlayed activation map
    cmap = 'viridis'
    # stand-alone activation map
    cmap2 = 'coolwarm'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    yid = str(train_data['yid'] * 10)
    name = str(train_data['name'])
    print('timestep: ', name, 'rotation angle: ', yid)

    raw_img = np.uint8((np.transpose(train_data['raw_img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = Image.fromarray(np.uint8(raw_img)).convert('RGB')

    img = np.uint8((np.transpose(train_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(img)).convert('RGB')

    edge_img = calculate_image_gradients(img)

    heatmap, heatmap_on_image = apply_colormap_on_image(img, cam, cmap2)

    heatmap_on_image = np.array(heatmap_on_image)
    heatmap = np.array(heatmap)
    img = np.array(img)
    raw_img = np.array(raw_img)

    img = np.uint8(np.clip(img * img_int, a_min=0, a_max=255))
    raw_img = np.uint8(np.clip(raw_img * img_int, a_min=0, a_max=255))


    # Plotting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.rcParams['figure.constrained_layout.use'] = True

    # fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(15, 6.5))
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(10, 4.25))
    ax1, ax2, ax3 = axes.flatten()
    # Note: grads are in BGR format
    p1 = ax1.imshow(raw_img)
    p2 = ax2.imshow(heatmap / 255, vmin=0, vmax=1, cmap=cmap2)
    px = ax3.imshow(heatmap / 255, vmin=0, vmax=1)

    cbar1 = fig.colorbar(p1, ax=ax1, location='bottom')
    cbar2 = fig.colorbar(p2, ax=ax2, location='bottom')
    cbar3 = fig.colorbar(px, ax=ax3, location='bottom')
    cbar1.ax.tick_params(labelsize=16, which='major')
    cbar2.ax.tick_params(labelsize=16, which='major')
    cbar3.ax.tick_params(labelsize=16, which='major')
    px.set_cmap(cmap2)
    cbar1.remove()

    p3 = ax3.imshow(heatmap_on_image)
    cbar_ticks = [0.0, 0.5, 1.0]
    cbar2.set_ticks(cbar_ticks)
    cbar3.set_ticks(cbar_ticks)
    fig.execute_constrained_layout()

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    filename = dir_path + name + '_' + yid + '_' + layer + '_score_cam.pdf'
    plt.savefig(filename, format='pdf', dpi=1200)


    ''' separate plotting '''
    # input image
    fig, ax = plt.subplots(figsize=(3, 4))
    p = ax.imshow(edge_img)
    fig.execute_constrained_layout()
    ax.axis('off')
    filename = dir_path + name + '_' + yid + '_' + layer + '_raw_img_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)
    plt.close(fig)

    # heatmap
    fig, ax = plt.subplots(figsize=(3, 4))
    p = ax.imshow(heatmap / 255, vmin=0, vmax=1, cmap=cmap2)
    cbar = fig.colorbar(p, ax=ax, location='bottom')
    cbar.ax.tick_params(labelsize=16, which='major')
    cbar.set_ticks(cbar_ticks)
    fig.execute_constrained_layout()
    ax.axis('off')
    filename = dir_path + name + '_' + yid + '_' + layer + '_heatmap_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)
    plt.close(fig)

    # heatmap on image
    fig, ax = plt.subplots(figsize=(3, 4))
    p = ax.imshow(heatmap / 255, vmin=0, vmax=1)
    cbar = fig.colorbar(px, ax=ax3, location='bottom')
    cbar.ax.tick_params(labelsize=16, which='major')
    cbar.set_ticks(cbar_ticks)
    px = ax.imshow(heatmap_on_image)
    fig.execute_constrained_layout()
    ax.axis('off')
    filename = dir_path + name + '_' + yid + '_' + layer + '_heat_on_img_score_cam.svg'
    plt.savefig(filename, format='svg', dpi=1200)
    plt.close(fig)
