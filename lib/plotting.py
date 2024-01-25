import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.interpolate import griddata

def plot_contour(opt, samples, preds, labels, plane_dim, name, type, sample_name, dataset_type):
    sample_x = samples[0, 0, :].detach().cpu().numpy()
    sample_y = samples[0, 1, :].detach().cpu().numpy()
    sample_z = samples[0, 2, :].detach().cpu().numpy()
    sample = np.vstack((sample_x, sample_y, sample_z)).T
    #sample = samples.detach().cpu().numpy()
    label = labels.detach().cpu().numpy()
    pred = preds.detach().cpu().numpy()

    # interpolate point cloud to 2D-plane
    grid_res = complex(0, opt.resolution)
    X, Y, Z = np.mgrid[-128:128:grid_res, -28:228:grid_res, -0.01:0.01:1j]

    pred_interpn = griddata(sample, pred, (X,Y,Z), method='linear')
    pred_linear = griddata(sample, pred, (X,Y,Z), method='nearest')
    pred_interpn[np.isnan(pred_interpn)] = pred_linear[np.isnan(pred_interpn)]

    label_interpn = griddata(sample, label, (X,Y,Z), method='linear')
    label_linear = griddata(sample, label, (X,Y,Z), method='nearest')
    label_interpn[np.isnan(label_interpn)] = label_linear[np.isnan(label_interpn)]

    if plane_dim == 'x':
        var_plot = pred_interpn[0, :, :].T
        gt_plot = label_interpn[0, :, :].T
    if plane_dim == 'y':
        var_plot = pred_interpn[:, 0, :].T
        gt_plot = label_interpn[:, 0, :].T
    if plane_dim == 'z':
        var_plot = pred_interpn[:, :, 0].T
        gt_plot = label_interpn[:, :, 0].T

    err_plot = np.absolute(gt_plot - var_plot)

    if type == 'alpha':
        levels = np.linspace(0, 1.0, 10)
        colormap = 'RdBu_r'
    if type == 'vel':
        levels = np.linspace(-1.5, 1.5, 10)
        colormap = 'RdBu_r'
    if type == 'pres':
        levels = np.linspace(-1.2, 4.0, 10)
        colormap = 'viridis'

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    p1 = axs[0].contourf(var_plot, levels=levels, cmap=colormap)
    p2 = axs[1].contourf(gt_plot, levels=levels, cmap=colormap)
    p3 = axs[2].contourf(err_plot, cmap=colormap)
    axs[0].contourf(var_plot, levels=levels, cmap=colormap)
    axs[1].contourf(gt_plot, levels=levels, cmap=colormap)
    axs[2].contourf(err_plot, cmap=colormap)

    axs[0].set_ylabel('y', fontsize=16)
    axs[0].set_xlabel('x', fontsize=16)
    axs[1].set_xlabel('x', fontsize=16)
    axs[2].set_xlabel('x', fontsize=16)


    axs[0].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[1].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)
    axs[2].tick_params(axis ='both', which ='major', labelsize = 12, pad = 10)

    axs[0].set_title(r'{0} pred'.format(name), fontsize=16, y=1, pad=20 )
    axs[1].set_title(r'{0} gt'.format(name), fontsize=16, y=1, pad=20)
    axs[2].set_title(r'{0} err'.format(name), fontsize=16, y=1, pad=20)

    plt.tight_layout(w_pad=4.5)
    axs[0].set_aspect('equal', adjustable="datalim")
    axs[1].set_aspect('equal', adjustable="datalim")
    axs[2].set_aspect('equal', adjustable="datalim")
    fig.colorbar(p1, ax=axs[0], location='bottom')
    fig.colorbar(p2, ax=axs[1], location='bottom')
    fig.colorbar(p3, ax=axs[2], location='bottom')
    filename = 'results/'+ dataset_type + '_' + sample_name + '_' + name + '_pred.pdf'
    plt.savefig(filename)
    #plt.show()

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

    axs[0].set_title(r'$U_{pred}$', fontsize=16, y=1, pad=20 )
    axs[1].set_title(r'$U_{GT}$', fontsize=16, y=1, pad=20)
    axs[2].set_title(r'$U_{err}$', fontsize=16, y=1, pad=20)

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

    #cax = plt.axes([1.01, 0.15, 0.0091, 0.76])
    #cax = plt.axes()
    #cbar1 = plt.colorbar(p2)
    #cbar2 = plt.colorbar(p1)
    #cbar3 = plt.colorbar(p3)
    #cax.yaxis.tick_right()
    #cax.yaxis.set_label_position('right')
    #cbar1.set_label(r'$U$' , fontsize=12, fontweight='bold')
    #cbar1.ax.tick_params(labelsize=12)
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
    u_pred = pred[0, 2, :]
    u_pred = pred[0, 3, :]

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

def plot_backup():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_plot = num_samples // num_samples
    x = coords[0, :]
    y = coords[1, :]
    z = coords[2, :]
    a = sdf[::num_plot]

    plane_dim = z
    xp = x[0.0 < plane_dim]
    yp = y[0.0 < plane_dim]

    a = a[0.0 < plane_dim]
    u = u[0.0 < plane_dim]
    v = v[0.0 < plane_dim]
    w = w[0.0 < plane_dim]
    p = p[0.0 < plane_dim]
    zp = z[0.0 < plane_dim]

    plane_dim = zp
    xp = xp[5.0 > plane_dim]
    yp = yp[5.0 > plane_dim]

    a = a[5.0 > plane_dim]
    u = u[5.0 > plane_dim]
    v = v[5.0 > plane_dim]
    w = w[5.0 > plane_dim]
    p = p[5.0 > plane_dim]
    zp = zp[5.0 > plane_dim]

    ax.scatter(xp, yp, zp, s=10, c=p, cmap='viridis')
    # ax.scatter(xp, yp, zp, s=10, c=w, vmin=-1.5, vmax=1.5, cmap='bwr')
    # ax.scatter(xp, yp, zp, s=10, c=u, vmin=-0.1, vmax=0.1, cmap='bwr')
    # ax.scatter(x, y, z, s=10, c=sdf_plot, cmap='bwr')
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    # ax.set_ylim3d(ymin, yground)
    # ax.set_zlim3d(zmin, zmax)
    ax.set_box_aspect((1, 1, 1))
    plt.show()
    plt.savefig('pred.png')
    # plt.waitforbuttonpress()
